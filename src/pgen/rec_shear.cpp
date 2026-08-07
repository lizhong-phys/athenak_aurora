//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rec_shear.cpp
//! \brief Problem generator for reconnection in a double-layer (Harris) current sheet
//! embedded in an unstratified MHD shearing box.  AthenaK version based on the Athena++
//! problem generator rec_shear.cpp; radiation removed and EOS restricted to isothermal.
//! Structure follows pgen/tests/mri3d.cpp and pgen/gr_torus.cpp closely.
//!
//! Compile with '-D PROBLEM=rec_shear' to enroll as user-specific problem generator.
//!
//! Two current sheets sit at z = z_lower and z = z_upper.  Writing
//!     S(z) = tanh((z-z_upper)/jwidth) - tanh((z-z_lower)/jwidth) + 1
//! the field is
//!     Bx = bx2by*By0*S(z),   By = By0*S(z),   Bz = bx2by*bz2bx*By0
//! S(z) -> +1 at both z-boundaries and -> -1 midway between the sheets, so the setup is
//! periodic in z (double Harris sheet).  Density is set so total pressure is uniform:
//!     rho(z) = d0 + (By0/cs)^2 (1 + bx2by^2)/2 (1 - S(z)^2)
//!
//! Field configurations (ifield, following the mri3d convention):
//! - ifield = 1 - double Harris current sheet [default]
//! - ifield = 2 - uniform field and uniform density (diagnostic; no profile math)
//!
//! An optional zero-net-flux Bz perturbation localized on the sheets (beta_mri > 0) is
//! added as the discrete curl of a vector potential A_y, so div(B) = 0 to machine
//! precision.  Random velocity perturbations are added as in HGB.
//!
//! CAVEAT: this file uses only constructs that appear in pgen/tests/mri3d.cpp or
//! pgen/gr_torus.cpp, with ONE exception -- user_srcs_func.  Neither example enrols a
//! user source term (they use user_bcs_func and user_hist_func only), so the vertical
//! gravity below is the one piece of this pgen with no validated precedent.  Set
//! <problem>/grav = 0.0 to disable it entirely: user_srcs is then never set and
//! RecShearSrcTerms is never called.
//!
//! Optional user source terms (enrolled only when their parameters are non-zero):
//! - vertical tidal gravity about the nearest sheet, set by <problem>/grav.  Note this
//!   replaces <shearing_box>/stratified, which is centred on z = 0 instead of on the
//!   sheets, so 'stratified' must be left false.
//! - density sinks in cos^2 windows of width <problem>/sinkwidth at the midplane and the
//!   two z-boundaries, with e-folding time <problem>/tau_sink.
//!
//! NOTE: AthenaK always applies orbital advection when a <shearing_box> block is present
//! in 3D, so the background shear -q*Omega0*x is carried by the remap and must NOT be
//! added to the initial y-momentum.
//!
//! REFERENCE: Hawley, J. F., Gammie, C.F. & Balbus, S. A., ApJ 440, 742-763 (1995).

// C headers
#include <algorithm>

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "shearing_box/shearing_box.hpp"
#include "pgen/pgen.hpp"

#include <Kokkos_Random.hpp>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// Useful container for physical parameters of the current sheets.  Copied by value into
// every kernel, following the pattern used by pgen/gr_torus.cpp.
struct rec_shear_pgen {
  Real d0, iso_cs;                  // background density and sound speed
  Real by0, bz0, apot;              // field amplitudes; apot = bz0/kx
  Real bx2by, bz2bx;                // field ratios
  Real kx;                          // x-wavenumber of the seed perturbation
  Real jwidth;                      // current sheet thickness
  Real z_mid, z_min, z_max;         // vertical coordinate parameters
  Real z_lower, z_upper;            // locations of the two current sheets
  Real dcoef;                       // amplitude of the density enhancement
  Real amp_v;                       // amplitude of random velocity perturbations
  Real grav;                        // strength of vertical tidal gravity
  Real sinkwidth, tau_sink;         // sink parameters
  Real dfloor;                      // density floor used by the sink
  Real omega0;                      // orbital frequency
  int ifield;                       // field configuration
};

rec_shear_pgen rshear;

//----------------------------------------------------------------------------------------
//! \fn HarrisS
//! \brief S(z) = tanh((z-z_upper)/jwidth) - tanh((z-z_lower)/jwidth) + 1

KOKKOS_INLINE_FUNCTION
static Real HarrisS(const struct rec_shear_pgen pgen, Real z) {
  return tanh((z - pgen.z_upper)/pgen.jwidth)
       - tanh((z - pgen.z_lower)/pgen.jwidth) + 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn SheetGauss
//! \brief Gaussian envelope peaked on each of the two current sheets, used by the vector
//! potential A_y = -apot*cos(kx*x)*SheetGauss(z) that seeds the Bz perturbation

KOKKOS_INLINE_FUNCTION
static Real SheetGauss(const struct rec_shear_pgen pgen, Real z) {
  return exp(-0.5*SQR((z - pgen.z_upper)/pgen.jwidth))
       + exp(-0.5*SQR((z - pgen.z_lower)/pgen.jwidth));
}
}  // namespace

// prototypes for user-defined history and source-term functions
void RecShearHistory(HistoryData *pdata, Mesh *pm);
void RecShearSrcTerms(Mesh *pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets a double Harris current sheet in an unstratified MHD shearing box

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // First, do some error checks
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works in 3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->psbox_u == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " <<__LINE__ << std::endl
                << "Shearing box not enabled for rec_shear problem, likely missing "
                << "<shearing_box> block in input file" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works in mhd" << std::endl;
    exit(EXIT_FAILURE);
  }

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works with isothermal EOS"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->psbox_u->is_stratified) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear supplies its own vertical gravity about each current sheet; "
              << "set <shearing_box>/stratified = false and use <problem>/grav"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // vertical geometry of the two current sheets
  rshear.z_min   = pmy_mesh_->mesh_size.x3min;
  rshear.z_max   = pmy_mesh_->mesh_size.x3max;
  rshear.z_mid   = 0.5*(rshear.z_min + rshear.z_max);
  rshear.z_lower = 0.5*(rshear.z_mid + rshear.z_min);
  rshear.z_upper = 0.5*(rshear.z_mid + rshear.z_max);

  // source-term parameters, and enroll user functions.  These are set on restarts too.
  rshear.grav      = pin->GetOrAddReal("problem","grav",0.0);
  rshear.sinkwidth = pin->GetOrAddReal("problem","sinkwidth",0.0);
  rshear.tau_sink  = pin->GetOrAddReal("problem","tau_sink",1.0);
  rshear.dfloor    = pin->GetOrAddReal("mhd","dfloor",1.0e-10);
  rshear.omega0    = pmbp->pmhd->psbox_u->omega0;

  user_hist_func = RecShearHistory;
  if ((rshear.grav != 0.0) || (rshear.sinkwidth > 0.0)) {
    user_srcs = true;
    user_srcs_func = RecShearSrcTerms;
  }
  if (restart) return;

  // initialize problem variables
  Real amp    = pin->GetReal("problem","amp");
  Real beta   = pin->GetReal("problem","beta");
  int nwx     = pin->GetOrAddInteger("problem","nwx",1);
  rshear.ifield = pin->GetOrAddInteger("problem","ifield",1);
  rshear.d0     = pin->GetOrAddReal("problem","d0",1.0);
  rshear.jwidth = pin->GetOrAddReal("problem","jwidth",0.1);
  rshear.bx2by  = pin->GetOrAddReal("problem","bx2by",0.02);
  rshear.bz2bx  = pin->GetOrAddReal("problem","bz2bx",0.01);
  // beta_mri sets the seed Bz perturbation; <= 0 (default) means no perturbation
  Real beta_mri = pin->GetOrAddReal("problem","beta_mri",-1.0);
  // optional removal of the mean momentum of the random perturbations.  Off by default:
  // it costs an extra reduction and the perturbations are tiny.
  bool zero_net_mom = pin->GetOrAddBoolean("problem","zero_net_momentum",false);

  rshear.iso_cs = eos.iso_cs;
  Real p0 = rshear.d0*SQR(rshear.iso_cs);
  rshear.by0 = std::sqrt(2.0*p0/beta);
  rshear.bz0 = (beta_mri > 0.0) ? std::sqrt(2.0*p0/beta_mri) : 0.0;
  if (rshear.ifield != 1) rshear.bz0 = 0.0;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  rshear.kx = 2.0*(M_PI/x1size)*(static_cast<Real>(nwx));
  // amplitude of the vector potential; guard against nwx = 0 producing a 0/0
  rshear.apot = (rshear.kx != 0.0) ? (rshear.bz0/rshear.kx) : 0.0;

  // density enhancement that keeps the total pressure uniform across the sheets
  rshear.dcoef = SQR(rshear.by0/rshear.iso_cs)*(1.0 + SQR(rshear.bx2by))/2.0;

  // Following HGB, the perturbations to V/Cs are (1/5)amp/sqrt(gamma); this reproduces
  // the amplitude used by the Athena++ version of this problem, with gamma = 1.
  rshear.amp_v = pin->GetOrAddReal("problem","vamp",(0.4/std::sqrt(3.0))*1.0e-3)*amp;

  // capture variables for kernel
  auto rsp = rshear;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;

  // Initialize magnetic field first, so entire arrays are initialized before they are
  // used in the next loop.  For a 3D shearing box B1=Bx, B2=By, B3=Bz.
  // ifield = 1 - double Harris current sheet [default]
  // ifield = 2 - uniform field
  auto b0 = pmbp->pmhd->b0;
  par_for("rec_shear_b", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;

    Real x3v  = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x1f  = LeftEdgeX(i-is,   nx1, x1min, x1max);
    Real x1fp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x3f  = LeftEdgeX(k-ks,   nx3, x3min, x3max);
    Real x3fp = LeftEdgeX(k+1-ks, nx3, x3min, x3max);

    // S(z) reverses the field across each sheet; identically 1 for a uniform field
    Real sprof = 1.0;
    if (rsp.ifield == 1) {
      sprof = HarrisS(rsp, x3v);
    }

    // equilibrium field
    Real bxeq = rsp.by0*rsp.bx2by*sprof;
    Real byeq = rsp.by0*sprof;
    Real bzeq = rsp.by0*rsp.bx2by*rsp.bz2bx;

    // seed perturbation as the discrete curl of A_y = -apot*cos(kx*x)*SheetGauss(z),
    // which gives Bx = -dA_y/dz and Bz = +dA_y/dx and so is divergence free
    Real gf = 0.0, gfp = 0.0;
    if (rsp.bz0 != 0.0) {
      gf  = SheetGauss(rsp, x3f);
      gfp = SheetGauss(rsp, x3fp);
    }
    Real dcos = cos(rsp.kx*x1fp) - cos(rsp.kx*x1f);
    Real &dx1 = size.d_view(m).dx1;
    Real &dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = bxeq + rsp.apot*cos(rsp.kx*x1f)*(gfp - gf)/dx3;
    b0.x2f(m,k,j,i) = byeq;
    b0.x3f(m,k,j,i) = bzeq - rsp.apot*gf*dcos/dx1;
    if (i==ie) {
      b0.x1f(m,k,j,ie+1) = bxeq + rsp.apot*cos(rsp.kx*x1fp)*(gfp - gf)/dx3;
    }
    if (j==je) {
      b0.x2f(m,k,je+1,i) = byeq;
    }
    if (k==ke) {
      b0.x3f(m,ke+1,j,i) = bzeq - rsp.apot*gfp*dcos/dx1;
    }
  });

  // Initialize conserved variables.  Density balances the magnetic pressure of the
  // sheets; random perturbations to the velocity seed reconnection as in HGB.
  auto &u0 = pmbp->pmhd->u0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

  par_for("rec_shear_u", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    Real rval;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Set density, uniform in total (gas + magnetic) pressure
    Real rd = rsp.d0;
    if (rsp.ifield == 1) {
      Real sprof = HarrisS(rsp, x3v);
      rd = rsp.d0 + rsp.dcoef*(1.0 - SQR(sprof));
    }
    u0(m,IDN,k,j,i) = rd;

    // Set momenta (with random perturbations to velocity).  NOTE no -q*Omega0*x term:
    // the background shear is carried by AthenaK's orbital advection.
    rval = rsp.amp_v*(rand_gen.frand() - 0.5);
    u0(m,IM1,k,j,i) = rd*rval;
    rval = rsp.amp_v*(rand_gen.frand() - 0.5);
    u0(m,IM2,k,j,i) = rd*rval;
    rval = rsp.amp_v*(rand_gen.frand() - 0.5);
    u0(m,IM3,k,j,i) = rd*rval;

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  // Optionally subtract the mean momentum of the random perturbations.  Uses the same
  // array_sum::GlobalSum reducer as the history outputs.
  if (zero_net_mom && (rshear.amp_v != 0.0)) {
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;
    array_sum::GlobalSum msum;
    Kokkos::parallel_reduce("rec_shear_mom", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
      // compute m,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      array_sum::GlobalSum mvars;
      mvars.the_array[0] = u0(m,IM1,k,j,i);
      mvars.the_array[1] = u0(m,IM2,k,j,i);
      mvars.the_array[2] = u0(m,IM3,k,j,i);
      for (int n=3; n<NHISTORY_VARIABLES; ++n) {
        mvars.the_array[n] = 0.0;
      }
      mb_sum += mvars;
    }, Kokkos::Sum<array_sum::GlobalSum>(msum));
    Kokkos::fence();

    Real m1tot = msum.the_array[0];
    Real m2tot = msum.the_array[1];
    Real m3tot = msum.the_array[2];
#if MPI_PARALLEL_ENABLED
    // sum the net momentum over all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, &m1tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &m2tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &m3tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Real ncells = static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                 *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                 *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
    Real mavg1 = m1tot/ncells;
    Real mavg2 = m2tot/ncells;
    Real mavg3 = m3tot/ncells;

    par_for("rec_shear_dmom", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IM1,k,j,i) -= mavg1;
      u0(m,IM2,k,j,i) -= mavg2;
      u0(m,IM3,k,j,i) -= mavg3;
    });
  }

  std::cout << std::endl
    << "--- rec_shear: double Harris sheet in a shearing box ---" << std::endl
    << "  ifield           = " << rshear.ifield << std::endl
    << "  iso_cs           = " << rshear.iso_cs << std::endl
    << "  d0, p0           = " << rshear.d0 << ", " << p0 << std::endl
    << "  By0 (beta)       = " << rshear.by0 << " (" << beta << ")" << std::endl
    << "  Bz0 (beta_mri)   = " << rshear.bz0 << " (" << beta_mri << ")" << std::endl
    << "  bx2by, bz2bx     = " << rshear.bx2by << ", " << rshear.bz2bx << std::endl
    << "  jwidth           = " << rshear.jwidth << std::endl
    << "  sheets at z      = " << rshear.z_lower << ", " << rshear.z_upper << std::endl
    << "  rho at sheets    = " << rshear.d0 + rshear.dcoef << std::endl
    << "  vpert amplitude  = " << rshear.amp_v << std::endl
    << "  grav             = " << rshear.grav << std::endl
    << "  sinkwidth        = " << rshear.sinkwidth << ", tau_sink = "
    << rshear.tau_sink << std::endl << std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RecShearSrcTerms()
//! \brief Vertical tidal gravity about the nearest current sheet, plus optional density
//! sinks at the midplane and the two vertical boundaries.  Source terms are computed from
//! the primitives (w0) and applied to the conserved variables (u0).

void RecShearSrcTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto rsp = rshear;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;

  auto &u0 = pmbp->pmhd->u0;
  auto &w0 = pmbp->pmhd->w0;

  Real gcoef = rsp.grav*bdt*SQR(rsp.omega0);
  Real zsink_l = rsp.z_mid - 0.5*rsp.sinkwidth;
  Real zsink_u = rsp.z_mid + 0.5*rsp.sinkwidth;
  Real zbot_u  = rsp.z_min + 0.5*rsp.sinkwidth;
  Real ztop_l  = rsp.z_max - 0.5*rsp.sinkwidth;

  par_for("rec_shear_src", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real den = w0(m,IDN,k,j,i);

    // vertical tidal gravity toward the nearest current sheet
    if (rsp.grav != 0.0) {
      Real z0 = (x3v > rsp.z_mid) ? rsp.z_upper : rsp.z_lower;
      u0(m,IM3,k,j,i) -= gcoef*den*(x3v - z0);
    }

    // density sink in cos^2 windows at the midplane and the vertical boundaries
    if (rsp.sinkwidth > 0.0) {
      Real window = 0.0;
      if ((x3v >= zsink_l) && (x3v <= zsink_u)) {
        window = SQR(cos(M_PI*(x3v - rsp.z_mid)/rsp.sinkwidth));
      } else if ((x3v >= rsp.z_min) && (x3v <= zbot_u)) {
        window = SQR(cos(0.5*M_PI*(x3v - rsp.z_min)/rsp.sinkwidth));
      } else if ((x3v >= ztop_l) && (x3v <= rsp.z_max)) {
        window = SQR(cos(0.5*M_PI*(x3v - rsp.z_max)/rsp.sinkwidth));
      }
      if (window > 0.0) {
        // exponential decay factor, stable for any bdt/tau_sink ratio
        Real fac = exp(-window*bdt/rsp.tau_sink);
        Real dden = fmax(den*fac, rsp.dfloor) - den;
        u0(m,IDN,k,j,i) += dden;
        u0(m,IM1,k,j,i) += dden*w0(m,IVX,k,j,i);
        u0(m,IM2,k,j,i) += dden*w0(m,IVY,k,j,i);
        u0(m,IM3,k,j,i) += dden*w0(m,IVZ,k,j,i);
      }
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RecShearHistory()
//! \brief Compute and store history data.  Adds Reynolds and Maxwell stress and net
//! magnetic flux to usual list of MHD history variables.  Follows MRIHistory in mri3d.cpp

void RecShearHistory(HistoryData *pdata, Mesh *pm) {
  int &nmhd_ = pm->pmb_pack->pmhd->nmhd;

  // set number of and names of history variables for isothermal mhd
  pdata->nhist = 15;
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  pdata->label[nmhd_  ] = "1-KE";
  pdata->label[nmhd_+1] = "2-KE";
  pdata->label[nmhd_+2] = "3-KE";
  pdata->label[nmhd_+3] = "1-ME";
  pdata->label[nmhd_+4] = "2-ME";
  pdata->label[nmhd_+5] = "3-ME";
  pdata->label[nmhd_+6] = "1-bcc";
  pdata->label[nmhd_+7] = "2-bcc";
  pdata->label[nmhd_+8] = "3-bcc";
  pdata->label[nmhd_+9] = "dVxVy";
  pdata->label[nmhd_+10] = "dBxBy";

  // capture class variables for kernel
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &bx1f = pm->pmb_pack->pmhd->b0.x1f;
  auto &bx2f = pm->pmb_pack->pmhd->b0.x2f;
  auto &bx3f = pm->pmb_pack->pmhd->b0.x3f;
  auto &bcc = pm->pmb_pack->pmhd->bcc0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // MHD conserved variables:
    array_sum::GlobalSum hvars;
    hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
    hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
    hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
    hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);

    // MHD KE
    hvars.the_array[nmhd_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

    // MHD ME
    hvars.the_array[nmhd_+3] = vol*0.25*(SQR(bx1f(m,k,j,i+1)) + SQR(bx1f(m,k,j,i)));
    hvars.the_array[nmhd_+4] = vol*0.25*(SQR(bx2f(m,k,j+1,i)) + SQR(bx2f(m,k,j,i)));
    hvars.the_array[nmhd_+5] = vol*0.25*(SQR(bx3f(m,k+1,j,i)) + SQR(bx3f(m,k,j,i)));

    // net B fluxes
    hvars.the_array[nmhd_+6] = vol*bcc(m,IBX,k,j,i);
    hvars.the_array[nmhd_+7] = vol*bcc(m,IBY,k,j,i);
    hvars.the_array[nmhd_+8] = vol*bcc(m,IBZ,k,j,i);

    // Reynolds and Maxwell stresses.  With orbital advection the y-momentum is already
    // the deviation from the background shear, so no vshear correction is needed.
    hvars.the_array[nmhd_+9] = vol*u0_(m,IM1,k,j,i)*u0_(m,IM2,k,j,i)/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+10] = -vol*bcc(m,IBX,k,j,i)*bcc(m,IBY,k,j,i);

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
  return;
}
