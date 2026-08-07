//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rec_shear.cpp
//! \brief Problem generator for a double-layer (Harris) current sheet in an unstratified
//! MHD shearing box.
//!
//! STAGE 2.  The stage-1 core (a minimal edit of pgen/tests/mri3d.cpp, verified to run
//! on Frontier) with the four optional features restored.  EVERY optional feature is OFF
//! by default and is switched on purely from the input file, so a single build lets you
//! bisect one feature per job with no recompilation:
//!
//!   <problem>/user_hist = true      -> enables RecShearHistory (a verbatim copy of
//!                                      MRIHistory from mri3d.cpp, only renamed).
//!                                      Also needs <output1>/user_hist_only = true.
//!   <problem>/grav      = 1.0       -> enables the vertical tidal gravity source term.
//!                                      This is the one hook neither mri3d nor gr_torus
//!                                      uses, so it is the least attested feature here.
//!   <problem>/sinkwidth = 0.2       -> enables the density sinks (same source term).
//!   <problem>/zero_net_momentum = true -> enables the mean-momentum reduction.
//!   <problem>/beta_mri  = 1.0e4     -> enables the divergence-free Bz seed perturbation.
//!
//! With all of them at their defaults this file behaves exactly like stage 1.
//!
//! Two current sheets sit at z = z_lower and z = z_upper.  Writing
//!     S(z) = tanh((z-z_upper)/jwidth) - tanh((z-z_lower)/jwidth) + 1
//! the field is
//!     Bx = bx2by*By0*S(z),   By = By0*S(z),   Bz = bx2by*bz2bx*By0
//! S(z) -> +1 at both z-boundaries and -> -1 midway between the sheets, so the setup is
//! periodic in z.  Density balances the magnetic pressure so total pressure is uniform:
//!     rho(z) = d0 + (By0/cs)^2 (1 + bx2by^2)/2 (1 - S(z)^2)
//! Random perturbations to the velocity are added in the initial conditions, as in HGB.
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
// Container for the parameters the source-term function needs, copied by value into the
// kernel.  Same pattern as struct torus_pgen in pgen/gr_torus.cpp.
struct rec_shear_pgen {
  Real z_mid, z_min, z_max;    // vertical coordinate parameters
  Real z_lower, z_upper;       // locations of the two current sheets
  Real grav;                   // strength of vertical tidal gravity
  Real sinkwidth, tau_sink;    // sink parameters
  Real dfloor;                 // density floor used by the sink
  Real omega0;                 // orbital frequency
};

rec_shear_pgen rshear;
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

  // Locations of the two current sheets, and the source-term parameters.  These are set
  // on restarts too, so they must be read before the restart guard below.
  rshear.z_min   = pmy_mesh_->mesh_size.x3min;
  rshear.z_max   = pmy_mesh_->mesh_size.x3max;
  rshear.z_mid   = 0.5*(rshear.z_min + rshear.z_max);
  rshear.z_lower = 0.5*(rshear.z_mid + rshear.z_min);
  rshear.z_upper = 0.5*(rshear.z_mid + rshear.z_max);
  rshear.grav      = pin->GetOrAddReal("problem","grav",0.0);
  rshear.sinkwidth = pin->GetOrAddReal("problem","sinkwidth",0.0);
  rshear.tau_sink  = pin->GetOrAddReal("problem","tau_sink",1.0);
  rshear.dfloor    = pin->GetOrAddReal("mhd","dfloor",1.0e-10);
  rshear.omega0    = pmbp->pmhd->psbox_u->omega0;

  // enroll user history function (only called when <problem>/user_hist = true)
  user_hist_func = RecShearHistory;
  // enroll user source terms only if they are actually switched on
  if ((rshear.grav != 0.0) || (rshear.sinkwidth > 0.0)) {
    user_srcs = true;
    user_srcs_func = RecShearSrcTerms;
  }
  if (restart) return;

  // initialize problem variables
  Real amp    = pin->GetReal("problem","amp");
  Real beta   = pin->GetReal("problem","beta");
  Real d0     = pin->GetOrAddReal("problem","d0",1.0);
  Real jwidth = pin->GetOrAddReal("problem","jwidth",0.1);
  Real bx2by  = pin->GetOrAddReal("problem","bx2by",0.02);
  Real bz2bx  = pin->GetOrAddReal("problem","bz2bx",0.01);
  // beta_mri <= 0 (default) disables the Bz seed perturbation entirely
  Real beta_mri = pin->GetOrAddReal("problem","beta_mri",-1.0);
  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  bool zero_net_mom = pin->GetOrAddBoolean("problem","zero_net_momentum",false);

  // background density, pressure, and magnetic field
  Real p0 = d0*SQR(eos.iso_cs);
  Real binit = std::sqrt(2.0*p0/beta);
  Real bpert = (beta_mri > 0.0) ? std::sqrt(2.0*p0/beta_mri) : 0.0;
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real kx = 2.0*(M_PI/x1size)*(static_cast<Real>(nwx));
  // amplitude of the vector potential A_y = -apot*cos(kx*x)*[gaussians on each sheet]
  Real apot = (kx != 0.0) ? (bpert/kx) : 0.0;

  // locations of the two current sheets, at the quarter points of the z-domain
  Real zmin = pmy_mesh_->mesh_size.x3min;
  Real zmax = pmy_mesh_->mesh_size.x3max;
  Real zmid = 0.5*(zmin + zmax);
  Real zlwr = 0.5*(zmid + zmin);
  Real zupr = 0.5*(zmid + zmax);

  // density enhancement inside the sheets that keeps total pressure uniform
  Real dcoef = SQR(binit/eos.iso_cs)*(1.0 + SQR(bx2by))/2.0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Initialize magnetic field first, so entire arrays are initialized before adding
  // magnetic energy to conserved variables in next loop.  For 3D shearing box
  // B1=Bx, B2=By, B3=Bz.  Bx and By depend on z only and Bz is uniform, so div(B)=0.
  auto b0 = pmbp->pmhd->b0;
  par_for("rec_shear", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real sprof = tanh((x3v - zupr)/jwidth) - tanh((x3v - zlwr)/jwidth) + 1.0;
    Real bxv = binit*bx2by*sprof;
    Real byv = binit*sprof;
    Real bzv = binit*bx2by*bz2bx;

    // Optional seed perturbation: the discrete curl of
    //   A_y = -apot*cos(kx*x)*[exp(-((z-zupr)/jw)^2/2) + exp(-((z-zlwr)/jw)^2/2)]
    // which gives Bx = -dA_y/dz and Bz = +dA_y/dx, so div(B) stays zero.  With
    // apot = 0 (the default) every term below is identically zero.
    Real bxp = 0.0, bzp = 0.0, bxp1 = 0.0, bzp1 = 0.0;
    if (apot != 0.0) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1f  = LeftEdgeX(i-is,   nx1, x1min, x1max);
      Real x1fp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real x3f  = LeftEdgeX(k-ks,   nx3, x3min, x3max);
      Real x3fp = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
      Real gf  = exp(-0.5*SQR((x3f  - zupr)/jwidth))
               + exp(-0.5*SQR((x3f  - zlwr)/jwidth));
      Real gfp = exp(-0.5*SQR((x3fp - zupr)/jwidth))
               + exp(-0.5*SQR((x3fp - zlwr)/jwidth));
      Real dcos = cos(kx*x1fp) - cos(kx*x1f);
      Real &dx1 = size.d_view(m).dx1;
      Real &dx3 = size.d_view(m).dx3;
      bxp  = apot*cos(kx*x1f)*(gfp - gf)/dx3;
      bxp1 = apot*cos(kx*x1fp)*(gfp - gf)/dx3;
      bzp  = -apot*gf*dcos/dx1;
      bzp1 = -apot*gfp*dcos/dx1;
    }

    b0.x1f(m,k,j,i) = bxv + bxp;
    b0.x2f(m,k,j,i) = byv;
    b0.x3f(m,k,j,i) = bzv + bzp;
    if (i==ie) b0.x1f(m,k,j,ie+1) = bxv + bxp1;
    if (j==je) b0.x2f(m,k,je+1,i) = byv;
    if (k==ke) b0.x3f(m,ke+1,j,i) = bzv + bzp1;
  });

  // Initialize conserved variables.  Density balances the magnetic pressure of the
  // sheets; random perturbations to the velocity seed reconnection.
  auto &u0 = pmbp->pmhd->u0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

  par_for("rec_shear-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    Real rval;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Set density, uniform in total (gas + magnetic) pressure
    Real sprof = tanh((x3v - zupr)/jwidth) - tanh((x3v - zlwr)/jwidth) + 1.0;
    Real rd = d0 + dcoef*(1.0 - SQR(sprof));
    u0(m,IDN,k,j,i) = rd;

    // Set momenta (with random perturbations to velocity).  NOTE no -q*Omega0*x term:
    // the background shear is carried by AthenaK's orbital advection.
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM1,k,j,i) = rd*rval;
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM2,k,j,i) = rd*rval;
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM3,k,j,i) = rd*rval;

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  // Optionally subtract the mean momentum of the random perturbations.  Uses the same
  // array_sum::GlobalSum reducer as the history output in mri3d.cpp, and the same
  // per-scalar MPI_Allreduce idiom as gr_torus.cpp.
  if (zero_net_mom) {
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;
    array_sum::GlobalSum msum;
    Kokkos::parallel_reduce("rec_shear-mom", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
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

    par_for("rec_shear-dmom", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IM1,k,j,i) -= mavg1;
      u0(m,IM2,k,j,i) -= mavg2;
      u0(m,IM3,k,j,i) -= mavg3;
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RecShearSrcTerms()
//! \brief Vertical tidal gravity about the nearest current sheet, plus optional density
//! sinks at the midplane and the two vertical boundaries.  Written from the primitives
//! (w0) and applied to the conserved variables (u0), as AthenaK requires.
//! NOTE: user_srcs_func is the one hook neither mri3d.cpp nor gr_torus.cpp uses.

void RecShearSrcTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto rsp = rshear;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  auto &u0 = pmbp->pmhd->u0;
  auto &w0 = pmbp->pmhd->w0;

  Real gcoef = rsp.grav*bdt*SQR(rsp.omega0);
  Real zsink_l = rsp.z_mid - 0.5*rsp.sinkwidth;
  Real zsink_u = rsp.z_mid + 0.5*rsp.sinkwidth;
  Real zbot_u  = rsp.z_min + 0.5*rsp.sinkwidth;
  Real ztop_l  = rsp.z_max - 0.5*rsp.sinkwidth;

  par_for("rec_shear-src", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
//! \brief VERBATIM copy of MRIHistory() from pgen/tests/mri3d.cpp -- only the function
//! name differs.  Adds Reynolds and Maxwell stress and net magnetic flux to the usual
//! list of MHD history variables.  Only called when <problem>/user_hist = true.

void RecShearHistory(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->pmhd->peos->eos_data;
  int &nmhd_ = pm->pmb_pack->pmhd->nmhd;

  // set number of and names of history variables for mhd
  if (eos_data.is_ideal) {
    pdata->nhist = 16;
  } else {
    pdata->nhist = 15;
  }
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  if (eos_data.is_ideal) {
    pdata->label[IEN] = "tot-E";
  }
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

  // capture class variabels for kernel
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
    if (eos_data.is_ideal) {
      hvars.the_array[IEN] = vol*u0_(m,IEN,k,j,i);
    }

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

    // Reynolds and Maxwell stresses
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
