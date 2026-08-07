//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rec_shear.cpp
//! \brief Problem generator for reconnection in a double-layer (Harris) current sheet
//!        embedded in an unstratified MHD shearing box.
//!
//! AthenaK port of the Athena++ problem generator rec_shear.cpp.  Radiation removed and
//! the EOS restricted to isothermal, so the setup reduces to:
//!
//!   * Two horizontal current sheets at z = z_lower and z = z_upper, across which the
//!     toroidal field By reverses.  Writing
//!
//!         S(z) = tanh((z-z_upper)/jwidth) - tanh((z-z_lower)/jwidth) + 1
//!
//!     the equilibrium field is
//!
//!         Bx = bx2by * By0 * S(z)          (in-plane guide field)
//!         By = By0 * S(z)                  (reversing component)
//!         Bz = bx2by * bz2bx * By0         (uniform vertical field)
//!
//!     S(z) -> +1 near both z-boundaries and -> -1 midway between the sheets, so the
//!     configuration is periodic in z (double Harris sheet).
//!
//!   * Density chosen so total pressure P + B^2/2 is uniform (isothermal EOS):
//!
//!         rho(z) = d0 + (By0/cs)^2 * (1 + bx2by^2)/2 * (1 - S(z)^2)
//!
//!   * Optional divergence-free Bz perturbation localized on the two sheets, built as the
//!     discrete curl of a vector potential A_y so that div(B) = 0 to machine precision on
//!     the staggered mesh.
//!
//!   * Optional small random velocity perturbations with the mean momentum removed.
//!
//! Two user source terms (both optional, enrolled only when their parameters are set):
//!
//!   * Vertical tidal gravity about the nearest current sheet,
//!         dM3/dt = -grav * rho * Omega0^2 * (z - z0),  z0 = z_lower or z_upper
//!     This replaces the built-in <shearing_box>/stratified gravity, which is centred on
//!     z = 0 rather than on the sheets, so `stratified` must be left false.
//!
//!   * Density sinks in cos^2 windows of width `sinkwidth` at the midplane and at the two
//!     z-boundaries, with e-folding time `tau_sink`.
//!
//! NOTE ON ORBITAL ADVECTION: AthenaK always applies orbital advection when a
//! <shearing_box> block is present in 3D, i.e. the background shear -q*Omega0*x is
//! carried by the remap and must NOT be added to the initial y-momentum.
//!
//! GPU PORTABILITY NOTES.  This file deliberately avoids several constructs that are
//! legal but behave differently (or are simply untested) on HIP/CUDA backends:
//!   * No __host__ __device__ helper functions in an anonymous namespace -- all the
//!     profile math is written inline inside the kernels.
//!   * All math uses Kokkos::-qualified functions (Kokkos::tanh, Kokkos::exp, ...) so
//!     device overloads are always selected, and M_PI is passed in as a captured Real
//!     rather than used as a macro inside device code.
//!   * Every scalar captured by a kernel is a by-value copy: no references to host
//!     structs (int &is = indcs.is) and no references bound into device Views.
//!   * Kokkos::fence() after every kernel launched here, so no object that a kernel
//!     captured (in particular the random-number pool, whose Views are reference counted
//!     and would otherwise be released while an async kernel is still using them) can go
//!     out of scope while that kernel is in flight.
//!   * Only a single custom reducer (array_sum::GlobalSum, as used by AthenaK's history
//!     outputs) -- never Kokkos' combined multi-reducer path.
//!
//! BISECTION SWITCHES.  Two input parameters exist purely to localize problems without
//! recompiling:
//!   <problem>/uniform_init = true   -- uniform density and uniform field, skipping all
//!                                      tanh/exp profile math and the vector potential
//!   <problem>/amp          = 0.0    -- no random perturbations, so the random pool is
//!                                      never constructed and its kernel never runs
//!
//! REFERENCE: Hawley, J. F., Gammie, C.F. & Balbus, S. A., ApJ 440, 742-763 (1995).

// C headers

// C++ headers
#include <algorithm>  // max
#include <cmath>      // sqrt()
#include <cstdlib>    // exit(), EXIT_FAILURE
#include <iostream>   // endl

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
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

// prototypes for user-defined source-term and history functions
void RecShearSrcTerms(Mesh *pm, const Real bdt);
void RecShearHistory(HistoryData *pdata, Mesh *pm);

namespace {
// Parameters shared between the pgen and the user source/history functions.  Read on the
// host only; always copied into local variables before being captured by a kernel.
Real z_mid_, z_min_, z_max_;         // vertical coordinate parameters
Real z_lower_, z_upper_;             // locations of the two current sheets
Real grav_;                          // strength of vertical tidal gravity
Real sinkwidth_, tau_sink_;          // sink parameters
Real dfloor_;                        // density floor used by the sink
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets up a double Harris current sheet in an unstratified MHD shearing box.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // ---- error checks ------------------------------------------------------------------
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator requires a 3D mesh" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator requires MHD (add <mhd> block)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->psbox_u == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shearing box not enabled for rec_shear problem, likely missing "
              << "<shearing_box> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only supports an isothermal EOS; set "
              << "<mhd>/eos = isothermal" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->psbox_u->is_stratified) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear supplies its own vertical gravity about each current sheet; "
              << "set <shearing_box>/stratified = false and use <problem>/grav"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- vertical geometry of the two current sheets -----------------------------------
  z_min_   = pmy_mesh_->mesh_size.x3min;
  z_max_   = pmy_mesh_->mesh_size.x3max;
  z_mid_   = 0.5*(z_min_ + z_max_);
  z_lower_ = 0.5*(z_mid_ + z_min_);
  z_upper_ = 0.5*(z_mid_ + z_max_);

  // ---- source-term parameters (must be set on restarts too) --------------------------
  grav_      = pin->GetOrAddReal("problem", "grav", 0.0);
  sinkwidth_ = pin->GetOrAddReal("problem", "sinkwidth", 0.0);
  tau_sink_  = pin->GetOrAddReal("problem", "tau_sink", 1.0);
  dfloor_    = eos.dfloor;   // set by <mhd>/dfloor

  user_hist_func = RecShearHistory;
  if ((grav_ != 0.0) || (sinkwidth_ > 0.0)) {
    user_srcs = true;
    user_srcs_func = RecShearSrcTerms;
  }
  if (restart) return;

  // ---- problem parameters ------------------------------------------------------------
  Real d0     = pin->GetOrAddReal("problem", "d0", 1.0);
  Real jwidth = pin->GetOrAddReal("problem", "jwidth", 0.1);
  Real bx2by  = pin->GetOrAddReal("problem", "bx2by", 0.02);
  Real bz2bx  = pin->GetOrAddReal("problem", "bz2bx", 0.01);
  Real amp    = pin->GetReal("problem", "amp");
  Real beta   = pin->GetReal("problem", "beta");
  // beta_mri sets the strength of the vertical seed perturbation; <= 0 disables it
  Real beta_mri = pin->GetOrAddReal("problem", "beta_mri", -1.0);
  int nwx     = pin->GetOrAddInteger("problem", "nwx", 1);
  bool zero_net_mom = pin->GetOrAddBoolean("problem", "zero_net_momentum", true);
  // diagnostic switch: uniform density and field, no profile math at all
  bool uniform_init = pin->GetOrAddBoolean("problem", "uniform_init", false);

  Real iso_cs = eos.iso_cs;
  Real p0 = d0*SQR(iso_cs);

  Real by0 = std::sqrt(2.0*p0/beta);
  Real bz0 = (beta_mri > 0.0) ? std::sqrt(2.0*p0/beta_mri) : 0.0;
  if (uniform_init) bz0 = 0.0;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real pi_ = static_cast<Real>(M_PI);       // never use the M_PI macro inside a kernel
  Real kx = (2.0*pi_/x1size)*static_cast<Real>(nwx);
  // amplitude of the vector potential, A_y = -abz*cos(kx*x)*g(z).  Guard against nwx = 0
  // so that a zero wavenumber can never produce a 0/0 NaN inside the kernel.
  const Real abz = (kx != 0.0) ? (bz0/kx) : 0.0;

  // amplitude of the random velocity perturbations.  Reproduces the Athena++ version,
  // where each component was (0.4/sqrt(3)) * amp * (ran2-0.5) * 1e-3 with gamma = 1.
  Real vamp = pin->GetOrAddReal("problem", "vamp", (0.4/std::sqrt(3.0))*1.0e-3)*amp;

  // by-value copies of everything the kernels need
  const Real zl = z_lower_;
  const Real zu = z_upper_;
  const Real jw = jwidth;
  const Real dcoef = SQR(by0/iso_cs)*(1.0 + SQR(bx2by))/2.0;
  const Real bxeq = by0*bx2by;               // Bx amplitude
  const Real bzeq = by0*bx2by*bz2bx;         // uniform Bz
  const bool uinit = uniform_init;

  // ---- capture mesh variables for kernels (plain by-value copies, no references) -----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;

  // ---- initialize face-centered magnetic field ---------------------------------------
  // The equilibrium part depends on z only (Bx, By) or is uniform (Bz), so div(B) = 0
  // identically.  The seed perturbation is the discrete curl of
  //     A_y = -(bz0/kx) cos(kx x) [exp(-((z-zu)/jw)^2/2) + exp(-((z-zl)/jw)^2/2)]
  // written out inline, so it is divergence free to machine precision as well.
  auto b0 = pmbp->pmhd->b0;
  par_for("rec_shear_b", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real dx1 = size.d_view(m).dx1;
    Real dx3 = size.d_view(m).dx3;

    Real x3v  = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x1f  = LeftEdgeX(i-is,   nx1, x1min, x1max);
    Real x1fp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x3f  = LeftEdgeX(k-ks,   nx3, x3min, x3max);
    Real x3fp = LeftEdgeX(k+1-ks, nx3, x3min, x3max);

    // S(z) = tanh((z-zu)/jw) - tanh((z-zl)/jw) + 1, or 1 for a uniform field
    Real sprof = 1.0;
    if (!uinit) {
      sprof = Kokkos::tanh((x3v - zu)/jw) - Kokkos::tanh((x3v - zl)/jw) + 1.0;
    }

    // Gaussian envelopes of the vector potential at the two z-faces of this cell
    Real gf = 0.0, gfp = 0.0;
    if (bz0 != 0.0) {
      gf  = Kokkos::exp(-0.5*SQR((x3f  - zu)/jw))
          + Kokkos::exp(-0.5*SQR((x3f  - zl)/jw));
      gfp = Kokkos::exp(-0.5*SQR((x3fp - zu)/jw))
          + Kokkos::exp(-0.5*SQR((x3fp - zl)/jw));
    }

    // Bx = equilibrium - dA_y/dz ; By = equilibrium ; Bz = uniform + dA_y/dx
    Real bx = bxeq*sprof + abz*Kokkos::cos(kx*x1f)*(gfp - gf)/dx3;
    Real by = by0*sprof;
    Real bz = bzeq - abz*gf*(Kokkos::cos(kx*x1fp) - Kokkos::cos(kx*x1f))/dx1;

    b0.x1f(m,k,j,i) = bx;
    b0.x2f(m,k,j,i) = by;
    b0.x3f(m,k,j,i) = bz;

    // upper faces of the last active cell in each direction
    if (i == ie) {
      b0.x1f(m,k,j,ie+1) = bxeq*sprof
                         + abz*Kokkos::cos(kx*x1fp)*(gfp - gf)/dx3;
    }
    if (j == je) {
      b0.x2f(m,k,je+1,i) = by;
    }
    if (k == ke) {
      b0.x3f(m,ke+1,j,i) = bzeq
                         - abz*gfp*(Kokkos::cos(kx*x1fp) - Kokkos::cos(kx*x1f))/dx1;
    }
  });
  Kokkos::fence();

  // ---- initialize density and zero the momenta ---------------------------------------
  auto &u0 = pmbp->pmhd->u0;
  par_for("rec_shear_u", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // density profile giving uniform total (gas + magnetic) pressure
    Real rd = d0;
    if (!uinit) {
      Real sprof = Kokkos::tanh((x3v - zu)/jw) - Kokkos::tanh((x3v - zl)/jw) + 1.0;
      rd = d0 + dcoef*(1.0 - SQR(sprof));
    }

    u0(m,IDN,k,j,i) = rd;
    // NOTE: no -q*Omega0*x term; the background shear is carried by orbital advection.
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });
  Kokkos::fence();

  // ---- add random velocity perturbations ---------------------------------------------
  // Kept in its own kernel and skipped entirely when vamp == 0, so setting <problem>/amp
  // to zero removes the random pool from the run completely.  The fence below is required
  // for correctness, not just tidiness: rand_pool64 owns reference-counted Views that
  // would otherwise be released at the closing brace while the kernel is still running.
  if (vamp != 0.0) {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids + 1);
    par_for("rec_shear_vpert", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      auto rand_gen = rand_pool64.get_state();
      Real rvx = vamp*(rand_gen.frand() - 0.5);
      Real rvy = vamp*(rand_gen.frand() - 0.5);
      Real rvz = vamp*(rand_gen.frand() - 0.5);
      rand_pool64.free_state(rand_gen);

      Real rd = u0(m,IDN,k,j,i);
      u0(m,IM1,k,j,i) = rd*rvx;
      u0(m,IM2,k,j,i) = rd*rvy;
      u0(m,IM3,k,j,i) = rd*rvz;
    });
    Kokkos::fence();
  }

  // ---- remove the net momentum of the random perturbations ---------------------------
  // Uses AthenaK's own array_sum::GlobalSum custom reducer (as the history outputs do).
  // Kokkos' combined multi-reducer path is deliberately avoided here.
  if (zero_net_mom && (vamp != 0.0)) {
    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum msum_this_rank;
    Kokkos::parallel_reduce("rec_shear_mom", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      array_sum::GlobalSum mvars;
      mvars.the_array[0] = u0(m,IM1,k,j,i);
      mvars.the_array[1] = u0(m,IM2,k,j,i);
      mvars.the_array[2] = u0(m,IM3,k,j,i);
      for (int n=3; n<NREDUCTION_VARIABLES; ++n) {
        mvars.the_array[n] = 0.0;
      }
      mb_sum += mvars;
    }, Kokkos::Sum<array_sum::GlobalSum>(msum_this_rank));
    Kokkos::fence();

    Real msum[3] = {msum_this_rank.the_array[0],
                    msum_this_rank.the_array[1],
                    msum_this_rank.the_array[2]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, msum, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    // total number of active cells in the root grid
    Real ncells = static_cast<Real>(pmy_mesh_->mesh_indcs.nx1)
                 *static_cast<Real>(pmy_mesh_->mesh_indcs.nx2)
                 *static_cast<Real>(pmy_mesh_->mesh_indcs.nx3);
    const Real mavg1 = msum[0]/ncells;
    const Real mavg2 = msum[1]/ncells;
    const Real mavg3 = msum[2]/ncells;

    par_for("rec_shear_dmom", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IM1,k,j,i) -= mavg1;
      u0(m,IM2,k,j,i) -= mavg2;
      u0(m,IM3,k,j,i) -= mavg3;
    });
    Kokkos::fence();
  }

  // ---- diagnostic output -------------------------------------------------------------
  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- rec_shear: double Harris sheet in a shearing box ---" << std::endl
      << "  iso_cs           = " << iso_cs   << std::endl
      << "  d0, p0           = " << d0 << ", " << p0 << std::endl
      << "  By0 (beta)       = " << by0 << " (" << beta << ")" << std::endl
      << "  Bz0 (beta_mri)   = " << bz0 << " (" << beta_mri << ")" << std::endl
      << "  bx2by, bz2bx     = " << bx2by << ", " << bz2bx << std::endl
      << "  jwidth           = " << jwidth << std::endl
      << "  sheets at z      = " << z_lower_ << ", " << z_upper_ << std::endl
      << "  rho at sheets    = " << d0 + dcoef << std::endl
      << "  vpert amplitude  = " << vamp << std::endl
      << "  grav             = " << grav_ << std::endl
      << "  sinkwidth        = " << sinkwidth_ << ", tau_sink = " << tau_sink_
      << std::endl;
    if (uniform_init) {
      std::cout << "  *** uniform_init = true: uniform density and field ***"
                << std::endl;
    }
    std::cout << std::endl;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn RecShearSrcTerms
//! \brief User source terms: vertical tidal gravity about the nearest current sheet, plus
//! optional density sinks at the midplane and at the two vertical boundaries.
//! Source terms are computed from the primitives (w0) and applied to the conserved
//! variables (u0), as required by AthenaK.

void RecShearSrcTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;

  auto &u0 = pmbp->pmhd->u0;
  auto &w0 = pmbp->pmhd->w0;
  Real omega0 = pmbp->pmhd->psbox_u->omega0;

  // by-value copies of the file-scope parameters
  const Real grav = grav_;
  const Real zmid = z_mid_, zmin = z_min_, zmax = z_max_;
  const Real zl = z_lower_, zu = z_upper_;
  const Real swidth = sinkwidth_, tausink = tau_sink_, dfloor = dfloor_;
  const Real pi_ = static_cast<Real>(M_PI);

  const Real gcoef = grav*bdt*SQR(omega0);
  const Real zsink_l = zmid - 0.5*swidth, zsink_u = zmid + 0.5*swidth;
  const Real zbot_u  = zmin + 0.5*swidth;
  const Real ztop_l  = zmax - 0.5*swidth;

  par_for("rec_shear_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real den = w0(m,IDN,k,j,i);

    // vertical tidal gravity toward the nearest current sheet
    if (grav != 0.0) {
      Real z0 = (x3v > zmid) ? zu : zl;
      u0(m,IM3,k,j,i) -= gcoef*den*(x3v - z0);
    }

    // density sink in cos^2 windows at the midplane and the vertical boundaries
    if (swidth > 0.0) {
      Real window = 0.0;
      if ((x3v >= zsink_l) && (x3v <= zsink_u)) {
        window = SQR(Kokkos::cos(pi_*(x3v - zmid)/swidth));
      } else if ((x3v >= zmin) && (x3v <= zbot_u)) {
        window = SQR(Kokkos::cos(0.5*pi_*(x3v - zmin)/swidth));
      } else if ((x3v >= ztop_l) && (x3v <= zmax)) {
        window = SQR(Kokkos::cos(0.5*pi_*(x3v - zmax)/swidth));
      }
      if (window > 0.0) {
        // exponential decay factor, stable for any bdt/tau_sink ratio
        Real fac = Kokkos::exp(-window*bdt/tausink);
        // do not drain below the density floor
        Real dden = Kokkos::fmax(den*fac, dfloor) - den;
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
//! \fn RecShearHistory
//! \brief User history output.  Adds Reynolds stress (dVxVy), Maxwell stress (-BxBy) and
//! the net magnetic flux to the standard isothermal-MHD history variables.
//! Same structure as MRIHistory in pgen/tests/mri3d.cpp.

void RecShearHistory(HistoryData *pdata, Mesh *pm) {
  const int nmhd_ = pm->pmb_pack->pmhd->nmhd;

  pdata->nhist = 15;
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  pdata->label[nmhd_   ] = "1-KE";
  pdata->label[nmhd_+1 ] = "2-KE";
  pdata->label[nmhd_+2 ] = "3-KE";
  pdata->label[nmhd_+3 ] = "1-ME";
  pdata->label[nmhd_+4 ] = "2-ME";
  pdata->label[nmhd_+5 ] = "3-ME";
  pdata->label[nmhd_+6 ] = "1-bcc";
  pdata->label[nmhd_+7 ] = "2-bcc";
  pdata->label[nmhd_+8 ] = "3-bcc";
  pdata->label[nmhd_+9 ] = "dVxVy";
  pdata->label[nmhd_+10] = "-BxBy";

  auto &u0_  = pm->pmb_pack->pmhd->u0;
  auto &bx1f = pm->pmb_pack->pmhd->b0.x1f;
  auto &bx2f = pm->pmb_pack->pmhd->b0.x2f;
  auto &bx3f = pm->pmb_pack->pmhd->b0.x3f;
  auto &bcc  = pm->pmb_pack->pmhd->bcc0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const int nhist_ = pdata->nhist;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  const int is = indcs.is; const int nx1 = indcs.nx1;
  const int js = indcs.js; const int nx2 = indcs.nx2;
  const int ks = indcs.ks; const int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("RecShearHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    array_sum::GlobalSum hvars;
    hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
    hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
    hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
    hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);

    // kinetic energy
    hvars.the_array[nmhd_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

    // magnetic energy
    hvars.the_array[nmhd_+3] = vol*0.25*(SQR(bx1f(m,k,j,i+1)) + SQR(bx1f(m,k,j,i)));
    hvars.the_array[nmhd_+4] = vol*0.25*(SQR(bx2f(m,k,j+1,i)) + SQR(bx2f(m,k,j,i)));
    hvars.the_array[nmhd_+5] = vol*0.25*(SQR(bx3f(m,k+1,j,i)) + SQR(bx3f(m,k,j,i)));

    // net magnetic flux
    hvars.the_array[nmhd_+6] = vol*bcc(m,IBX,k,j,i);
    hvars.the_array[nmhd_+7] = vol*bcc(m,IBY,k,j,i);
    hvars.the_array[nmhd_+8] = vol*bcc(m,IBZ,k,j,i);

    // Reynolds and Maxwell stresses.  With orbital advection the y-momentum already is
    // the deviation from the background shear flow, so no vshear correction is needed.
    hvars.the_array[nmhd_+9]  = vol*u0_(m,IM1,k,j,i)*u0_(m,IM2,k,j,i)/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+10] = -vol*bcc(m,IBX,k,j,i)*bcc(m,IBY,k,j,i);

    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
  return;
}
