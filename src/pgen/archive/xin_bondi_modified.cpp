//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xin_bondi_modified.cpp
//! \brief Full history-based GR free-fall accretion IC around a Kerr BH.
//!
//! Solves the Lagrangian map a(r, t_hist) per cell for ideal-MHD flux freezing
//! of an initially uniform vertical B-field through a radial free-fall flow,
//! using Schwarzschild radial free-fall in proper time:
//!   tau = sqrt(a^3 / 2GM) * I(r/a),    I(x) = arccos(sqrt(x)) + sqrt(x(1-x))
//! (Exact for the Schwarzschild metric; tau equals coordinate time for
//! Newtonian free-fall and is an approximation for Kerr at moderate spin --
//! standard GR-Bondi assumption.)  Code units: GM = c = 1, so 2GM = 2.
//!
//! Density from the Lagrangian Jacobian (spherical mass conservation):
//!   rho(r,tau) = rho_low / (x^2 * lambda_r),    x = r/a
//!   lambda_r(x) = dr/da|_tau = x + (3/2) * I(x) * sqrt((1-x)/x)
//!
//! Temperature: uniform.
//!   T(r) = t0
//!
//! Vector potential (frozen-in flux function of initial uniform B_z = B0):
//!   A_phi(r,theta,tau) = 0.5 * B0 * a(r,tau)^2 * sin^2(theta)
//!   B0 = sqrt(2 * rho_low * t0 / beta_box)    (ambient beta at t=0)
//!
//! Limits:
//!   tau -> 0:           a -> r everywhere   => rho = rho_low uniform,
//!                                              B = uniform vertical B0  (IC)
//!   tau large, r << a:  inner-radialized    => rho ~ r^(-3/2),
//!                                              B_r ~ (a/r)^2 cos(theta),
//!                                              B_theta ~ (a/r) lambda_r sin(theta)
//!                                              (full anisotropic free-fall field)
//!
//! The Lagrangian map self-consistently produces BOTH polar amplification AND
//! the equatorial component -- no eps mixing needed.  Density and field share
//! the same a(r,tau) by construction (flux freezing).
//!
//! Velocity: initially at rest (v = 0); gas relaxes under gravity + B.

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <vector>

#include <Kokkos_Random.hpp>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

// prototypes for functions used internally to this pgen
namespace {

struct bondi_pgen {
  Real spin;
  Real dexcise;
  Real pexcise;
  Real gamma_adi;
  Real arad;

  // Full history-based GR free-fall accretion IC parameters.
  //   rho(r,tau) = rho_low / (x^2 lambda_r),  x=r/a, a from cycloid solve
  //   T(r)       = t0
  //   A_phi      = 0.5 * B0 * a(r,tau)^2 * sin^2(theta)
  //   B0         = sqrt(2 * rho_low * t0 / beta_box)
  Real rho_low;    // initial uniform density (pre-accretion ambient; t=0 state)
  Real t0;         // uniform temperature
  Real beta_box;   // ambient plasma-beta at t=0 (sets B0 of initial uniform B_z)
  Real t_hist;     // proper time since accretion began (code units of M)
};

bondi_pgen bondi;

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

// History-based GR free-fall accretion helpers (Schwarzschild proper time).
// Code units: GM = c = r_g = 1, so the cycloid is tau = sqrt(a^3/2) * I(r/a).

// Cycloid integrand: I(x) = arccos(sqrt(x)) + sqrt(x*(1-x)), x = r/a in [0,1].
// I(1) = 0 (a=r, tau=0).  I(0) = pi/2 (collapse to center).
KOKKOS_INLINE_FUNCTION
static Real cycloid_I(Real x) {
  const Real eps = 1.0e-14;
  x = fmin(1.0 - eps, fmax(eps, x));
  return acos(sqrt(x)) + sqrt(x * (1.0 - x));
}

// Schwarzschild proper time to fall from rest at a to radius r (a >= r):
//   tau = sqrt(a^3 / 2) * I(r/a)
// Monotone increasing in a at fixed r (both factors increase).
KOKKOS_INLINE_FUNCTION
static Real freefall_tau(Real a, Real r) {
  if (a <= r) return 0.0;
  return sqrt(0.5 * a * a * a) * cycloid_I(r / a);
}

// Lagrangian Jacobian dr/da|_tau, by implicit differentiation of
// tau = sqrt(a^3/2) * I(r/a) at fixed tau, using I'(x) = -sqrt(x/(1-x)):
//   lambda_r(x) = x + (3/2) * I(x) * sqrt((1-x)/x)
// Limits: lambda_r(1) = 1 (t=0).  lambda_r(0) ~ (3 pi / 4) x^(-1/2) (large tau).
KOKKOS_INLINE_FUNCTION
static Real freefall_lambda_r(Real x) {
  const Real eps = 1.0e-14;
  x = fmin(1.0 - eps, fmax(eps, x));
  const Real I = cycloid_I(x);
  return x + 1.5 * I * sqrt((1.0 - x) / x);
}

// Solve a(r, tau): initial radius of the gas element now at r at proper time tau.
// Bisection on the monotone freefall_tau(a, r), with an analytic warm-start
// from the large-tau asymptote tau ~ (pi/2) sqrt(a^3/2).  Fixed iteration
// counts (no recursion, no allocation) -- device-friendly.
KOKKOS_INLINE_FUNCTION
static Real solve_a_of_r_tau(Real r, Real tau) {
  if (tau <= 0.0) return r;
  const Real eps = 1.0e-12;

  Real a_lo = r;
  Real a_hi = pow(pow(r, 1.5) + 1.5 * sqrt(2.0) * tau, 2.0/3.0);
  a_hi = fmax(a_hi, r * (1.0 + 1.0e-8) + eps);

  // Guarantee bracket: t_ff(a_hi, r) >= tau.
  for (int n = 0; n < 16; ++n) {
    if (freefall_tau(a_hi, r) >= tau) break;
    a_hi *= 2.0;
  }

  // Bisection (64 iters >> needed precision; cheap, runs only at IC).
  for (int n = 0; n < 64; ++n) {
    const Real a_mid = 0.5 * (a_lo + a_hi);
    if (freefall_tau(a_mid, r) < tau) {
      a_lo = a_mid;
    } else {
      a_hi = a_mid;
    }
  }
  return 0.5 * (a_lo + a_hi);
}

// Uniform temperature.
KOKKOS_INLINE_FUNCTION
static Real T_profile(struct bondi_pgen pgen, Real r) {
  (void)r;
  return pgen.t0;
}

// History-based density from spherical mass conservation:
//   rho(r,tau) = rho_low / (x^2 * lambda_r),   x = r/a(r,tau)
// At tau=0 (a=r, lambda_r=1): rho = rho_low (initial uniform state).
// For r << a (deep radialized): rho ~ rho_low * (a/r)^(3/2)  =>  rho ~ r^(-3/2).
KOKKOS_INLINE_FUNCTION
static Real rho_profile(struct bondi_pgen pgen, Real r) {
  const Real a = solve_a_of_r_tau(r, pgen.t_hist);
  const Real x = r / a;
  const Real lambda_r = freefall_lambda_r(x);
  return pgen.rho_low / (x * x * lambda_r);
}

// Initial uniform-B_z amplitude (t=0 ambient field) from beta_box:
//   B0 = sqrt(2 * rho_low * t0 / beta_box)
KOKKOS_INLINE_FUNCTION
static Real B0_amplitude(struct bondi_pgen pgen) {
  return sqrt(2.0 * pgen.rho_low * pgen.t0 / pgen.beta_box);
}

// Diagnostic polar field strength B_r_hat at theta=0:
//   B_r(pole) = B0 * (a/r)^2
KOKKOS_INLINE_FUNCTION
static Real b_profile(struct bondi_pgen pgen, Real r) {
  const Real B0 = B0_amplitude(pgen);
  const Real a  = solve_a_of_r_tau(r, pgen.t_hist);
  return B0 * (a/r) * (a/r);
}

// Diagnostic polar plasma-beta: beta_pole = p_gas / (0.5 * B_pole^2).
KOKKOS_INLINE_FUNCTION
static Real beta_profile(struct bondi_pgen pgen, Real r) {
  const Real rho_l = rho_profile(pgen, r);
  const Real b_l   = b_profile(pgen, r);
  return 2.0 * rho_l * pgen.t0 / (b_l * b_l);
}

// History-based vector potential -- frozen-in flux function of initial uniform B_z:
//   A_phi(r,theta,tau) = F(r,tau) * sin^2(theta)
//   F(r,tau)           = 0.5 * B0 * a(r,tau)^2
//
// Limits:
//   tau -> 0:      a = r           => F = 0.5 * B0 * r^2  (uniform vertical B_z)
//   r << a(tau):   a ~ const(tau)  => F ~ const           (radial-dipole inner)
//
// The full a(r,tau) self-consistently yields both polar (B_r ~ (a/r)^2 cos theta)
// and equatorial (B_theta ~ (a/r) lambda_r sin theta) components.  No eps mixing.
//
// Density (rho_profile) and field (aphi_F) call solve_a_of_r_tau with the SAME
// arguments and therefore share the same Lagrangian map by construction.
KOKKOS_INLINE_FUNCTION
static Real aphi_F(struct bondi_pgen pgen, Real r) {
  const Real B0 = B0_amplitude(pgen);
  const Real a  = solve_a_of_r_tau(r, pgen.t_hist);
  return 0.5 * B0 * a * a;
}

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotential(struct bondi_pgen pgen,
                                     Real r, Real theta, Real phi,
                                     Real *patheta, Real *paphi);

KOKKOS_INLINE_FUNCTION
Real A1(struct bondi_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct bondi_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct bondi_pgen pgen, Real x1, Real x2, Real x3);

} // namespace

// Prototypes for user-defined BCs and history functions
void ReservoirBondiBounded(Mesh *pm);
void BondiBoundedFluxes(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Full history-based GR free-fall accretion IC around a Kerr BH
//!        (uniform T; Lagrangian map a(r,t_hist) from Schwarzschild proper
//!        time; rho and A_phi self-consistent from flux freezing of initial
//!        uniform B_z).
//! Compile with '-D PROBLEM=xin_bondi_modified' to enroll.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR bondi problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = ReservoirBondiBounded;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // Extract BH parameters
  bondi.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux = (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(bondi.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = BondiBoundedFluxes;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Extract radiation parameters if enabled
  int nangles_;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled) {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    bondi.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bondi.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bondi.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  bondi.arad = 0.0;
  if (pmbp->prad != nullptr) {
    bondi.arad = pmbp->prad->arad;
  }

  // Read history-based GR free-fall IC parameters.
  bondi.rho_low  = pin->GetReal("problem", "rho_low");
  bondi.t0       = pin->GetReal("problem", "t0");
  bondi.beta_box = pin->GetReal("problem", "beta_box");
  bondi.t_hist   = pin->GetReal("problem", "t_hist");

  // Local-only: 1% random pressure perturbation to seed MRI / break axisymmetry.
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.01);

  // Sanity check
  if (bondi.rho_low <= 0.0 || bondi.t0 <= 0.0 || bondi.beta_box <= 0.0 ||
      bondi.t_hist < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Invalid <problem> parameters: rho_low, t0, beta_box must be > 0; "
              << "t_hist must be >= 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // excision parameters
  bondi.dexcise = coord.dexcise;
  bondi.pexcise = coord.pexcise;

  // Return on restart AFTER the params above: the reservoir BC and history
  // function run on restart too and read this struct, so it must be populated
  // (else the BC pins all boundary ghosts to vacuum -> global NaN).
  if (restart) return;

  // initialize primitive variables for new run
  auto trs = bondi;
  auto &size = pmbp->pmb->mb_size;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

  par_for("pgen_bondi_ic", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    // Extract metric and inverse (needed for radiation intensity step)
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);

    // BL radius for the density profile
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);

    // Excision check: use the cell corner farthest from origin so that cells
    // with only a tiny tip inside the horizon are NOT excised.
    Real r_excise, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(trs, x1v + copysign(0.5*dx1, x1v),
                                      x2v + copysign(0.5*dx2, x2v),
                                      x3v + copysign(0.5*dx3, x3v),
                                      &r_excise, &theta_excise, &phi_excise);

    Real rho, pgas, T_local, urad = 0.0;
    Real perturbation = 0.0;
    if (r_excise > 1.0) {
      // Bounded Bondi-equilibrium profile (gamma = 5/3)
      rho      = rho_profile(trs, r);
      T_local  = T_profile  (trs, r);
      pgas     = rho * T_local;
      if (is_radiation_enabled) urad = trs.arad * SQR(SQR(T_local));

      // 1% random pressure perturbation: ρ, T, B, u_rad untouched (preserves
      // div(B)=0 and gas/radiation thermal equilibrium); only p_gas gets the wiggle.
      auto rand_gen = rand_pool64.get_state();
      perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
      rand_pool64.free_state(rand_gen);
    } else {
      rho     = trs.dexcise;
      pgas    = trs.pexcise;
      T_local = (rho > 0.0) ? (pgas / rho) : 0.0;
    }

    // Initial velocity: v = 0 (gas relaxes under gravity + B).
    Real uu1 = 0.0, uu2 = 0.0, uu3 = 0.0;

    // Write primitives
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas * (1.0 + perturbation) / gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

    // Coordinate-frame specific intensity (if radiation enabled)
    if (is_radiation_enabled) {
      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
             + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      for (int n=0; n<nangles_; ++n) {
        Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                     u_tet_[3]*nh_c_.d_view(n,3));
        Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;
        Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
        i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
      }
    }
  });

  // initialize ADM variables -----------------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // initialize magnetic fields ---------------------------------------
  if (pmbp->pmhd != nullptr) {
    // No global b0: A_phi(r,theta) = F(r) * sin^2(theta) via aphi_F().
    // This gives B_z(R, z=0) = b(R) = sqrt(2 p_gas/beta) on the equator.

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;
    auto trs = bondi;

    par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(trs, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(trs, x1f, x2v, x3f);
      a3(m,k,j,i) = A3(trs, x1f, x2f, x3v);

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(trs, xl,x2f,x3f) + A1(trs, xr,x2f,x3f));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(trs, x1f,xl,x3f) + A2(trs, x1f,xr,x3f));
      }

      // Correct A3 at x1-faces, x2-faces, and x1x2-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
          (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
        Real xl = x3v + 0.25*dx3;
        Real xr = x3v - 0.25*dx3;
        a3(m,k,j,i) = 0.5*(A3(trs, x1f,x2f,xl) + A3(trs, x1f,x2f,xr));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A); div-free at discrete level.
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  } else {
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating vector potential in Spherical KS given CKS coordinates
// Uses A_phi = F(r) * sin^2(theta) with F(r) = integral_0^r R' b(R') dR'.
// This construction guarantees B_z(R, z=0) = b(R) on the equator, so the local
// magnetization beta(r) is honored at every radius (unlike the standard Wald
// form, where the (r^2 - a^2) factor cancels the r-dependence of b(r)).

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotential(struct bondi_pgen pgen,
                                     Real r, Real theta, Real phi,
                                     Real *patheta, Real *paphi) {
  Real sin_theta = sin(theta);

  *patheta = 0.0;
  *paphi   = aphi_F(pgen, r) * SQR(sin_theta);
  return;
}

//----------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in spherical KS coordinates, then transforms to Cartesian KS

KOKKOS_INLINE_FUNCTION
Real A1(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A2(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A3(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ReservoirBondiBounded
//  \brief Dirichlet BC pinning ghost zones to the asymptotic (floor) profile values:
//         rho = rho_low, T = t0, p = rho_low*t0, v copied from interior.
//         B at boundary is the curl-of-A from the IC profile (already in face arrays);
//         we pin the face fields to be a copy of the innermost active face values.
//         Radiation BC is the standard "no-inflow" filter on i0.
// FIXME: Boundaries need to be adjusted for DynGRMHD

void ReservoirBondiBounded(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  DvceArray6D<Real> tc; DualArray2D<Real> nh_c_;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    nh_c_ = pm->pmb_pack->prad->nh_c;
    tc    = pm->pmb_pack->prad->tet_c;
  }

  // Reservoir (asymptotic floor) state
  const Real gm1 = bondi.gamma_adi - 1.0;
  const Real rho_res  = bondi.rho_low;
  const Real T_res    = bondi.t0;
  const Real pgas_res = bondi.rho_low * bondi.t0;

  // X1-Boundary: copy face B from interior (preserves curl-of-A profile)
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("dirichlet_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
  }

  // ConsToPrim over X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }

  // Dirichlet primitives in X1 ghost zones (asymptotic floor state, v copied)
  par_for("dirichlet_hydro_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        int ig = is - i - 1;
        w0_(m,IDN,k,j,ig) = rho_res;
        w0_(m,IEN,k,j,ig) = pgas_res / gm1;
        w0_(m,IVX,k,j,ig) = w0_(m,IVX,k,j,is);
        w0_(m,IVY,k,j,ig) = w0_(m,IVY,k,j,is);
        w0_(m,IVZ,k,j,ig) = w0_(m,IVZ,k,j,is);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        int ig = ie + i + 1;
        w0_(m,IDN,k,j,ig) = rho_res;
        w0_(m,IEN,k,j,ig) = pgas_res / gm1;
        w0_(m,IVX,k,j,ig) = w0_(m,IVX,k,j,ie);
        w0_(m,IVY,k,j,ig) = w0_(m,IVY,k,j,ie);
        w0_(m,IVZ,k,j,ig) = w0_(m,IVZ,k,j,ie);
      }
    }
  });

  if (is_radiation_enabled) {
    par_for("dirichlet_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        Real n1 = tc(m,0,1,k,j,is)*nh0 + tc(m,1,1,k,j,is)*nh1 + tc(m,2,1,k,j,is)*nh2 + tc(m,3,1,k,j,is)*nh3;
        Real val = (n1 > 0) ? 0.0 : i0_(m,n,k,j,is);
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        Real n1 = tc(m,0,1,k,j,ie)*nh0 + tc(m,1,1,k,j,ie)*nh1 + tc(m,2,1,k,j,ie)*nh2 + tc(m,3,1,k,j,ie)*nh3;
        Real val = (n1 < 0) ? 0.0 : i0_(m,n,k,j,ie);
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = val;
        }
      }
    });
  }

  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  }

  // X2-Boundary
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("dirichlet_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
  }

  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
  }

  par_for("dirichlet_hydro_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        int jg = js - j - 1;
        w0_(m,IDN,k,jg,i) = rho_res;
        w0_(m,IEN,k,jg,i) = pgas_res / gm1;
        w0_(m,IVX,k,jg,i) = w0_(m,IVX,k,js,i);
        w0_(m,IVY,k,jg,i) = w0_(m,IVY,k,js,i);
        w0_(m,IVZ,k,jg,i) = w0_(m,IVZ,k,js,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        int jg = je + j + 1;
        w0_(m,IDN,k,jg,i) = rho_res;
        w0_(m,IEN,k,jg,i) = pgas_res / gm1;
        w0_(m,IVX,k,jg,i) = w0_(m,IVX,k,je,i);
        w0_(m,IVY,k,jg,i) = w0_(m,IVY,k,je,i);
        w0_(m,IVZ,k,jg,i) = w0_(m,IVZ,k,je,i);
      }
    }
  });

  if (is_radiation_enabled) {
    par_for("dirichlet_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        Real n2 = tc(m,0,2,k,js,i)*nh0 + tc(m,1,2,k,js,i)*nh1 + tc(m,2,2,k,js,i)*nh2 + tc(m,3,2,k,js,i)*nh3;
        Real val = (n2 > 0) ? 0.0 : i0_(m,n,k,js,i);
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        Real n2 = tc(m,0,2,k,je,i)*nh0 + tc(m,1,2,k,je,i)*nh1 + tc(m,2,2,k,je,i)*nh2 + tc(m,3,2,k,je,i)*nh3;
        Real val = (n2 < 0) ? 0.0 : i0_(m,n,k,je,i);
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = val;
        }
      }
    });
  }

  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  }

  // X3-Boundary
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("dirichlet_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }

  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }

  par_for("dirichlet_hydro_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        int kg = ks - k - 1;
        w0_(m,IDN,kg,j,i) = rho_res;
        w0_(m,IEN,kg,j,i) = pgas_res / gm1;
        w0_(m,IVX,kg,j,i) = w0_(m,IVX,ks,j,i);
        w0_(m,IVY,kg,j,i) = w0_(m,IVY,ks,j,i);
        w0_(m,IVZ,kg,j,i) = w0_(m,IVZ,ks,j,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        int kg = ke + k + 1;
        w0_(m,IDN,kg,j,i) = rho_res;
        w0_(m,IEN,kg,j,i) = pgas_res / gm1;
        w0_(m,IVX,kg,j,i) = w0_(m,IVX,ke,j,i);
        w0_(m,IVY,kg,j,i) = w0_(m,IVY,ke,j,i);
        w0_(m,IVZ,kg,j,i) = w0_(m,IVZ,ke,j,i);
      }
    }
  });

  if (is_radiation_enabled) {
    par_for("dirichlet_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        Real n3 = tc(m,0,3,ks,j,i)*nh0 + tc(m,1,3,ks,j,i)*nh1 + tc(m,2,3,ks,j,i)*nh2 + tc(m,3,3,ks,j,i)*nh3;
        Real val = (n3 > 0) ? 0.0 : i0_(m,n,ks,j,i);
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        Real n3 = tc(m,0,3,ke,j,i)*nh0 + tc(m,1,3,ke,j,i)*nh1 + tc(m,2,3,ke,j,i)*nh2 + tc(m,3,3,ke,j,i)*nh3;
        Real val = (n3 < 0) ? 0.0 : i0_(m,n,ke,j,i);
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = val;
        }
      }
    });
  }

  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces
// (identical to BondiFluxes in xin_new.cpp, renamed to avoid symbol collision)

void BondiBoundedFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  Real to_ien = 1.;
  if (pmbp->pdyngr != nullptr) {
    to_ien = 1.0 / (gamma - 1.);
  }

  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = (is_mhd) ? 4 : 3;

  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
    if (is_mhd) {
      pdata->label[nflux*g+3] = "phi_" + rad_str;
    }
  }

  DualArray2D<Real> interpolated_bcc;
  for (int g=0; g<nradii; ++g) {
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    if (is_mhd) pdata->hdata[nflux*g+3] = 0.0;

    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    for (int n=0; n<grids[g]->nangles; ++n) {
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN)*to_ien;

      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      Real a2 = SQR(spin);
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;

      if (is_mhd) {
        pdata->hdata[nflux*g+3] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
      }
    }
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
