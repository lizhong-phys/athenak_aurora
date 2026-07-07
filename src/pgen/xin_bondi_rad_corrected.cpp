//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xin_bondi_rad_corrected.cpp
//! \brief Problem generator: uniform gas at rest around a Kerr black hole, threaded
//!        by an (optional) uniform vertical B-field. Used to test relaxation toward
//!        Bondi-like accretion equilibrium.
//!
//! CORRECTED version of xin_final.cpp with two fixes:
//!   (1) beta_target is the GAS plasma-beta: b0 = sqrt(2*p_gas/beta_target), with NO
//!       radiation term (the old code used p_gas + arad*T^4, i.e. radiation ENERGY
//!       density treated as a pressure, so beta_target did not equal the realized beta).
//!   (2) Radiation reservoir BC: INCOMING directions are set to the ambient at-rest
//!       equilibrium intensity via BondiEquilibIntensity (the old code zeroed them,
//!       which acted as a spurious radiation sink).  OUTGOING directions still escape.
//!
//! Inputs (all physical, code units c = G = 1):
//!    rho0        : asymptotic / background rest-mass density rho_inf
//!    t0          : pressure-per-density p/rho ("temperature"; set directly)
//!    beta_target : ambient plasma-beta beta_inf (sets B0 amplitude)
//!
//! Initial state (Cartesian Kerr-Schild):
//!    rho(r, t=0) = rho0                                    (uniform)
//!    p           = rho * t0                                (uniform T = t0)
//!    uu^i        = 0                                       (gas at rest)
//!    A_phi       = 0.5 * b0 * (r sin theta)^2              (-> uniform B_z)
//!
//! B-field amplitude b0 from beta_target (GAS plasma-beta).  v=0 at IC, no (1-v^2):
//!    b0 = sqrt(2 * p_gas / beta_target),   p_gas = rho0*t0
//!
//! Outer boundary: Dirichlet-to-uniform-reservoir (ReservoirBondi).
//! Inside the horizon: primitives replaced by excision floors (dexcise, pexcise).
//!
//! Adapted from gr_torus.cpp by stripping the FM/Chakrabarti torus equilibrium
//! and replacing the density-weighted poloidal field with a uniform B_z.

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
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

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

// Parameters for Gaussian-dip IC with uniform vertical B-field
struct bondi_pgen {
  // Spacetime / excision / EOS
  Real spin;            // black hole spin a/M
  Real dexcise;         // density floor inside excision radius
  Real pexcise;         // pressure floor inside excision radius
  Real gamma_adi;       // ideal-gas adiabatic index
  Real arad;            // radiation constant (only if radiation enabled)

  // Initial density profile:  rho(r) = rho0 (uniform; no dip).
  Real rho0;            // asymptotic / background rest-mass density

  // Pressure law for the initial state.  Convention: p = rho * t0 (T = t0 const).
  Real t0;              // uniform initial temperature (p/rho)

  // Uniform vertical magnetic field in the Wald/asymptotic sense:
  // A_phi = 0.5 * b0 * (r sin theta)^2, giving B -> B0 z-hat at large r,
  // divergence-free everywhere by construction (B = curl A on faces).
  // Set b0 = 0.0 to disable MHD initialization.
  Real b0;              // vertical field amplitude
};

  bondi_pgen bondi;

} // namespace

// Prototypes for user-defined BCs and history functions
void ReservoirBondi(Mesh *pm);
void BondiFluxes(HistoryData *pdata, Mesh *pm);

// Prototypes for user-defined source functions
void MySourceTerms(Mesh* pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Uniform gas + Gaussian central dip around a Kerr BH, optionally threaded
//!        by a uniform vertical B-field. See file header for full description.
//! Compile with '-D PROBLEM=xin_final' to enroll as user-specific problem generator.

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
  user_bcs_func = ReservoirBondi;

  // User src terms
  user_srcs_func = MySourceTerms;

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
  // NOTE(@pdmullen): Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = BondiFluxes;

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
  if (pmbp->prad != nullptr) {
    bondi.arad = pmbp->prad->arad;
  }

  // Read Bondi IC parameters from input file
  //   rho(r, t=0) = rho0  (uniform; no dip)
  //   p           = rho * t0   (uniform T = t0)
  bondi.rho0 = pin->GetReal("problem", "rho0");
  bondi.t0   = pin->GetReal("problem", "t0");

  // excision parameters
  bondi.dexcise = coord.dexcise;
  bondi.pexcise = coord.pexcise;

  // Vertical field amplitude b0 from beta_target.  Computed here, BEFORE the
  // restart return, so bondi.b0 is populated on restart too (the reservoir BC
  // reads it as bz_res).  Harmless today since the field BC is zero-gradient
  // copy, but this keeps the struct restart-correct and avoids a latent bug if
  // the BC is ever switched to pin a uniform B_z.
  if (pmbp->pmhd != nullptr) {
    Real beta_target = pin->GetOrAddReal("problem", "beta_target", 100.0);
    // GAS plasma-beta:  beta_target = p_gas / (b^2/2)  ->  b0 = sqrt(2*p_gas/beta_target).
    // Radiation is intentionally NOT in this balance, so beta_target equals the
    // realized gas plasma-beta independent of density/radiation.  (For a total-pressure
    // beta, add the radiation PRESSURE bondi.arad*SQR(SQR(bondi.t0))/3.0 -- NOTE the /3:
    // arad*T^4 is the radiation ENERGY density, the pressure is one third of it.)
    Real p_gas = bondi.rho0*bondi.t0;
    bondi.b0 = sqrt(2.0*p_gas / beta_target);
  }

  // Return on restart AFTER the params above: the reservoir BC and history
  // function run on restart too and read this struct, so it must be populated
  // (else the BC pins all boundary ghosts to vacuum -> global NaN).
  if (restart) return;

  // initialize primitive variables for new run ---------------------------------------
  //   rho(r, t=0) = rho0  (uniform; no dip)
  //   p = rho * t0,  uu^i = 0,  uniform B_z via vector potential (below)

  auto trs = bondi;
  auto &size = pmbp->pmb->mb_size;

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

    Real rho, pgas, urad = 0.0;
    if (r_excise > 1.0) {
      // Uniform background (no dip)
      rho  = trs.rho0;
      pgas = rho * trs.t0;
      if (is_radiation_enabled) urad = trs.arad * SQR(SQR(trs.t0));
    } else {
      // Inside horizon: apply excision floors
      rho  = trs.dexcise;
      pgas = trs.pexcise;
    }
    Real uu1 = 0.0, uu2 = 0.0, uu3 = 0.0;

    // Write primitives
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas / gm1;
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
    // Uniform vertical B_z via target plasma beta; bondi.b0 was already
    // computed above (before the restart return):
    //   beta = p_gas / (b^2 / 2)  ->  b0 = sqrt(2 * p_gas / beta_target)

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
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
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
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // No global beta-renormalization needed: bondi.b0 was set analytically from
    // beta_target above, so the vector-potential kernel already produced B at the
    // correct amplitude. Cell-centered B (bcc_) was filled by pgen_bcc.
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
    //pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

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
// Inputs:
//   r,theta,phi spherical Boyer-Lindquist coordinates of point
// Outputs:
//   patheta,paphi: pointers to lower theta, phi components in desired coordinates

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotential(struct bondi_pgen pgen,
                                     Real r, Real theta, Real phi,
                                     Real *patheta, Real *paphi) {
  // Uniform vertical B_z via  A_phi = 0.5 * b0 * (r sin theta)^2
  //   (in spherical Boyer-Lindquist coords; a_r = a_theta = 0)
  //   b0 is set analytically from beta_target by the pgen setup.
  Real sin_theta = sin(theta);
  Real a = pgen.spin;

  *patheta = 0.0;
  *paphi   = 0.5 * pgen.b0 * (SQR(r) - SQR(a)) * SQR(sin_theta);
  return;
}
//----------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in spherical KS coordinates, then transforms to Cartesian KS

KOKKOS_INLINE_FUNCTION
Real A1(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotential(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));
}

} // namespace



void MySourceTerms(Mesh* pm, const Real bdt) {

  // damp gas velocity near BH mask
  {
    // capture variables for kernel
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &indcs = pm->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb = pmbp->nmb_thispack;
    auto &size = pmbp->pmb->mb_size;

    // MHD variables
    DvceArray5D<Real> u0_ = pmbp->pmhd->u0;
    DvceArray5D<Real> w0_ = pmbp->pmhd->w0;
    DvceArray5D<Real> bcc0_ = pmbp->pmhd->bcc0;

    // excision masks
    auto &excision_floor_ = pmbp->pcoord->excision_floor;
    auto &excision_flux_  = pmbp->pcoord->excision_flux;

    // coordinate data
    auto &coord = pmbp->pcoord->coord_data;
    bool flat_  = coord.is_minkowski;

    Real gm1 = bondi.gamma_adi - 1.0;
    auto bondi_dvce = bondi;

    // Damping coefficient [0, 1]: fraction of lateral velocity removed per timestep.
    // Lateral components are damped unconditionally as they drive the instability.
    // Radial component is left untouched entirely.
    // Use alpha=1.0 to fully suppress lateral oscillations; tune down if too aggressive.
    const Real alpha = 0.3;

    // Loop over ACTIVE cells only (is..ie, etc.).  The kernel reads i+-1/j+-1/
    // k+-1 neighbors, so looping over ghost zones (0..nx+2ng-1) would read out
    // of bounds at the array edges -> GPU segfault.  Active range keeps every
    // neighbor access within [is-1,ie+1] (valid ghosts); all mask-edge cells
    // are interior, so none are missed.
    par_for("mask_vel_damp", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

      // ----------------------------------------------------------------
      // 1. Cell selection: active cells directly adjacent to BH mask.
      //    Condition: excision_flux=true, excision_floor=false, and at
      //    least one face neighbor inside the mask (excision_floor=true).
      // ----------------------------------------------------------------
      if ( excision_floor_(m,k,j,i)) return;
      if (!excision_flux_ (m,k,j,i)) return;
      bool has_masked_neighbor =
        excision_floor_(m,k,j,i+1) || excision_floor_(m,k,j,i-1) ||
        excision_floor_(m,k,j+1,i) || excision_floor_(m,k,j-1,i) ||
        excision_floor_(m,k+1,j,i) || excision_floor_(m,k-1,j,i);
      if (!has_masked_neighbor) return;

      // ----------------------------------------------------------------
      // 2. Cell-center coordinates
      // ----------------------------------------------------------------
      Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      // ----------------------------------------------------------------
      // 3. Radial unit vector in CKS Cartesian components
      // ----------------------------------------------------------------
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(bondi_dvce, x1v, x2v, x3v, &r, &theta, &phi);
      Real rad2 = SQR(x1v) + SQR(x2v) + SQR(x3v);
      Real r2   = SQR(r);
      Real a2   = SQR(bondi_dvce.spin);
      Real denom = 2.0*r2 - rad2 + a2;   // = 2r^2 - (x^2+y^2+z^2) + a^2

      Real drdx = r*x1v / denom;
      Real drdy = r*x2v / denom;
      Real drdz = (r*x3v + a2*x3v/r) / denom;

      // normalize to get radial unit vector
      Real norm  = sqrt(SQR(drdx) + SQR(drdy) + SQR(drdz));
      Real inv_n = 1.0 / fmax(norm, 1.0e-12);
      Real nr_x  = drdx * inv_n;
      Real nr_y  = drdy * inv_n;
      Real nr_z  = drdz * inv_n;

      // ----------------------------------------------------------------
      // 4. Current primitive velocity and decomposition into radial
      //    and lateral components.
      //    In GRMHD (Valencia formulation), w0_(IVX/IVY/IVZ) = u^i in
      //    the normal frame. Free-fall target is u^i = 0.
      // ----------------------------------------------------------------
      Real uu1 = w0_(m,IVX,k,j,i);
      Real uu2 = w0_(m,IVY,k,j,i);
      Real uu3 = w0_(m,IVZ,k,j,i);

      // radial component (positive = outward)
      Real vr  = uu1*nr_x + uu2*nr_y + uu3*nr_z;

      // lateral (tangential) components
      Real uu1_lat = uu1 - vr*nr_x;
      Real uu2_lat = uu2 - vr*nr_y;
      Real uu3_lat = uu3 - vr*nr_z;

      // ----------------------------------------------------------------
      // 5. Damp lateral components unconditionally toward zero.
      //    Radial component is left untouched entirely — only the lateral
      //    oscillations are driving the numerical instability.
      // ----------------------------------------------------------------
      Real uu1_new = uu1 - alpha * uu1_lat;
      Real uu2_new = uu2 - alpha * uu2_lat;
      Real uu3_new = uu3 - alpha * uu3_lat;

      w0_(m,IVX,k,j,i) = uu1_new;
      w0_(m,IVY,k,j,i) = uu2_new;
      w0_(m,IVZ,k,j,i) = uu3_new;

      // ----------------------------------------------------------------
      // 6. Recompute conserved variables with full CKS metric to keep
      //    primitives and conserved arrays consistent.
      // ----------------------------------------------------------------
      Real dens = w0_(m,IDN,k,j,i);
      Real eint = w0_(m,IEN,k,j,i);
      Real bcc1 = bcc0_(m,IBX,k,j,i);
      Real bcc2 = bcc0_(m,IBY,k,j,i);
      Real bcc3 = bcc0_(m,IBZ,k,j,i);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat_, bondi_dvce.spin, glower, gupper);

      Real lapse  = sqrt(-1.0 / gupper[0][0]);
      Real q      = glower[1][1]*SQR(uu1_new)
                  + 2.0*glower[1][2]*uu1_new*uu2_new
                  + 2.0*glower[1][3]*uu1_new*uu3_new
                  + glower[2][2]*SQR(uu2_new)
                  + 2.0*glower[2][3]*uu2_new*uu3_new
                  + glower[3][3]*SQR(uu3_new);
      Real lor    = sqrt(1.0 + q);
      Real u0_con = lor / lapse;
      Real u1_con = uu1_new - lapse * lor * gupper[0][1];
      Real u2_con = uu2_new - lapse * lor * gupper[0][2];
      Real u3_con = uu3_new - lapse * lor * gupper[0][3];

      Real u_0 = glower[0][0]*u0_con + glower[0][1]*u1_con + glower[0][2]*u2_con + glower[0][3]*u3_con;
      Real u_1 = glower[1][0]*u0_con + glower[1][1]*u1_con + glower[1][2]*u2_con + glower[1][3]*u3_con;
      Real u_2 = glower[2][0]*u0_con + glower[2][1]*u1_con + glower[2][2]*u2_con + glower[2][3]*u3_con;
      Real u_3 = glower[3][0]*u0_con + glower[3][1]*u1_con + glower[3][2]*u2_con + glower[3][3]*u3_con;

      Real b0_4 = u_1*bcc1 + u_2*bcc2 + u_3*bcc3;
      Real b1_4 = (bcc1 + b0_4*u1_con) / u0_con;
      Real b2_4 = (bcc2 + b0_4*u2_con) / u0_con;
      Real b3_4 = (bcc3 + b0_4*u3_con) / u0_con;
      Real b_0  = glower[0][0]*b0_4 + glower[0][1]*b1_4 + glower[0][2]*b2_4 + glower[0][3]*b3_4;
      Real b_1  = glower[1][0]*b0_4 + glower[1][1]*b1_4 + glower[1][2]*b2_4 + glower[1][3]*b3_4;
      Real b_2  = glower[2][0]*b0_4 + glower[2][1]*b1_4 + glower[2][2]*b2_4 + glower[2][3]*b3_4;
      Real b_3  = glower[3][0]*b0_4 + glower[3][1]*b1_4 + glower[3][2]*b2_4 + glower[3][3]*b3_4;
      Real b_sq = b0_4*b_0 + b1_4*b_1 + b2_4*b_2 + b3_4*b_3;

      Real wtot = dens + (gm1+1.0)*eint + b_sq;

      u0_(m,IDN,k,j,i) = dens * u0_con;
      u0_(m,IEN,k,j,i) = wtot*u0_con*u_0 - b0_4*b_0 + (gm1*eint + 0.5*b_sq) + dens*u0_con;
      u0_(m,IM1,k,j,i) = wtot*u0_con*u_1 - b0_4*b_1;
      u0_(m,IM2,k,j,i) = wtot*u0_con*u_2 - b0_4*b_2;
      u0_(m,IM3,k,j,i) = wtot*u0_con*u_3 - b0_4*b_3;

    }); // end par_for mask_vel_damp
  } // end vel damping

  return;
}





//----------------------------------------------------------------------------------------
//! \fn BondiEquilibIntensity
//  \brief Coordinate-frame specific intensity (AthenaK i0 convention) of an isotropic
//  blackbody of energy density urad in the fluid frame, for gas AT REST (v=0), at cell
//  (m,k,j,i) for discrete angle (nh0..nh3).  Identical to the IC formula (UserProblem)
//  with uu=0, so u_tet = the normal-frame column 0 of norm_to_tet.  Used by the reservoir
//  BC to set INCOMING radiation directions to the ambient equilibrium instead of zeroing
//  them (the old behaviour acted as a spurious radiation sink at the outer boundary).
KOKKOS_INLINE_FUNCTION
Real BondiEquilibIntensity(int m, int k, int j, int i,
                           Real nh0, Real nh1, Real nh2, Real nh3, Real urad,
                           const DvceArray6D<Real> &ntt,
                           const DvceArray6D<Real> &tetc,
                           const DvceArray6D<Real> &tcov) {
  // gas at rest: uu^i = 0 -> uu0 = 1, so u_tet = column 0 of norm_to_tet
  Real ut0 = ntt(m,0,0,k,j,i);
  Real ut1 = ntt(m,1,0,k,j,i);
  Real ut2 = ntt(m,2,0,k,j,i);
  Real ut3 = ntt(m,3,0,k,j,i);
  Real un_t = ut1*nh1 + ut2*nh2 + ut3*nh3;
  Real n0_f = ut0*nh0 - un_t;
  Real n0  = tetc(m,0,0,k,j,i);
  Real n_0 = tcov(m,0,0,k,j,i)*nh0 + tcov(m,1,0,k,j,i)*nh1 +
             tcov(m,2,0,k,j,i)*nh2 + tcov(m,3,0,k,j,i)*nh3;
  return n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
}

//----------------------------------------------------------------------------------------
//! \fn ReservoirBondi
//  \brief Dirichlet-to-uniform-reservoir BC on all 6 outer faces.
//         Pins ghost zones to (rho0, p=rho0*t0, v=0, B_z=b0). Radiation: OUTGOING
//         directions escape (copy active cell); INCOMING set to the at-rest ambient
//         equilibrium intensity via BondiEquilibIntensity (no longer zeroed/sink).
// FIXME: Boundaries need to be adjusted for DynGRMHD

void ReservoirBondi(Mesh *pm) {
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
  DvceArray6D<Real> tc, norm_to_tet_, tetcov_c_; DualArray2D<Real> nh_c_;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    nh_c_ = pm->pmb_pack->prad->nh_c;
    tc    = pm->pmb_pack->prad->tet_c;
    norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;
    tetcov_c_    = pm->pmb_pack->prad->tetcov_c;
  }

  // Reservoir state, captured by value into device lambdas
  const Real gm1 = bondi.gamma_adi - 1.0;
  const Real rho_res = bondi.rho0;
  const Real pgas_res = bondi.rho0 * bondi.t0;
  const Real bz_res = bondi.b0;
  // ambient radiation energy density (= a*T^4) for the equilibrium incoming intensity
  const Real urad_res = (is_radiation_enabled) ? bondi.arad*SQR(SQR(bondi.t0)) : 0.0;

  // X1-Boundary: uniform B_z on ghost faces (x1f=0, x2f=0, x3f=b0)
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
  // ConsToPrim over X1 ghost zones + innermost/outermost X1-active zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }
  // Dirichlet primitives in X1 ghost zones: (rho0, pgas_res, v=0)
  par_for("dirichlet_hydro_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        int ig = is - i - 1;
        w0_(m,IDN,k,j,ig) = rho_res;
        //w0_(m,IDN,k,j,ig) = w0_(m,IDN,k,j,is);
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
  // Radiation reservoir BC: OUTGOING directions copy the innermost active cell (escape);
  // INCOMING directions are set to the ambient at-rest equilibrium intensity (urad_res)
  // via BondiEquilibIntensity -- NOT zeroed (zeroing acted as a spurious sink).
  if (is_radiation_enabled) {
    par_for("dirichlet_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        Real n1 = tc(m,0,1,k,j,is)*nh0 + tc(m,1,1,k,j,is)*nh1 + tc(m,2,1,k,j,is)*nh2 + tc(m,3,1,k,j,is)*nh3;
        Real i_amb = BondiEquilibIntensity(m,k,j,is, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n1 > 0) ? i_amb : i0_(m,n,k,j,is);
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        Real n1 = tc(m,0,1,k,j,ie)*nh0 + tc(m,1,1,k,j,ie)*nh1 + tc(m,2,1,k,j,ie)*nh2 + tc(m,3,1,k,j,ie)*nh3;
        Real i_amb = BondiEquilibIntensity(m,k,j,ie, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n1 < 0) ? i_amb : i0_(m,n,k,j,ie);
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = val;
        }
      }
    });
  }
  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  }

  // X2-Boundary: uniform B_z on ghost faces (x1f=0, x2f=0, x3f=b0)
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
  // ConsToPrim over X2 ghost zones + innermost/outermost X2-active zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
  }
  // Dirichlet primitives in X2 ghost zones
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
        Real i_amb = BondiEquilibIntensity(m,k,js,i, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n2 > 0) ? i_amb : i0_(m,n,k,js,i);
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        Real n2 = tc(m,0,2,k,je,i)*nh0 + tc(m,1,2,k,je,i)*nh1 + tc(m,2,2,k,je,i)*nh2 + tc(m,3,2,k,je,i)*nh3;
        Real i_amb = BondiEquilibIntensity(m,k,je,i, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n2 < 0) ? i_amb : i0_(m,n,k,je,i);
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = val;
        }
      }
    });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  }

  // X3-Boundary: uniform B_z on ghost faces (x1f=0, x2f=0, x3f=b0)
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
  // ConsToPrim over X3 ghost zones + innermost/outermost X3-active zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }
  // Dirichlet primitives in X3 ghost zones
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
        Real i_amb = BondiEquilibIntensity(m,ks,j,i, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n3 > 0) ? i_amb : i0_(m,n,ks,j,i);
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        Real n3 = tc(m,0,3,ke,j,i)*nh0 + tc(m,1,3,ke,j,i)*nh1 + tc(m,2,3,ke,j,i)*nh2 + tc(m,3,3,ke,j,i)*nh3;
        Real i_amb = BondiEquilibIntensity(m,ke,j,i, nh0,nh1,nh2,nh3, urad_res,
                                           norm_to_tet_, tc, tetcov_c_);
        Real val = (n3 < 0) ? i_amb : i0_(m,n,ke,j,i);
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = val;
        }
      }
    });
  }
  // PrimToCons on X3 ghost zones
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

void BondiFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
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

  // Calculate conversion for P to e if using DynGRMHD.
  Real to_ien = 1.;
  if (pmbp->pdyngr != nullptr) {
    to_ien = 1.0 / (gamma - 1.);
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = (is_mhd) ? 4 : 3;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
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

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    if (is_mhd) pdata->hdata[nflux*g+3] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN)*to_ien;

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // compute mass flux
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

      // compute angular momentum flux
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;

      // compute magnetic flux
      if (is_mhd) {
        pdata->hdata[nflux*g+3] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
