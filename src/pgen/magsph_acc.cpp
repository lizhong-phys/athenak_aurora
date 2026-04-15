//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file magsph_acc.cpp
//! \brief Problem generator for relativistic magnetospheric accretion onto a neutron star.
//!        Initializes a dipole magnetic field in Schwarzschild metric (Wasserman & Shapiro
//!        1983), with a smooth interior mask field matched at the stellar surface, and a
//!        TOV stellar profile as the hydrostatic background.
//!
//! References:
//!   Wasserman & Shapiro 1983, ApJ, 265, 1036

// C headers
#include <stdio.h>
#include <math.h>

// MPI header (compiled in only when MPI is enabled)
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// C++ standard library headers
#include <algorithm>   // std::max(), std::min()
#include <iostream>    // std::cout, std::endl
#include <limits>      // std::numeric_limits
#include <memory>      // std::make_unique
#include <sstream>     // std::stringstream
#include <string>      // std::string
#include <vector>      // std::vector

// AthenaK core headers
#include "athena.hpp"           // Real, SQR, macros
#include "parameter_input.hpp"  // ParameterInput (reads .athinput)
#include "mesh/mesh.hpp"        // Mesh, MeshBlockPack

// Coordinate headers — Schwarzschild/GR metric, cell locations
#include "coordinates/coordinates.hpp"     // CoordData, is_general_relativistic
#include "coordinates/cartesian_ks.hpp"    // ComputeMetricAndInverse, Kerr-Schild coords
#include "coordinates/cell_locations.hpp"  // CellCenterX, LeftEdgeX

// Physics module headers
#include "eos/eos.hpp"       // equation of state (gamma, PrimToCons)
#include "hydro/hydro.hpp"   // Hydro module (pure hydro fallback)
#include "mhd/mhd.hpp"       // MHD module (u0, w0, b0, bcc0)
#include "radiation/radiation.hpp"

// Kokkos parallel execution and random number pool
#include <Kokkos_Random.hpp>


// Prototypes for helper functions used internally to this pgen
namespace {
  // container for physical parameters of magnetospheric accretion problem
  struct pgen_param {
    // neutron star parameters
    Real M_star = 3.1509328975826186e+19; // mass
    Real R_star = 5.804742809147471;      // radius
    Real rho_c = 3.710025734387253e+17;   // central density
    Real K_const = 7.166065217868681e-13; // polytropic coefficient
    Real rg = 1.;                         // gravitational radius
    Real gamma_poly = 5./3;               // polytropic index
    int  n_pts = 2000;                    // number of radial points for TOV integration

    // magnetosphere parameters
    Real B_star = 0.09409669397816478;    // surface magnetic field strength
    Real r_mask = 3.8698285394316474;     // mask radius
    Real mu_dipole; // dipole moment
    Real c1_dipole; // coefficient c1 for smooth internal dipole fields
    Real c2_dipole; // coefficient c2 for smooth internal dipole fields

    // gas
    Real gamma_adi = 5./3;
    Real dfloor = 1e-8;
    Real pfloor = 1e-12;
  };

  pgen_param pp;

  // helper functions to compute radial profiles of TOV star
  static void TOVEquations(const Real r, const Real mass, const Real press, const struct pgen_param pp, Real &dmass, Real &dpress);
  static void CalculateTOV(const struct pgen_param pp, HostArray1D<Real> &r_host, HostArray1D<Real> &rho_host, HostArray1D<Real> &press_host, HostArray1D<Real> &mass_host);

  //
  KOKKOS_INLINE_FUNCTION
  static void GetSchwarzschildCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi);
  KOKKOS_INLINE_FUNCTION
  Real A1dipole(struct pgen_param pp, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A2dipole(struct pgen_param pp, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A3dipole(struct pgen_param pp, Real x1, Real x2, Real x3);

} // end of namespace


// Prototypes for user-defined BCs and history functions
void NoInflowBC(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for relativistic magnetospheric accretion
//! Compile with '-D PROBLEM=gr_magnetosphere'

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  // 1. SANITY CHECKS
  //    - Verify GR is enabled in <coord> block
  //    - Verify MHD is being used (pmbp->pmhd != nullptr)
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "magnetospheric_accretion problem can only be run with GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "magnetospheric_accretion problem can only be run with MHD"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // 2. ENROLL USER-DEFINED FUNCTIONS
  //    - user_bcs_func  : stellar surface boundary condition / no-inflow
  user_bcs_func = NoInflowBC;

  // 3. READ PARAMETERS FROM INPUT FILE
  //    - Neutron star: radius R*, surface field B*, spin Omega*
  //    - Mask radius R_m (interior/exterior field matching radius)
  //    - EOS: gamma, K (polytropic)
  //    - Atmosphere floors: rho_floor, press_floor

  // 4. LOAD TOV STELLAR PROFILE
  //    - Compute TOV solution inline (RK4) or read from external table
  //    - Store rho(r), P(r), m(r) arrays accessible inside Kokkos kernels
  HostArray1D<Real> r_host, rho_host, press_host, mass_host;
  Kokkos::realloc(r_host,     pp.n_pts);
  Kokkos::realloc(rho_host,   pp.n_pts);
  Kokkos::realloc(press_host, pp.n_pts);
  Kokkos::realloc(mass_host,  pp.n_pts);
  CalculateTOV(pp, r_host, rho_host, press_host, mass_host);

  DvceArray1D<Real> r_tov, rho_tov, press_tov;
  Kokkos::realloc(r_tov,     pp.n_pts);
  Kokkos::realloc(rho_tov,   pp.n_pts);
  Kokkos::realloc(press_tov, pp.n_pts);
  Kokkos::deep_copy(r_tov,     r_host);
  Kokkos::deep_copy(rho_tov,   rho_host);
  Kokkos::deep_copy(press_tov, press_host);

  // 5. COMPUTE MAGNETIC FIELD PARAMETERS
  //    - Compute magnetic moment mu from B* and R* (Eq. 2)
  //    - Evaluate A(R_m) and A'(R_m) from exterior potential (Eq. 6)
  //    - Compute mask coefficients C1, C2 at R_m (Eq. 5)
  pp.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  pp.mu_dipole = -4./3*pp.rg*SQR(pp.rg)*pp.B_star / (log(1.-2./pp.R_star) + 2./pp.R_star + 2./SQR(pp.R_star));

  Real coeff_a  = SQR(pp.r_mask)*log(1.-2./pp.r_mask) + 2*pp.r_mask + 2;
  Real coeff_ap = 2*pp.r_mask*log(1.-2./pp.r_mask) + 4./(pp.r_mask-2) + 4;
  pp.c1_dipole = -3./8*pp.mu_dipole/pp.rg * (4*coeff_a-pp.r_mask*coeff_ap) / (2*SQR(pp.r_mask));
  pp.c2_dipole = -3./8*pp.mu_dipole/pp.rg * (pp.r_mask*coeff_ap-2*coeff_a) / (2*SQR(SQR(pp.r_mask)));

  // 6. RETURN EARLY IF RESTART
  if (restart) return;

  // 7. INITIALIZE PRIMITIVE VARIABLES
  //    Loop over all cells (Kokkos parallel_for):
  //    - Compute (r, theta) from mesh coordinates
  //    - If r > R* : set tenuous atmosphere (rho_floor, press_floor, u^i = 0)
  //    - If r <= R*: set TOV profile (rho, P interpolated from table), u^i = 0

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // MHD variables
  DvceArray5D<Real> u0_ = pmbp->pmhd->u0;
  DvceArray5D<Real> w0_ = pmbp->pmhd->w0;

  // Get ideal gas EOS data



  // initialize primitive variables for new run ---------------------------------------

  auto &size = pmbp->pmb->mb_size;
  // const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  // const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  // const int nji  = indcs.nx2*indcs.nx1;

  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;
  par_for("pgen_star_ini", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Calculate Schwarzschild coordinates of cell
    Real r, theta, phi;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &r, &theta, &phi);

    // Interpolate rho_tov, press_tov
    Real rho_interp, press_interp;
    Real r0_tov = r_tov(0);
    Real r1_tov = r_tov(pp_dvce.n_pts-1);

    if (r <= r0_tov) {
      // inside innermost point, use central values
      rho_interp   = rho_tov(0);
      press_interp = press_tov(0);
    } else if (r >= r1_tov) {
      // outside stellar surface, use floor values
      rho_interp   = pp_dvce.dfloor;
      press_interp = pp_dvce.pfloor;
    } else {
      // linear interpolation
      Real dr_tov = (r1_tov - r0_tov) / (pp_dvce.n_pts - 1);
      int idx = static_cast<int>((r - r0_tov) / dr_tov);
      idx = (idx < 0) ? 0 : ((idx >= pp_dvce.n_pts-1) ? pp_dvce.n_pts-2 : idx);
      Real frac = (r - r_tov(idx)) / (r_tov(idx+1) - r_tov(idx));
      rho_interp   = (1.0 - frac) * rho_tov(idx)   + frac * rho_tov(idx+1);
      press_interp = (1.0 - frac) * press_tov(idx) + frac * press_tov(idx+1);
    }
    rho_interp   = fmax(rho_interp,   pp_dvce.dfloor);
    press_interp = fmax(press_interp, pp_dvce.pfloor);

    // Set primitive values
    w0_(m,IDN,k,j,i) = rho_interp;
    w0_(m,IPR,k,j,i) = press_interp/gm1;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;

  }); // end par_for "pgen_star_ini"



  // 8. INITIALIZE MAGNETIC FIELD
  //    Loop over all faces (Kokkos parallel_for):
  //    - Compute vector potential A_phi at each face location
  //      * r > R_m  : Wasserman & Shapiro exterior dipole (Eq. 1)
  //      * r <= R_m : interior mask polynomial (Eq. 4)
  //    - Compute face-centered B^i = curl(A) via finite differences
  //    - Compute cell-centered bcc as average of face values



  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;
  par_for("pgen_magdipole_ini", DevExeSpace(), 0,nmb-1, ks,ke+1, js,je+1, is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX(i  -is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX(j  -js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX(k  -ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1dipole(pp_dvce, x1v, x2f, x3f);
    a2(m,k,j,i) = A2dipole(pp_dvce, x1f, x2v, x3f);
    a3(m,k,j,i) = A3dipole(pp_dvce, x1f, x2f, x3v);

    // TODO: add disk field later

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
      a1(m,k,j,i) = 0.5*(A1dipole(pp_dvce, xl, x2f, x3f) + A1dipole(pp_dvce, xr, x2f, x3f));
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
      a2(m,k,j,i) = 0.5*(A2dipole(pp_dvce, x1f, xl, x3f) + A2dipole(pp_dvce, x1f, xr, x3f));
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
      a3(m,k,j,i) = 0.5*(A3dipole(pp_dvce, x1f, x2f, xl) + A3dipole(pp_dvce, x1f, x2f, xr));
    }
  }); // end par_for "pgen_magdipole_ini"

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_b0", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
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
  }); // end par_for "pgen_b0"


  // Compute cell-centered fields
  auto &bcc0_ = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc0_(m,IBX,k,j,i);
    Real& w_by = bcc0_(m,IBY,k,j,i);
    Real& w_bz = bcc0_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  }); // end par_for "pgen_bcc"





  // 9. CONVERT PRIMITIVES TO CONSERVED VARIABLES
  //    - Call pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, ...)
  pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);

  return;
} // end ProblemGenerator::UserProblem



namespace {
//----------------------------------------------------------------------------------------
// Helper function to compute TOV star profiles
static void TOVEquations(const Real r, const Real mass, const Real press,
                         const struct pgen_param pp,
                         Real &dmass, Real &dpress) {

  if (press <= pp.pfloor) {
    dmass = 0.0; dpress = 0.0;
    return;
  }

  Real rho = pow(press/pp.K_const, 1.0/pp.gamma_poly);
  Real eps = rho + press/(pp.gamma_poly-1.0);
  dmass  = 4.0*M_PI * SQR(r) * eps;
  dpress = -(eps+press)/pp.M_star * (mass+4.0*M_PI*r*SQR(r)*press) / (r*(r-2.0*mass/pp.M_star));
}

static void CalculateTOV(const struct pgen_param pp, HostArray1D<Real> &r_host,
                         HostArray1D<Real> &rho_host, HostArray1D<Real> &press_host,
                         HostArray1D<Real> &mass_host) {
  Real r0 = fmin(1.0e-6, 1e-2*pp.R_star/pp.n_pts);
  Real dr = (pp.R_star - r0) / (pp.n_pts - 1);

  // Initial conditions
  Real press_c  = pp.K_const * pow(pp.rho_c, pp.gamma_poly);
  Real eps_c = pp.rho_c + press_c / (pp.gamma_poly-1.0);
  r_host(0)     = r0;
  mass_host(0)  = (4.0/3.0) * M_PI * r0*SQR(r0) * eps_c;
  press_host(0) = press_c;
  rho_host(0)   = pp.rho_c;

  for (int i=0; i < pp.n_pts-1; i++) {
    Real r_ = r_host(i);
    Real m_ = mass_host(i);
    Real p_ = press_host(i);

    // RK4 integration
    Real dm1, dp1, dm2, dp2, dm3, dp3, dm4, dp4;
    TOVEquations(r_,        m_,            p_,            pp, dm1, dp1);
    TOVEquations(r_+0.5*dr, m_+0.5*dr*dm1, p_+0.5*dr*dp1, pp, dm2, dp2);
    TOVEquations(r_+0.5*dr, m_+0.5*dr*dm2, p_+0.5*dr*dp2, pp, dm3, dp3);
    TOVEquations(r_+dr,     m_+dr*dm3,     p_+dr*dp3,     pp, dm4, dp4);

    r_host(i+1)     = r_ + dr;
    mass_host(i+1)  = m_ + (dr/6.0) * (dm1 + 2.0*dm2 + 2.0*dm3 + dm4);
    press_host(i+1) = fmax(p_ + (dr/6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4), pp.pfloor);
    rho_host(i+1)   = pow(press_host(i+1)/pp.K_const, 1.0/pp.gamma_poly);

    // stop if pressure vanishes
    if (press_host(i+1) <= pp.pfloor) {
      for (int j=i+2; j < pp.n_pts; j++) {
        r_host(j)     = r_host(i+1) + (j-i-1)*dr;
        mass_host(j)  = mass_host(i+1);
        press_host(j) = pp.pfloor;
        rho_host(j)   = pp.dfloor;
      }
      break;
    }
  } // endfor i

  // diagnostic output
  std::cout << "TOV integration complete: M(R*) = " << mass_host(pp.n_pts-1)
            << ", R* = " << r_host(pp.n_pts-1) << std::endl;
}

// convert CKS to Schwarzschild coordinates
KOKKOS_INLINE_FUNCTION
static void GetSchwarzschildCoordinates(Real x1, Real x2, Real x3,
                                        Real *pr, Real *ptheta, Real *pphi) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  r = fmax(r, 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(x2, x1);
  return;
}

KOKKOS_INLINE_FUNCTION
Real A1dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  // Schwarzschild coordinates
  Real r, theta, phi;
  GetSchwarzschildCoordinates(x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential
  Real aphi_rcomp = (r > pp.r_mask) ? -3./8*pp.mu_dipole/pp.rg * (SQR(r)*log(1.-2./r) + 2.*r + 2.)
                                    : pp.c1_dipole*SQR(r) + pp.c2_dipole*SQR(SQR(r));
  Real aphi = SQR(sin(theta)) * aphi_rcomp;
  return aphi * (-x2 / (SQR(x1)+SQR(x2)));
}


KOKKOS_INLINE_FUNCTION
Real A2dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  // Schwarzschild coordinates
  Real r, theta, phi;
  GetSchwarzschildCoordinates(x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential
  Real aphi_rcomp = (r > pp.r_mask) ? -3./8*pp.mu_dipole/pp.rg * (SQR(r)*log(1.-2./r) + 2.*r + 2.)
                                    : pp.c1_dipole*SQR(r) + pp.c2_dipole*SQR(SQR(r));
  Real aphi = SQR(sin(theta)) * aphi_rcomp;
  return aphi * (x1 / (SQR(x1)+SQR(x2)));
}

KOKKOS_INLINE_FUNCTION
Real A3dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  return 0.0;
}









} // end of namespace

//----------------------------------------------------------------------------------------
//! \fn NoInflowTorus
//  \brief Sets boundary condition on surfaces of computational domain
void NoInflowBC(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &ie  = indcs.ie;
  int &js = indcs.js; int &je  = indcs.je;
  int &ks = indcs.ks; int &ke  = indcs.ke;
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
  int nvar = u0_.extent_int(1);

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);
        } // endfor i
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);
        } // endfor i
      }
    });
  } // endif (pm->pmb_pack->pmhd != nullptr)

  // TODO: reformat everything below

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = i0_(m,n,k,j,ie);
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

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
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
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
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

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
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
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
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
} // end NoInflowBC





//
