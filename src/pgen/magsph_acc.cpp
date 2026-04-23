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
#include "eos/eos.hpp"           // equation of state (gamma, PrimToCons)
#include "hydro/hydro.hpp"       // Hydro module (pure hydro fallback)
#include "mhd/mhd.hpp"           // MHD module (u0, w0, b0, bcc0)
#include "radiation/radiation.hpp"

// Kokkos parallel execution and random number pool
#include <Kokkos_Random.hpp>


// Prototypes for helper functions used internally to this pgen
namespace {

  // helper functions to compute radial profiles of TOV star
  static void TOVEquations(const Real r, const Real mass, const Real press, const struct pgen_param pp, Real &dmass, Real &dpress);
  static void CalculateTOV(const struct pgen_param pp, HostArray1D<Real> &r_host, HostArray1D<Real> &rho_host, HostArray1D<Real> &press_host, HostArray1D<Real> &mass_host);

  // compute magnetosphere parameters
  // KOKKOS_INLINE_FUNCTION
  // Real ComputeMagParam(Real x1, Real x2, Real x3, Real Omega, Real r_surf, Real rho_surf, Real h_surf);

  // compute neutron star atmosphere
  KOKKOS_INLINE_FUNCTION
  Real ComputeAtmosphere(Real x1, Real x2, Real x3, Real Omega, Real r_surf, Real rho_surf, Real h_surf);

  // helper functions to transform coordinates
  KOKKOS_INLINE_FUNCTION
  static void GetSchwarzschildCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi);

  // helper functions to compute dipole magnetic fields
  KOKKOS_INLINE_FUNCTION
  Real A1dipole(struct pgen_param pp, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A2dipole(struct pgen_param pp, Real x1, Real x2, Real x3);
  KOKKOS_INLINE_FUNCTION
  Real A3dipole(struct pgen_param pp, Real x1, Real x2, Real x3);

  // container for physical parameters of magnetospheric accretion problem
  struct pgen_param {
    // neutron star parameters
    Real M_star;        // mass
    Real R_star;        // radius
    Real B_star;        // magnetic field strength
    Real Omega_star;    // angular speed
    Real r_mask;        // mask radius

    // unit parameters
    Real rg;            // gravitational radius
    Real rho_unit;      // density unit
    Real mu;            // molecular weight

    // atmosphere parameters
    Real rho_surf;      // density of atmosphere bottom
    Real tgas_surf;     // atmosphere temperature
    Real r_surf;        // radius of atmosphere bottom
    Real rmax_atm_pole; // radius of atmosphere top at pole
    Real rmax_atm_eqtr; // radius of atmosphere top at equator
    Real h_surf;        // minimum required scale height

    // magnetosphere parameters
    Real r_core;        // critial radius for interior dipole fields
    Real mu_dipole;     // dipole moment
    Real c1_dipole;     // coefficient c1 for smooth interior dipole fields
    Real c2_dipole;     // coefficient c2 for smooth interior dipole fields

    // gas parameters
    Real gamma_adi;
    Real sigma_max;
    Real dfloor;
    Real pfloor;

    // tov star parameters
    bool use_tov;    // initialize with tov star
    Real rho_c;      // central density
    Real K_const;    // polytropic coefficient
    Real gamma_poly; // polytropic index
    int  n_pts;      // number of radial points for TOV integration
  };

  pgen_param pp;

} // end of namespace

// Prototypes for user-defined source functions
void MySourceTerms(Mesh* pm, const Real bdt);
void ReducedGravSrcTerm(Mesh* pm, const Real bdt);
void NeutronStarMask(Mesh* pm, const Real bdt);

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
  user_srcs_func = MySourceTerms;

  // 3. READ PARAMETERS FROM INPUT FILE
  //    - Neutron star: radius R*, surface field B*, spin Omega*
  //    - Mask radius R_m (interior/exterior field matching radius)
  //    - EOS: gamma, K (polytropic)
  //    - Atmosphere floors: rho_floor, press_floor

  // Default setting:
  // M_star = 1.4 M_sun; R_star = 1.2 km; B_star = 1e9 G; Omega_star = 0;
  // length_unit = rg; rho_unit = 0.01 g/cm3;
  // TODO... (for atmosphere resolution 0.03125rg/cell)

  // neutron star parameters
  pp.M_star        = pin->GetOrAddReal("problem", "M_star",     3.1509328975826186e+19);
  pp.R_star        = pin->GetOrAddReal("problem", "R_star",     5.804742809147471);
  pp.B_star        = pin->GetOrAddReal("problem", "B_star",     0.09409669397816478);
  pp.Omega_star    = pin->GetOrAddReal("problem", "Omega_star", 0.0);
  pp.r_mask        = pin->GetOrAddReal("problem", "r_mask",     0.8*pp.R_star);

  // unit parameters
  pp.rg            = pin->GetOrAddReal("units", "rg",          1.0);
  pp.rho_unit      = pin->GetOrAddReal("units", "density_cgs", 0.01);
  pp.mu            = pin->GetOrAddReal("units", "mu",          0.5991611743559018);

  // atmosphere parameters
  pp.rho_surf      = pin->GetOrAddReal("problem", "rho_surf",      0.007927819363347719);
  pp.tgas_surf     = pin->GetOrAddReal("problem", "tgas_surf",     1.5328527189503498e-07);
  pp.r_surf        = pin->GetOrAddReal("problem", "r_surf",        0.8*pp.R_star);
  pp.rmax_atm_pole = pin->GetOrAddReal("problem", "rmax_atm_pole", 1.1*pp.R_star);
  pp.rmax_atm_eqtr = pin->GetOrAddReal("problem", "rmax_atm_eqtr", 1.1*pp.R_star);
  pp.h_surf        = pin->GetOrAddReal("problem", "h_surf",        5.0*0.03125);

  // magnetosphere parameters
  pp.r_core        = pin->GetOrAddReal("problem", "r_core", 0.5*pp.R_star);

  // gas parameters
  pp.gamma_adi     = pin->GetOrAddReal("mhd", "gamma_adi", 5./3);
  pp.dfloor        = pin->GetOrAddReal("mhd", "dfloor",    1.0e-08);
  pp.pfloor        = pin->GetOrAddReal("mhd", "pfloor",    1.0e-15);
  pp.sigma_max     = pin->GetOrAddReal("mhd", "sigma_max", 1000.0);

  // tov star parameters
  pp.use_tov       = pin->GetOrAddBoolean("problem", "use_tov",    false);
  pp.rho_c         = pin->GetOrAddReal(   "problem", "rho_c",      3.710025734387253e+17);
  pp.K_const       = pin->GetOrAddReal(   "problem", "K_const",    7.166065217868681e-13);
  pp.gamma_poly    = pin->GetOrAddReal(   "problem", "gamma_poly", 5./3);
  pp.n_pts         = pin->GetOrAddInteger("problem", "n_pts",      2000);

  // 4. LOAD TOV STELLAR PROFILE
  //    - Compute TOV solution inline (RK4) or read from external table
  //    - Store rho(r), P(r), m(r) arrays accessible inside Kokkos kernels
  HostArray1D<Real> r_host, rho_host, press_host, mass_host;
  DvceArray1D<Real> r_tov, rho_tov, press_tov;
  if (pp.use_tov) {
    int &n_pts = pp.n_pts;
    Kokkos::realloc(r_host,     n_pts);
    Kokkos::realloc(rho_host,   n_pts);
    Kokkos::realloc(press_host, n_pts);
    Kokkos::realloc(mass_host,  n_pts);
    CalculateTOV(pp, r_host, rho_host, press_host, mass_host);

    Kokkos::realloc(r_tov,     n_pts);
    Kokkos::realloc(rho_tov,   n_pts);
    Kokkos::realloc(press_tov, n_pts);
    Kokkos::deep_copy(r_tov,     r_host);
    Kokkos::deep_copy(rho_tov,   rho_host);
    Kokkos::deep_copy(press_tov, press_host);
  }


  // 5. COMPUTE MAGNETIC FIELD PARAMETERS
  //    - Compute magnetic moment mu from B* and R* (Eq. 2)
  //    - Evaluate A(R_m) and A'(R_m) from exterior potential (Eq. 6)
  //    - Compute mask coefficients C1, C2 at R_m (Eq. 5)
  {
    const Real &rg=pp.rg, &r_core=pp.r_core;
    const Real &B_star=pp.B_star, &R_star=pp.R_star;
    Real rg3 = rg*SQR(rg);
    Real R_star2 = SQR(R_star);
    Real r_core2 = SQR(r_core);

    pp.mu_dipole = -4./3*rg3*B_star / (log(1.-2./R_star) + 2./R_star + 2./R_star2);
    Real coeff_a  = r_core2*log(1.-2./r_core) + 2*r_core + 2;
    Real coeff_ap = 2*r_core*log(1.-2./r_core) + 4./(r_core-2) + 4;
    pp.c1_dipole = -3./8*pp.mu_dipole/rg * (4*coeff_a-r_core*coeff_ap) / (2*r_core2);
    pp.c2_dipole = -3./8*pp.mu_dipole/rg * (r_core*coeff_ap-2*coeff_a) / (2*SQR(r_core2));
  }

  // 6. RETURN EARLY IF RESTART
  if (restart) return;

  // 7. INITIALIZE PRIMITIVE VARIABLES
  //    Loop over all cells (Kokkos parallel_for):
  //    - Compute (r, theta) from mesh coordinates
  //    - If r > R* : set tenuous atmosphere (rho_floor, press_floor, u^i = 0)
  //    - If r <= R*: set TOV profile (rho, P interpolated from table), u^i = 0

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  int &nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // MHD variables
  DvceArray5D<Real> u0_ = pmbp->pmhd->u0;
  DvceArray5D<Real> w0_ = pmbp->pmhd->w0;

  // Get ideal gas EOS data



  // initialize primitive variables for new run ---------------------------------------


  // const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  // const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  // const int nji  = indcs.nx2*indcs.nx1;

  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;
  par_for("pgen_star_ini", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
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

    // compute local metric
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, false, 0.0, glower, gupper);

    // calculate Schwarzschild coordinates at the cell center
    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);

    // extract problem parameters
    const Real &R_star = pp_dvce.R_star, &B_star = pp_dvce.B_star, &Omg_star = pp_dvce.Omega_star;
    const Real &r_surf = pp_dvce.r_surf, &h_surf = pp_dvce.h_surf;
    const Real &rho_surf = pp_dvce.rho_surf, &tgas_surf = pp_dvce.tgas_surf;
    const Real &rmax_atm_eqtr = pp_dvce.rmax_atm_eqtr;
    const Real &sigma_max = pp_dvce.sigma_max;

    // compute local floors
    Real fac_reduce = (R_star/rv)*SQR(R_star/rv);
    Real dfloor = fmax(pp_dvce.dfloor, SQR(B_star*fac_reduce)/sigma_max);
    Real pfloor = pp_dvce.pfloor;

    // initialize gas profiles
    Real dens=dfloor, pgas=pfloor;
    Real uu1=0.0, uu2=0.0, uu3=0.0;

    // TOV star
    if (pp_dvce.use_tov) {
      const int &n_pts = pp_dvce.n_pts;
      Real r0_tov = r_tov(0);
      Real r1_tov = r_tov(n_pts-1);

      if (rv <= r0_tov) {
        // inside innermost point, use central values
        dens = rho_tov(0);
        pgas = press_tov(0);
      } else if (rv >= r1_tov) {
        // outside stellar surface, use floor values
        dens = dfloor;
        pgas = pfloor;
      } else {
        // linear interpolation
        Real dr_tov = (r1_tov - r0_tov) / (n_pts - 1);
        int idx = static_cast<int>((rv - r0_tov) / dr_tov);
        idx = (idx < 0) ? 0 : ((idx >= n_pts-1) ? n_pts-2 : idx);
        Real frac = (rv - r_tov(idx)) / (r_tov(idx+1) - r_tov(idx));
        dens = (1.0 - frac) * rho_tov(idx)   + frac * rho_tov(idx+1);
        pgas = (1.0 - frac) * press_tov(idx) + frac * press_tov(idx+1);
      }
      dens = fmax(dens, dfloor);
      pgas = fmax(pgas, pfloor);
    } // endif (pp_dvce.use_tov)

    // below the atmosphere (r < r_surf)
    if (rv < r_surf) {
      dens = rho_surf;
      pgas = rho_surf*tgas_surf;
      if (rv > 2.0) {
        // Keplerian speed is only valid when r > 2rg
        Real u0 = 1./sqrt(1. - 2./rv - SQR(Omg_star*rv*sin(thv)));
        Real u1 = -Omg_star*x2v*u0;
        Real u2 =  Omg_star*x1v*u0;
        Real u3 = 0.0;
        // convert velocity from coordinate frame to normal frame
        uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
        uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
        uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
      }
    } // endif below the atmosphere

    // within the atmosphere (r_surf <= r <= ~rmax_atm_eqtr)
    Real rv6 = rv*rv*rv*rv*rv*rv;
    Real R_star6 = SQR(R_star*R_star*R_star);
    Real lmd = -(r_surf-2.)*r_surf;
    Real pw_base = (1. - 2./rv - SQR(Omg_star*rv*sin(thv))) / (1 - 2./r_surf);
    Real pw_idx = 0.5*lmd/h_surf;
    bool below_atm = (rv6*pow(pw_base, pw_idx) >= SQR(B_star)*R_star6/(rho_surf*sigma_max));
    if (below_atm && (rv >= r_surf) && (rv <= rmax_atm_eqtr)) {
      dens = fmax(ComputeAtmosphere(x1v, x2v, x3v, Omg_star, r_surf, rho_surf, h_surf), dfloor);
      pgas = fmax(dens*tgas_surf, pfloor);
      if (rv > 2.0) {
        // Keplerian speed is only valid when r > 2rg
        Real u0 = 1./sqrt(1. - 2./rv - SQR(Omg_star*rv*sin(thv)));
        Real u1 = -Omg_star*x2v*u0;
        Real u2 =  Omg_star*x1v*u0;
        Real u3 = 0.0;
        // convert velocity from coordinate frame to normal frame
        uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
        uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
        uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
      }
    } // endif within the atmosphere

    // disk
    // TODO: implement disk initialization


    // set primitive values
    w0_(m,IDN,k,j,i) = dens;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;
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
  par_for("pgen_magdipole_ini", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
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
    Real &w_bx = bcc0_(m,IBX,k,j,i);
    Real &w_by = bcc0_(m,IBY,k,j,i);
    Real &w_bz = bcc0_(m,IBZ,k,j,i);
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
  int  n_pts   = pp.n_pts;   Real R_star = pp.R_star;
  Real dfloor  = pp.dfloor;  Real pfloor = pp.pfloor;
  Real K_const = pp.K_const; Real rho_c  = pp.rho_c; Real gamma_poly = pp.gamma_poly;

  Real r0 = fmin(1.0e-6, 1e-2*R_star/n_pts);
  Real dr = (R_star - r0) / (n_pts - 1);

  // Initial conditions
  Real press_c  = K_const * pow(rho_c, gamma_poly);
  Real eps_c = rho_c + press_c / (gamma_poly-1.0);
  r_host(0)     = r0;
  mass_host(0)  = (4.0/3.0) * M_PI * r0*SQR(r0) * eps_c;
  press_host(0) = press_c;
  rho_host(0)   = rho_c;

  for (int i=0; i < n_pts-1; i++) {
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
    press_host(i+1) = fmax(p_ + (dr/6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4), pfloor);
    rho_host(i+1)   = pow(press_host(i+1)/K_const, 1.0/gamma_poly);

    // stop if pressure vanishes
    if (press_host(i+1) <= pfloor) {
      for (int j=i+2; j < n_pts; j++) {
        r_host(j)     = r_host(i+1) + (j-i-1)*dr;
        mass_host(j)  = mass_host(i+1);
        press_host(j) = pfloor;
        rho_host(j)   = dfloor;
      }
      break;
    }
  } // endfor i

  // diagnostic output
  std::cout << "TOV integration complete: M(R*) = " << mass_host(n_pts-1)
            << ", R* = " << r_host(n_pts-1) << std::endl;
}

// compute neutron star atmosphere
KOKKOS_INLINE_FUNCTION
Real ComputeAtmosphere(Real x1, Real x2, Real x3, Real Omega, Real r_surf, Real rho_surf, Real h_surf) {
  // coordinate convertion
  Real r, theta, phi;
  GetSchwarzschildCoordinates(x1, x2, x3, &r, &theta, &phi);

  // compute density profile
  Real part1  = 1. - 2./r - SQR(Omega*r*sin(theta));
  Real part2  = 1. - 2./r_surf;
  Real pw_idx = -r_surf/h_surf * (0.5*r_surf-1.0);

  return rho_surf*pow(part1/part2, pw_idx);
}

// convert CKS to Schwarzschild coordinates
KOKKOS_INLINE_FUNCTION
static void GetSchwarzschildCoordinates(Real x1, Real x2, Real x3,
                                        Real *pr, Real *ptheta, Real *pphi) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(x2, x1);
  return;
}

KOKKOS_INLINE_FUNCTION
Real A1dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  Real &rg=pp.rg, &r_core=pp.r_core;
  Real &mu_dipole=pp.mu_dipole;
  Real &c1_dipole=pp.c1_dipole;
  Real &c2_dipole=pp.c2_dipole;

  // Schwarzschild coordinates
  Real r, theta, phi;
  GetSchwarzschildCoordinates(x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential
  Real r2 = SQR(r);
  Real aphi_rcomp = (r > r_core) ? -3./8*mu_dipole/rg * (r2*log(1.-2./r) + 2.*r + 2.)
                                 : c1_dipole*r2 + c2_dipole*SQR(r2);
  Real aphi = SQR(sin(theta)) * aphi_rcomp;
  return aphi * (-x2 / (SQR(x1)+SQR(x2)));
}

KOKKOS_INLINE_FUNCTION
Real A2dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  Real &rg=pp.rg, &r_core=pp.r_core;
  Real &mu_dipole=pp.mu_dipole;
  Real &c1_dipole=pp.c1_dipole;
  Real &c2_dipole=pp.c2_dipole;

  // Schwarzschild coordinates
  Real r, theta, phi;
  GetSchwarzschildCoordinates(x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential
  Real r2 = SQR(r);
  Real aphi_rcomp = (r > pp.r_core) ? -3./8*mu_dipole/rg * (r2*log(1.-2./r) + 2.*r + 2.)
                                    : c1_dipole*r2 + c2_dipole*SQR(r2);
  Real aphi = SQR(sin(theta)) * aphi_rcomp;
  return aphi * (x1 / (SQR(x1)+SQR(x2)));
}

KOKKOS_INLINE_FUNCTION
Real A3dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  return 0.0;
}









} // end of namespace




void MySourceTerms(Mesh* pm, const Real bdt) {

  ReducedGravSrcTerm(pm, bdt);

  NeutronStarMask(pm, bdt);

  return;
}


void ReducedGravSrcTerm(Mesh* pm, const Real bdt) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // MHD variables
  DvceArray5D<Real> u0_, w0_;
  u0_   = pmbp->pmhd->u0;
  w0_   = pmbp->pmhd->w0;

  // capture problem parameters
  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;
  par_for("pgen_gravsrc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // calculate Schwarzschild coordinates at the cell center
    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);
    if ((rv < pp_dvce.r_surf) || (rv > pp_dvce.rmax_atm_eqtr)) return;

    // extract problem paramters
    const Real &R_star=pp_dvce.R_star, &B_star=pp_dvce.B_star, &Omg_star=pp_dvce.Omega_star;
    const Real &r_surf=pp_dvce.r_surf, &h_surf=pp_dvce.h_surf;
    const Real &rho_surf=pp_dvce.rho_surf;
    const Real &rmax_atm_eqtr=pp_dvce.rmax_atm_eqtr;
    const Real &sigma_max=pp_dvce.sigma_max;
    Real rv3 = rv*SQR(rv); Real rv6 = SQR(rv3);
    Real R_star6 = SQR(R_star*R_star*R_star);

    Real lmd = -(r_surf-2.)*r_surf;
    Real pw_base = (1. - 2./rv - SQR(Omg_star*rv*sin(thv))) / (1 - 2./r_surf);
    Real pw_idx = 0.5*lmd/h_surf;
    bool below_atm = (rv6*pow(pw_base, pw_idx) >= SQR(B_star)*R_star6/(rho_surf*sigma_max));
    if (below_atm && (rv >= r_surf) && (rv <= rmax_atm_eqtr)) {
      Real dens = w0_(m,IDN,k,j,i);
      Real pgas = w0_(m,IEN,k,j,i)*gm1;
      Real ut2 = 1./(1. - 2./rv - SQR(Omg_star*rv*sin(thv)));
      Real wtot = dens + pgas/gm1 + pgas + lmd*pgas/h_surf;

      Real srcx = wtot*ut2 * (1./rv3 - SQR(Omg_star))*x1v;
      Real srcy = wtot*ut2 * (1./rv3 - SQR(Omg_star))*x2v;
      Real srcz = wtot*ut2 * (1./rv3)*x3v;

      u0_(m,IM1,k,j,i) -= bdt*srcx;
      u0_(m,IM2,k,j,i) -= bdt*srcy;
      u0_(m,IM3,k,j,i) -= bdt*srcz;
    } // endif
  }); // end par_for
} // end ReducedGravSrcTerm


void NeutronStarMask(Mesh* pm, const Real bdt) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // MHD variables
  DvceArray5D<Real> u0_, w0_, bcc0_;
  u0_   = pmbp->pmhd->u0;
  w0_   = pmbp->pmhd->w0;
  bcc0_ = pmbp->pmhd->bcc0;

  // capture problem parameters
  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;
  par_for("pgen_nsmask", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // calculate Schwarzschild coordinates at the cell center
    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);

    // extract problem parameters
    const Real &r_mask=pp_dvce.r_mask;
    const Real &R_star=pp_dvce.R_star, &B_star=pp_dvce.B_star, &Omg_star=pp_dvce.Omega_star;
    const Real &r_surf=pp_dvce.r_surf, &h_surf=pp_dvce.h_surf;
    const Real &rho_surf=pp_dvce.rho_surf, &tgas_surf=pp_dvce.tgas_surf;
    const Real &sigma_max=pp_dvce.sigma_max;

    // compute local floors
    Real fac_reduce = (R_star/rv)*SQR(R_star/rv);
    Real dfloor = fmax(pp_dvce.dfloor, SQR(B_star*fac_reduce)/sigma_max);
    Real pfloor = pp_dvce.pfloor;

    if (rv <= r_mask) {
      Real dens = (rv < r_surf) ? rho_surf :
                  ComputeAtmosphere(x1v, x2v, x3v, Omg_star, r_surf, rho_surf, h_surf);
      dens = fmax(dens, dfloor);
      Real pgas = fmax(dens*tgas_surf, pfloor);
      Real bx = bcc0_(m,IBX,k,j,i);
      Real by = bcc0_(m,IBY,k,j,i);
      Real bz = bcc0_(m,IBZ,k,j,i);

      // gas velocity in the normal frame
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, false, 0.0, glower, gupper);

      Real u0=1.0, u1=0.0, u2=0.0, u3=0.0;
      if (rv > 2.0) {
        u0 = 1./sqrt(1. - 2./rv - SQR(Omg_star*rv*sin(thv)));
        u1 = -Omg_star*x2v*u0;
        u2 =  Omg_star*x1v*u0;
        u3 = 0.0;
      }

      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      Real b0 = u_1*bx + u_2*by + u_3*bz;
      Real b1 = (bx + b0*u1) / u0;
      Real b2 = (by + b0*u2) / u0;
      Real b3 = (bz + b0*u3) / u0;

      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      Real wtot = dens + (gm1+1)/gm1*pgas + b_sq;
      Real ptot = pgas + 0.5*b_sq;
      u0_(m,IDN,k,j,i) = dens*u0;
      u0_(m,IEN,k,j,i) = wtot*u0*u_0 - b0*b_0 + ptot + dens*u0;
      u0_(m,IM1,k,j,i) = wtot*u0*u_1 - b0*b_1;
      u0_(m,IM2,k,j,i) = wtot*u0*u_2 - b0*b_2;
      u0_(m,IM3,k,j,i) = wtot*u0*u_3 - b0*b_3;
    } // endif
  }); // end par_for
} // end NeutronStarMask



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
