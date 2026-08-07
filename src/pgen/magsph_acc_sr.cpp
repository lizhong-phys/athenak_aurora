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

  // helper functions to compute outer torus
  KOKKOS_INLINE_FUNCTION
  static void FindTorusOuterEdge(struct pgen_param pgen, Real *r_outer_edge);
  KOKKOS_INLINE_FUNCTION
  static void CalculateCN(struct pgen_param pgen, Real *cparam, Real *nparam);
  KOKKOS_INLINE_FUNCTION
  static Real CalculateL(struct pgen_param pgen, Real r, Real sin_theta);
  KOKKOS_INLINE_FUNCTION
  static Real LogHAux(struct pgen_param pgen, Real r, Real sin_theta);
  KOKKOS_INLINE_FUNCTION
  static Real CalculateCovariantUT(struct pgen_param pgen, Real r, Real sin_theta, Real l);
  KOKKOS_INLINE_FUNCTION
  static void CalculateVelocityInTorus(struct pgen_param pgen, Real r, Real sin_theta, Real *pu0, Real *pu3);
  KOKKOS_INLINE_FUNCTION
  static void TransformVector(Real a0_sks, Real a1_sks, Real a2_sks, Real a3_sks, Real x1, Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3);


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
    Real h_surf;        // minimum required scale height

    // magnetosphere parameters
    Real r_core;        // critial radius for interior dipole fields (must be > 2rg)
    Real mu_dipole;     // dipole moment
    Real c1_dipole;     // coefficient c1 for smooth interior dipole fields
    Real c2_dipole;     // coefficient c2 for smooth interior dipole fields

    // gas parameters
    Real gamma_adi;
    Real sigma_max;
    Real beta_min;
    Real dfloor;
    Real pfloor;

    // disk paramters
    bool add_torus;                            // enable torus initialization
    bool thin_torus;                           // use Chakrabarti (thin) or Fishbone-Moncrief (thick)
    Real r_edge, r_peak, rho_max;              // fixed torus parameters
    Real l_peak;                               // specific ang. mom. at (r_peak, theta=pi/2)
    Real c_param;                              // l = c * lambda^n constant (from CalculateCN)
    Real n_param;                              // l = c * lambda^n slope; 0 => fit from edge/peak
    Real log_h_edge, log_h_peak;               // calculated torus parameters
    Real ptot_over_rho_peak, rho_peak;         // more calculated torus parameters
    Real r_outer_edge;                         // outermost equatorial radius where log_h >= 0
    Real rho_min, rho_pow, pgas_min, pgas_pow; // background parameters
    Real pert_amp;                             // pressure perturbation amplitude (seeds MRI)

    // testing paramters
    bool test_hydro_balance;

    // fixes
    bool apply_ns_mask;
    bool apply_atm_damper;
    bool smooth_atm_top;
    bool apply_floor_cooling;
    Real cool_floor_factor;   // typical 100; tunable

  };

  pgen_param pp;

} // end of namespace

// Prototypes for user-defined source functions
void MySourceTerms(Mesh* pm, const Real bdt);
void MyMaskTerms(Mesh* pm, const Real bdt);
void GravSrcTerm(Mesh* pm, const Real bdt);
void NeutronStarMask(Mesh* pm, const Real bdt);
void SurfaceDamper(Mesh* pm, const Real bdt);
void FloorCooling(Mesh* pm, const Real bdt);

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
  if ((!pmbp->pcoord->is_general_relativistic) && (!pmbp->pcoord->is_special_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "magnetospheric_accretion problem can only be run with relativity defined in <coord> block"
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
  user_mask_func = MyMaskTerms;

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

  // mask and excision
  pp.r_mask        = pin->GetOrAddReal("problem", "r_mask",     0.8*pp.R_star);

  // unit parameters
  pp.rg            = pin->GetOrAddReal("units", "rg",          1.0);
  pp.rho_unit      = pin->GetOrAddReal("units", "density_cgs", 0.01);
  pp.mu            = pin->GetOrAddReal("units", "mu",          0.5991611743559018);

  // atmosphere parameters
  pp.rho_surf      = pin->GetOrAddReal("problem", "rho_surf",      25859.690449505222);
  pp.tgas_surf     = pin->GetOrAddReal("problem", "tgas_surf",     1.5328527189503498e-07);
  pp.r_surf        = pin->GetOrAddReal("problem", "r_surf",        0.8*pp.R_star);
  pp.rmax_atm_pole = pin->GetOrAddReal("problem", "rmax_atm_pole", 1.1*pp.R_star);
  pp.h_surf        = pin->GetOrAddReal("problem", "h_surf",        5.0*0.03125);

  // magnetosphere parameters
  pp.r_core        = pin->GetOrAddReal("problem", "r_core", 0.5*pp.R_star);

  // gas parameters
  pp.gamma_adi     = pin->GetOrAddReal("mhd", "gamma_adi", 5./3);
  pp.dfloor        = pin->GetOrAddReal("mhd", "dfloor",    1.0e-08);
  pp.pfloor        = pin->GetOrAddReal("mhd", "pfloor",    1.0e-15);
  pp.beta_min      = pin->GetOrAddReal("mhd", "beta_min",  1.0e-05);
  pp.sigma_max     = pin->GetOrAddReal("mhd", "sigma_max", 1000.0);

  // testing parameters
  pp.test_hydro_balance = pin->GetOrAddBoolean("problem", "test_hydro_balance", false);

  // fixes
  pp.apply_ns_mask       = pin->GetOrAddBoolean("problem", "apply_ns_mask", true);
  pp.apply_atm_damper    = pin->GetOrAddBoolean("problem", "apply_atm_damper", true);
  pp.smooth_atm_top      = pin->GetOrAddBoolean("problem", "smooth_atm_top", false);
  pp.apply_floor_cooling = pin->GetOrAddBoolean("problem", "apply_floor_cooling", true);
  pp.cool_floor_factor   = pin->GetOrAddReal(   "problem", "cool_floor_factor", 100.0);

  // disk paramters
  pp.add_torus = pin->GetOrAddBoolean("problem", "add_torus", false);
  if (pp.add_torus) {
    // torus
    pp.thin_torus  = pin->GetOrAddBoolean("problem", "thin_torus", true);
    pp.r_edge      = pin->GetReal("problem", "r_edge");
    pp.r_peak      = pin->GetReal("problem", "r_peak");
    pp.rho_max     = pin->GetReal("problem", "rho_max");
    pp.n_param     = pin->GetOrAddReal("problem", "n_param",0.0);
    pp.rho_min     = pin->GetReal("problem", "rho_min");
    pp.rho_pow     = pin->GetReal("problem", "rho_pow");
    pp.pgas_min    = pin->GetReal("problem", "pgas_min");
    pp.pgas_pow    = pin->GetReal("problem", "pgas_pow");
    pp.pert_amp    = pin->GetOrAddReal("problem", "pert_amp", 0.0);
    // magnetic field parameters
    // TODO: add reading B-field parameters
  }

  // 5. COMPUTE MAGNETIC FIELD PARAMETERS
  //    - Compute magnetic moment mu from B* and R* (Eq. 2)
  //    - Evaluate A(R_m) and A'(R_m) from exterior potential (Eq. 6)
  //    - Compute mask coefficients C1, C2 at R_m (Eq. 5)
  {
    const Real &R_star=pp.R_star, &r_core=pp.r_core;
    Real B_star = pp.B_star;
    if (pp.test_hydro_balance) B_star *= 1.0e-12; // weak magnetic field for hydro balance test
    pp.mu_dipole =  0.5*B_star*R_star*R_star*R_star;
    pp.c1_dipole =  2.5*pp.mu_dipole/(r_core*r_core*r_core);
    pp.c2_dipole = -1.5*pp.mu_dipole/(r_core*r_core*r_core*r_core*r_core);
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
  int &nmhd_ = pmbp->pmhd->nmhd;
  DvceArray5D<Real> u0_ = pmbp->pmhd->u0;
  DvceArray5D<Real> w0_ = pmbp->pmhd->w0;

  // Get ideal gas EOS data
  Real gm1 = pp.gamma_adi - 1.0;


  // Compute paramters for torus initialization
  if (pp.add_torus) {
    if (pp.thin_torus) { // Chakrabarti torus
      CalculateCN(pp, &pp.c_param, &pp.n_param);
      pp.l_peak = CalculateL(pp, pp.r_peak, 1.0);
    } else { // FM torus
      Real r_peak = pp.r_peak;
      pp.l_peak = r_peak*sqrt(r_peak) / (r_peak-3.0);
    } // endelse
    // common to both Chakrabarti and FM tori
    pp.log_h_edge = LogHAux(pp, pp.r_edge, 1.0);
    pp.log_h_peak = LogHAux(pp, pp.r_peak, 1.0) - pp.log_h_edge;
    pp.ptot_over_rho_peak = gm1/pp.gamma_adi * (exp(pp.log_h_peak)-1.0);
    pp.rho_peak = pow(pp.ptot_over_rho_peak, 1.0/gm1) / pp.rho_max;
    FindTorusOuterEdge(pp, &pp.r_outer_edge);
  } // endif (pp.add_torus)


  // initialize primitive variables for new run ---------------------------------------
  auto pp_dvce = pp;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;
  Kokkos::parallel_reduce("pgen_star_ini", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

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
    const Real &R_star = pp_dvce.R_star, &B_star = pp_dvce.B_star, &Omg_star = pp_dvce.Omega_star;
    const Real &r_surf = pp_dvce.r_surf, &rho_surf = pp_dvce.rho_surf, &tgas_surf = pp_dvce.tgas_surf;
    const Real &rmax_atm_pole = pp_dvce.rmax_atm_pole, &h_surf = pp_dvce.h_surf;

    const Real &beta_min  = pp_dvce.beta_min;
    const Real &sigma_max = pp_dvce.sigma_max;

    // compute local floors
    Real emag_star = SQR(B_star*R_star*R_star*R_star);
    Real dfloor = fmax(pp_dvce.dfloor, emag_star/SQR(rv*rv*rv)/sigma_max);
    Real pfloor = fmax(pp_dvce.pfloor, emag_star/SQR(rv*rv*rv)/2*beta_min);

    // initialize gas profiles
    Real dens=dfloor, pgas=pfloor;
    Real uu1=0, uu2=0.0, uu3=0.0;
    Real dens_tracer=0.0;

    // torus (r > rmax_atm_eqtr)
    // TODO: add radiation in initialization !!!!!
    if (pp_dvce.add_torus) {
      // determine if we are in the torus
      Real log_h;
      bool in_torus = false;
      if ((rv >= pp_dvce.r_edge) && (rv <= pp_dvce.r_outer_edge)) {
        log_h = LogHAux(pp_dvce, rv, sin(thv)) - pp_dvce.log_h_edge;
        if (log_h >= 0.0) in_torus = true;
      }

      // initialize with background primitives
      Real rho_bg, pgas_bg;
      rho_bg  = pp_dvce.rho_min  * pow(rv, pp_dvce.rho_pow);
      pgas_bg = pp_dvce.pgas_min * pow(rv, pp_dvce.pgas_pow);
      dens = fmax(rho_bg,  dfloor);
      pgas = fmax(pgas_bg, pfloor);

      Real perturbation = 0.0;
      // Overwrite primitives inside torus
      if (in_torus) {
        // Calculate perturbation
        auto rand_gen = rand_pool64.get_state(); // get random number state this thread
        perturbation = 2.0*pp_dvce.pert_amp*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);        // free state for use by other threads

        // Calculate thermodynamic variables
        Real ptot_over_rho = gm1/(gm1+1) * (exp(log_h) - 1.0);
        dens = pow(ptot_over_rho, 1.0/gm1) / pp_dvce.rho_peak;
        pgas = ptot_over_rho * dens;
        dens = fmax(dens, fmax(rho_bg,  dfloor));
        pgas = fmax(pgas, fmax(pgas_bg, pfloor)) * (1.0+perturbation);

        // Calculate velocities in Boyer-Lindquist coordinates
        Real u0_sks, u1_sks=0, u2_sks=0, u3_sks;
        CalculateVelocityInTorus(pp_dvce, rv, sin(thv), &u0_sks, &u3_sks);
        Real u0, u1, u2, u3;
        TransformVector(u0_sks, u1_sks, u2_sks, u3_sks, x1v, x2v, x3v, &u0, &u1, &u2, &u3);

        // record maximum pressure
        max_ptot = fmax(pgas, max_ptot);

        // convert torus solution from GR to SR
        uu1 = u1;
        uu2 = u2;
        uu3 = u3;
      } // endif (in_torus)
      dens_tracer = fmax(dens, 0.0);
    } // endif torus

    // only trace torus density
    if (is_radiation_enabled) {
      w0_(m,nmhd_+0,k,j,i) = dens_tracer;
    }

    // below the atmosphere (r < R_star)
    if ((rv >= r_surf) && (rv <= rmax_atm_pole)) {
      // analytical isothermal atmosphere
      dens = rho_surf * exp(-(rv-r_surf)/h_surf);
      pgas = dens * tgas_surf;

      if (pp_dvce.smooth_atm_top) {
        const Real r_taper_in = 0.95 * rmax_atm_pole;
        if (rv > r_taper_in) {
          // endpoint values at r_taper_in
          Real rho_at_taper_in = rho_surf * exp(-(r_taper_in - r_surf) / h_surf);
          Real pgas_at_taper_in = rho_at_taper_in * tgas_surf;

          // smoothstep parameter
          Real r_frac = (rv - r_taper_in) / (rmax_atm_pole - r_taper_in);
          Real frac_smth = r_frac * r_frac * (3.0 - 2.0 * r_frac);

          // Independent log-blends for ρ and p
          //   ρ:  rho_at_taper_in --> dfloor   (large jump, this is the smoothing)
          //   p:  pgas_at_taper_in --> pfloor  (small jump by design — preserves continuity)
          Real log_rho = (1.0 - frac_smth) * log(rho_at_taper_in)  + frac_smth * log(dfloor);
          Real log_p   = (1.0 - frac_smth) * log(pgas_at_taper_in) + frac_smth * log(pfloor);
          dens = exp(log_rho);
          pgas = exp(log_p);
        }
      } // endif smooth atmosphere top

    } // end atmosphere

    // apply floors
    dens = fmax(dens, dfloor);
    pgas = fmax(pgas, pfloor);

    if (rv < r_surf) {
      dens = rho_surf;
      pgas = rho_surf*tgas_surf;
      // if (rv <= r_surf) {
      //   dens = rho_surf;
      //   pgas = rho_surf*tgas_surf;
      //   Real u0 = 1./sqrt(1. - SQR(Omg_star*rv*sin(thv)));
      //   uu1 = -Omg_star*x2v*u0;
      //   uu2 =  Omg_star*x1v*u0;
      // } else {
      //   dens = rho_surf;
      //   pgas = rho_surf*tgas_surf;
      // }
    }

    // set primitive values
    w0_(m,IDN,k,j,i) = dens;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

  }, Kokkos::Max<Real>(ptotmax)); // end "pgen_star_ini"


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
      // TODO: add disk field later
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
      // TODO: add disk field later
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
      // TODO: add disk field later
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




#if MPI_PARALLEL_ENABLED
    // get maximum value of gas pressure and bsq over all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE, &bsqmax_intorus, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif



  // 9. CONVERT PRIMITIVES TO CONSERVED VARIABLES
  //    - Call pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, ...)
  pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);

  return;
} // end ProblemGenerator::UserProblem



namespace {
//----------------------------------------------------------------------------------------

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

// compute dipole magnetic field
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
  Real aphi_rcomp = (r > r_core) ? mu_dipole/r : c1_dipole*r2 + c2_dipole*SQR(r2);
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
  Real aphi_rcomp = (r > r_core) ? mu_dipole/r : c1_dipole*r2 + c2_dipole*SQR(r2);
  Real aphi = SQR(sin(theta)) * aphi_rcomp;
  return aphi * (x1 / (SQR(x1)+SQR(x2)));
}

KOKKOS_INLINE_FUNCTION
Real A3dipole(struct pgen_param pp, Real x1, Real x2, Real x3) {
  return 0.0;
}





//----------------------------------------------------------------------------------------
// Function for finding outer edge of torus

KOKKOS_INLINE_FUNCTION
static void FindTorusOuterEdge(struct pgen_param pgen, Real *r_outer_edge) {
  // find "outer edge" of torus (first place log_h > 0)
  Real ra = pgen.r_peak;
  Real rb = 2. * ra;
  Real log_h_trial = LogHAux(pgen, rb, 1.) - pgen.log_h_edge;
  for (int iter=0; iter<10000; ++iter) {
    if (log_h_trial <= 0) break;
    rb *= 2.;
    log_h_trial = LogHAux(pgen, rb, 1.) - pgen.log_h_edge;
  } // endfor iter
  for (int iter=0; iter<10000; ++iter) {
    if (fabs(ra - rb) < 1.e-3) break;
    Real r_trial = (ra + rb) / 2.;
    if (LogHAux(pgen, r_trial, 1.) > pgen.log_h_edge) {
      ra = r_trial;
    } else {
      rb = r_trial;
    }
  } // endfor iter
  *r_outer_edge = ra;
  std::cout << "Found torus outer edge: " << ra << std::endl;
  return;
} // end FindTorusOuterEdge


//----------------------------------------------------------------------------------------
// Function for calculating c, n parameters controlling angular momentum profile
// in Chakrabarti torus, where l = c * lambda^n. edited so that n can be pre-specified
// such that the assumption of keplerian angular momentum at the inner edge is dropped

KOKKOS_INLINE_FUNCTION
static void CalculateCN(struct pgen_param pgen, Real *cparam, Real *nparam) {
  Real n_input = pgen.n_param;
  Real nn;  // slope of angular momentum profile
  Real cc;  // constant of angular momentum profile

  // Keplerian l at edge and peak
  Real l_edge = pgen.r_edge * sqrt(pgen.r_edge) / (pgen.r_edge - 2.0);
  Real l_peak = pgen.r_peak * sqrt(pgen.r_peak) / (pgen.r_peak - 2.0);

  // von Zeipel lambda at edge and peak
  Real lambda_edge = sqrt(SQR(pgen.r_edge) * pgen.r_edge / (pgen.r_edge - 2.0));
  Real lambda_peak = sqrt(SQR(pgen.r_peak) * pgen.r_peak / (pgen.r_peak - 2.0));

  if (n_input == 0.0) {
    // fit n so l matches Keplerian at both anchors
    nn = log(l_peak/l_edge) / log(lambda_peak/lambda_edge);
    cc = l_edge * pow(lambda_edge, -nn);
  } else {
    // prescribed n; fit c from the peak only
    nn = n_input;
    cc = l_peak * pow(lambda_peak, -nn);
  }
  *cparam = cc;
  *nparam = nn;
  return;
} // end CalculateCN

//----------------------------------------------------------------------------------------
// Function for calculating l in Chakrabarti torus
KOKKOS_INLINE_FUNCTION
static Real CalculateL(struct pgen_param pgen, Real r, Real sin_theta) {
  Real lambda_sq = SQR(r) * r * SQR(sin_theta) / (r - 2.0);
  return pgen.c_param * pow(lambda_sq, 0.5*pgen.n_param);
} // end CalculateL

//----------------------------------------------------------------------------------------
// Function to calculate enthalpy in Chakrabarti or Fishbone-Moncrief torus torus
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)
//   references Chakrabarti, S. 1985, ApJ 288, 1

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct pgen_param pgen, Real r, Real sin_theta) {
  Real logh;
  if (pgen.thin_torus) { // Chakrabarti torus
    Real l = CalculateL(pgen, r, sin_theta);
    Real u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
    Real l_edge = CalculateL(pgen, pgen.r_edge, 1.0);
    Real u_t_edge = CalculateCovariantUT(pgen, pgen.r_edge, 1.0, l_edge);
    Real h = u_t_edge/u_t;
    if (pgen.n_param==1.0) {
      h *= pow(l_edge/l, SQR(pgen.c_param)/(SQR(pgen.c_param)-1.0));
    } else {
      Real pow_c = 2.0/pgen.n_param;
      Real pow_l = 2.0-2.0/pgen.n_param;
      Real pow_abs = pgen.n_param/(2.0-2.0*pgen.n_param);
      h *= (pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l,      pow_l)), pow_abs) *
            pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l_edge, pow_l)), -1.0*pow_abs));
    }
    if (isfinite(h) && h >= 1.0) {
      logh = log(h);
    } else if (fabs(h-1.0) <= 1e-15) {
      // prevent confusion from the truncation error
      logh = 0.0;
    } else {
      logh = -1.0;
    }
  } else { // FM torus
    Real exp_2nu     = 1.0 - 2.0/r;
    Real exp_neg2chi = exp_2nu / SQR(r*sin_theta);
    Real var_a       = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
    logh = 0.5*log((1.0 + var_a)/exp_2nu) - 0.5*var_a;
  } // endelse

  return logh;
}

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct pgen_param pgen, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real g_00 = -1.0 + 2.0/r;
  Real g_33 = SQR(r*sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(fmax(-g_00*g_33/(g_33+SQR(l)*g_00), 0.0));
  return u_t;
} // end CalculateCovariantUT

//----------------------------------------------------------------------------------------
// Function to calculate BL 4-velocity (u^t, u^phi) on a circular equatorial orbit
// inside a Chakrabarti or Fishbone-Moncrief torus, specialized to Schwarzschild
// (spin = 0).  With a = 0 the BL metric is diagonal (g_{t phi} = 0), so the
// orbit equations collapse to:
//
//   Chakrabarti:
//     l     = CalculateL(r, sin_theta),  u_t = CalculateCovariantUT(r, sin_theta, l)
//     u^t   = -u_t / (1 - 2/r)
//     u^phi = -l u_t / (r^2 sin^2 theta)
//
//   Fishbone-Moncrief:
//     u_phi_proj = sqrt(0.5*(-1 + sqrt(1 + 4 l_peak^2 exp_neg2chi)))  [sign by prograde]
//     u^phi = u_phi_proj / (r sin_theta)
//     u^t   = sqrt((1 + u_phi_proj^2) / (1 - 2/r))

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct pgen_param pgen, Real r, Real sin_theta, Real *pu0, Real *pu3) {
  Real g_00 = -(1.0 - 2.0/r);
  Real g_33 = SQR(r) * SQR(sin_theta);
  Real u0 = 0.0, u3 = 0.0;
  if (pgen.thin_torus) { // Chakrabarti torus
    Real l   = CalculateL(pgen, r, sin_theta);
    Real u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
    u0 = u_t / g_00;          // u^t   = g^{tt} u_t,  g^{tt} = 1/g_tt when g_{t phi}=0
    u3 = -l * u_t / g_33;     // u^phi = -l u_t / g_{phi phi}
  } else { // FM torus
    Real exp_neg2chi = -g_00 / g_33;         // = (1 - 2/r) / (r^2 sin^2 theta)
    Real var_a       = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
    Real u_phi_proj  = sqrt(0.5*(var_a - 1.0));

    u3 = u_phi_proj / (r * sin_theta);
    u0 = sqrt((1.0 + SQR(u_phi_proj)) / (-g_00));
  } // endelse

  *pu0 = u0;
  *pu3 = u3;
  return;
}


//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from SKS to CKS
KOKKOS_INLINE_FUNCTION
static void TransformVector(Real a0_sks, Real a1_sks, Real a2_sks, Real a3_sks,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real s = sqrt(SQR(x1) + SQR(x2));
  *pa0 = a0_sks + 2.0/(r - 2.0) * a1_sks;
  *pa1 = (x1/r) * a1_sks + (x1*x3/s) * a2_sks - x2 * a3_sks;
  *pa2 = (x2/r) * a1_sks + (x2*x3/s) * a2_sks + x1 * a3_sks;
  *pa3 = (x3/r) * a1_sks -  s        * a2_sks;
  return;
}


} // end of namespace


void MySourceTerms(Mesh* pm, const Real bdt) {
  GravSrcTerm(pm, bdt);
  return;
}


void MyMaskTerms(Mesh* pm, const Real bdt) {
  if (pp.apply_ns_mask) {
    NeutronStarMask(pm, bdt);
  }

  if (pp.apply_atm_damper) {
    SurfaceDamper(pm, bdt);
  }

  if (pp.apply_floor_cooling) {
    FloorCooling(pm, bdt);
  }

  return;
}


void GravSrcTerm(Mesh* pm, const Real bdt) {
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
    if (rv > pp_dvce.rmax_atm_pole) {
      Real dens = w0_(m,IDN,k,j,i);
      Real pgas = w0_(m,IEN,k,j,i)*gm1;
      Real uu1  = w0_(m,IVX,k,j,i);
      Real uu2  = w0_(m,IVY,k,j,i);
      Real uu3  = w0_(m,IVZ,k,j,i);
      Real uu0  = sqrt(1.0+SQR(uu1)+SQR(uu2)+SQR(uu3));

      Real wg = dens + pgas/gm1 + pgas;
      Real rho_grav = uu0*uu0*wg;
      Real rv3 = rv*rv*rv;

      Real src_mx = -rho_grav * x1v/rv3;
      Real src_my = -rho_grav * x2v/rv3;
      Real src_mz = -rho_grav * x3v/rv3;
      Real src_e  = -wg * (x1v*uu1 + x2v*uu2 + x3v*uu3)/(uu0*rv3);

      u0_(m,IM1,k,j,i) += bdt * src_mx;
      u0_(m,IM2,k,j,i) += bdt * src_my;
      u0_(m,IM3,k,j,i) += bdt * src_mz;
      u0_(m,IEN,k,j,i) += bdt * src_e;
    }

    if ((rv >= pp_dvce.r_surf) && (rv <= pp_dvce.rmax_atm_pole)) {
      Real dens = w0_(m,IDN,k,j,i);
      Real pgas = w0_(m,IEN,k,j,i)*gm1;
      Real uu1  = w0_(m,IVX,k,j,i);
      Real uu2  = w0_(m,IVY,k,j,i);
      Real uu3  = w0_(m,IVZ,k,j,i);
      Real uu0  = sqrt(1.0+SQR(uu1)+SQR(uu2)+SQR(uu3));

      Real src_mx = -uu0*uu0 * pgas/pp_dvce.h_surf * x1v/rv;
      Real src_my = -uu0*uu0 * pgas/pp_dvce.h_surf * x2v/rv;
      Real src_mz = -uu0*uu0 * pgas/pp_dvce.h_surf * x3v/rv;
      Real src_e  = -pgas/pp_dvce.h_surf * (x1v*uu1 + x2v*uu2 + x3v*uu3)/(uu0*rv);

      u0_(m,IM1,k,j,i) += bdt * src_mx;
      u0_(m,IM2,k,j,i) += bdt * src_my;
      u0_(m,IM3,k,j,i) += bdt * src_mz;
      u0_(m,IEN,k,j,i) += bdt * src_e;
    } // endelse

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

  int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;

  // printf("is=%d, ie=%d, js=%d, je=%d, ks=%d, ke=%d \n", is,ie,js,je,ks,ke);
  // printf("n1=%d, n2=%d, n3=%d, ng=%d \n", indcs.nx1, indcs.nx2, indcs.nx3, indcs.ng);

  bool &entropy_fix_ = pmbp->pmhd->entropy_fix;
  int &nmhd  = pmbp->pmhd->nmhd;
  int &nscal = pmbp->pmhd->nscalars;
  int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;

  // MHD variables
  DvceArray5D<Real> u0_, w0_, bcc0_;
  u0_   = pmbp->pmhd->u0;
  w0_   = pmbp->pmhd->w0;
  bcc0_ = pmbp->pmhd->bcc0;

  // capture problem parameters
  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;
  // par_for("pgen_nsmask", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  par_for("pgen_nsmask", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
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

    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);

    // extract problem parameters
    const Real &R_star=pp_dvce.R_star, &Omg_star=pp_dvce.Omega_star;
    const Real &rho_surf=pp_dvce.rho_surf, &tgas_surf=pp_dvce.tgas_surf;

    // extract problem paramters
    if (rv < pp_dvce.r_surf) {
      // no modification in b-field
      Real dens = rho_surf;
      Real pgas = dens*tgas_surf;
      Real bx = bcc0_(m,IBX,k,j,i);
      Real by = bcc0_(m,IBY,k,j,i);
      Real bz = bcc0_(m,IBZ,k,j,i);

      // Real u0 = 1./sqrt(1. - SQR(Omg_star*rv*sin(thv)));
      // Real u1 = -Omg_star*x2v*u0;
      // Real u2 =  Omg_star*x1v*u0;
      // Real u3 = 0.0;

      Real u0 = 1.0;
      Real u1 = 0.0;
      Real u2 = 0.0;
      Real u3 = 0.0;

      // assign primitives first
      w0_(m,IDN,k,j,i) = dens;
      w0_(m,IEN,k,j,i) = pgas/gm1;
      w0_(m,IVX,k,j,i) = u1;
      w0_(m,IVY,k,j,i) = u2;
      w0_(m,IVZ,k,j,i) = u3;
      if (entropy_fix_) w0_(m,entropyIdx,k,j,i) = pgas/pow(dens,gm1)/dens;

      // compute and assign conservatives
      Real b0 = u1*bx + u2*by + u3*bz;
      Real b1 = (bx + b0*u1) / u0;
      Real b2 = (by + b0*u2) / u0;
      Real b3 = (bz + b0*u3) / u0;
      Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);
      Real wtot = dens + (gm1+1)/gm1*pgas + b_sq;
      Real ptot = pgas + 0.5*b_sq;
      u0_(m,IDN,k,j,i) = dens*u0;
      u0_(m,IEN,k,j,i) = wtot*u0*u0 - b0*b0 - ptot - dens*u0;
      u0_(m,IM1,k,j,i) = wtot*u0*u1 - b0*b1;
      u0_(m,IM2,k,j,i) = wtot*u0*u2 - b0*b2;
      u0_(m,IM3,k,j,i) = wtot*u0*u3 - b0*b3;
      if (entropy_fix_) u0_(m,entropyIdx,k,j,i) = pgas/pow(dens,gm1)*u0;
    } // endif (rv < pp_dvce.r_surf)

  }); // end par_for
} // end NeutronStarMask


void SurfaceDamper(Mesh* pm, const Real bdt) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;

  bool &entropy_fix_ = pmbp->pmhd->entropy_fix;
  int &nmhd  = pmbp->pmhd->nmhd;
  int &nscal = pmbp->pmhd->nscalars;
  int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;

  // MHD variables
  DvceArray5D<Real> u0_, w0_, bcc0_;
  u0_   = pmbp->pmhd->u0;
  w0_   = pmbp->pmhd->w0;
  bcc0_ = pmbp->pmhd->bcc0;
  auto &b0 = pmbp->pmhd->b0;

  // capture problem parameters
  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;

  // Buffer geometry: r_buf_in <= r <= r_top.
  // ramp = 0 at r_buf_in (no damping), ramp = 1 at r_top (full damp/reset).
  const Real r_top     = pp_dvce.rmax_atm_pole;
  const Real r_buf_in  = 0.9 * r_top;
  const Real r_buf_out = 1.05 * r_top;
  const Real inv_in_w   = 1.0 / (r_top - r_buf_in);
  const Real inv_out_w  = 1.0 / (r_top - r_buf_out);

  // par_for("damp_v_perp", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  par_for("damp_fluid", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
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

    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);

    if (rv < r_buf_in || rv > r_buf_out) return;

    Real pgas = w0_(m,IEN,k,j,i) * gm1;
    if (rv > r_top) { // r_top < rv <= r_buf_out
      Real t = (rv - r_buf_out) * inv_out_w;
      Real ramp = t*t*(3.0 - 2.0*t);     // 1 at r_top, 0 at r_buf_out

      Real emag_star = SQR(pp_dvce.B_star*pp_dvce.R_star*pp_dvce.R_star*pp_dvce.R_star);
      Real p_tar = fmax(pp_dvce.pfloor, emag_star/SQR(rv*rv*rv)/2*pp_dvce.beta_min);

      w0_(m,IEN,k,j,i) = (p_tar + (1.0 - ramp)*(pgas - p_tar))/gm1;
    } // endif r_top < rv <= r_buf_out

    if (rv <= r_top) { // r_buf_in <= rv <= r_top
      // smoothstep ramp in [0,1]
      Real t = (rv - r_buf_in) * inv_in_w;
      Real ramp = t*t*(3.0 - 2.0*t);

      // target velocity (spatial 4-velocity components). Currently rest frame;
      // promote to a function of position later for rotation / inflow.
      Real dens = w0_(m,IDN,k,j,i);
      Real uu1 = w0_(m,IVX,k,j,i);
      Real uu2 = w0_(m,IVY,k,j,i);
      Real uu3 = w0_(m,IVZ,k,j,i);

      Real ux_tar = 0.0;
      Real uy_tar = 0.0;
      Real uz_tar = 0.0;
      Real p_tar  = dens * pp_dvce.tgas_surf;

      // ramp == 1 -> snap to target this step; ramp == 0 -> leave unchanged.
      w0_(m,IVX,k,j,i) = ux_tar + (1.0 - ramp)*(uu1 - ux_tar);
      w0_(m,IVY,k,j,i) = uy_tar + (1.0 - ramp)*(uu2 - uy_tar);
      w0_(m,IVZ,k,j,i) = uz_tar + (1.0 - ramp)*(uu3 - uz_tar);
      w0_(m,IEN,k,j,i) = (p_tar + (1.0 - ramp)*(pgas - p_tar))/gm1;
    } // endif r_buf_in <= rv <= r_top

  }); // end par_for

  // --------------------------------------------------------------------------
  // Step 2: reset face-centered B to analytical dipole via curl(A_dipole),
  // computing A inline at the four edges of each face.
  // --------------------------------------------------------------------------
  // x1-face: B_x = (A_z(j+1) - A_z(j))/dx2 - (A_y(k+1) - A_y(k))/dx3
  // par_for("damp_b_x1f", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie+1,
  par_for("damp_b_x1f", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1f   = LeftEdgeX  (i  -is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v   = CellCenterX(j  -js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x2f   = LeftEdgeX  (j  -js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x2fp1 = LeftEdgeX  (j+1-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v   = CellCenterX(k  -ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real x3f   = LeftEdgeX  (k  -ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real x3fp1 = LeftEdgeX  (k+1-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rf = sqrt(SQR(x1f) + SQR(x2v) + SQR(x3v));

    if (rf < r_buf_in || rf > r_top) return;

    Real t = (rf - r_buf_in) * inv_in_w;
    Real ramp = t*t*(3.0 - 2.0*t);

    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    Real az_r = A3dipole(pp_dvce, x1f, x2fp1, x3v);
    Real az_l = A3dipole(pp_dvce, x1f, x2f,   x3v);
    Real ay_t = A2dipole(pp_dvce, x1f, x2v, x3fp1);
    Real ay_b = A2dipole(pp_dvce, x1f, x2v, x3f);
    Real bx_tgt = (az_r - az_l)/dx2 - (ay_t - ay_b)/dx3;

    b0.x1f(m,k,j,i) = (1.0 - ramp)*b0.x1f(m,k,j,i) + ramp*bx_tgt;
  });

  // x2-face: B_y = (A_x(k+1) - A_x(k))/dx3 - (A_z(i+1) - A_z(i))/dx1
  // par_for("damp_b_x2f", DevExeSpace(), 0,nmb-1, ks,ke, js,je+1, is,ie,
  par_for("damp_b_x2f", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1+1, 0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v   = CellCenterX(i  -is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1f   = LeftEdgeX  (i  -is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1fp1 = LeftEdgeX  (i+1-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2f   = LeftEdgeX  (j  -js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v   = CellCenterX(k  -ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real x3f   = LeftEdgeX  (k  -ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real x3fp1 = LeftEdgeX  (k+1-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rf = sqrt(SQR(x1v) + SQR(x2f) + SQR(x3v));

    if (rf < r_buf_in || rf > r_top) return;

    Real t = (rf - r_buf_in) * inv_in_w;
    Real ramp = t*t*(3.0 - 2.0*t);

    Real dx1 = size.d_view(m).dx1;
    Real dx3 = size.d_view(m).dx3;

    Real ax_t = A1dipole(pp_dvce, x1v, x2f, x3fp1);
    Real ax_b = A1dipole(pp_dvce, x1v, x2f, x3f);
    Real az_r = A3dipole(pp_dvce, x1fp1, x2f, x3v);
    Real az_l = A3dipole(pp_dvce, x1f,   x2f, x3v);
    Real by_tgt = (ax_t - ax_b)/dx3 - (az_r - az_l)/dx1;

    b0.x2f(m,k,j,i) = (1.0 - ramp)*b0.x2f(m,k,j,i) + ramp*by_tgt;
  });

  // x3-face: B_z = (A_y(i+1) - A_y(i))/dx1 - (A_x(j+1) - A_x(j))/dx2
  // par_for("damp_b_x3f", DevExeSpace(), 0,nmb-1, ks,ke+1, js,je, is,ie,
  par_for("damp_b_x3f", DevExeSpace(), 0,nmb-1, 0,n3m1+1, 0,n2m1, 0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v   = CellCenterX(i  -is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1f   = LeftEdgeX  (i  -is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1fp1 = LeftEdgeX  (i+1-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v   = CellCenterX(j  -js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x2f   = LeftEdgeX  (j  -js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x2fp1 = LeftEdgeX  (j+1-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3f   = LeftEdgeX  (k  -ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rf = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3f));

    if (rf < r_buf_in || rf > r_top) return;

    Real t = (rf - r_buf_in) * inv_in_w;
    Real ramp = t*t*(3.0 - 2.0*t);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;

    Real ay_r = A2dipole(pp_dvce, x1fp1, x2v, x3f);
    Real ay_l = A2dipole(pp_dvce, x1f,   x2v, x3f);
    Real ax_t = A1dipole(pp_dvce, x1v, x2fp1, x3f);
    Real ax_b = A1dipole(pp_dvce, x1v, x2f,   x3f);
    Real bz_tgt = (ay_r - ay_l)/dx1 - (ax_t - ax_b)/dx2;

    b0.x3f(m,k,j,i) = (1.0 - ramp)*b0.x3f(m,k,j,i) + ramp*bz_tgt;
  });

  // --------------------------------------------------------------------------
  // Step 3: rebuild bcc0 in the buffer (cell-centered B = avg of two faces).
  // --------------------------------------------------------------------------
  // par_for("damp_bcc", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
  par_for("damp_bcc", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rv  = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    if (rv < r_buf_in || rv > r_top) return;

    bcc0_(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc0_(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc0_(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // --------------------------------------------------------------------------
  // Step 4: sync conservatives with the new w0 and bcc0.
  // --------------------------------------------------------------------------
  // par_for("sync_cons", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
  par_for("sync_cons", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rv  = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    if (rv < r_buf_in || rv > r_buf_out) return;

    Real dens = w0_(m,IDN,k,j,i);
    Real eint = w0_(m,IEN,k,j,i);
    Real u1 = w0_(m,IVX,k,j,i);
    Real u2 = w0_(m,IVY,k,j,i);
    Real u3 = w0_(m,IVZ,k,j,i);
    Real bcc1 = bcc0_(m,IBX,k,j,i);
    Real bcc2 = bcc0_(m,IBY,k,j,i);
    Real bcc3 = bcc0_(m,IBZ,k,j,i);

    Real u0 = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));

    // Calculate 4-magnetic field
    Real b0 = bcc1*u1 + bcc2*u2 + bcc3*u3;
    Real b1 = (bcc1 + b0 * u1) / u0;
    Real b2 = (bcc2 + b0 * u2) / u0;
    Real b3 = (bcc3 + b0 * u3) / u0;
    Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

    // Set conserved quantities
    Real ud = dens * u0;
    Real wtot_u02 = (dens + (gm1+1) * eint + b_sq) * u0 * u0;
    u0_(m,IDN,k,j,i) = ud;
    u0_(m,IEN,k,j,i) = wtot_u02 - b0 * b0 - (gm1*eint + 0.5*b_sq) - ud;  // In SR, evolve E - D
    u0_(m,IM1,k,j,i) = wtot_u02 * u1 / u0 - b0 * b1;
    u0_(m,IM2,k,j,i) = wtot_u02 * u2 / u0 - b0 * b2;
    u0_(m,IM3,k,j,i) = wtot_u02 * u3 / u0 - b0 * b3;

    // entropy
    if (entropy_fix_) {
      w0_(m,entropyIdx,k,j,i) = (gm1*eint)/pow(dens,gm1)/dens;
      u0_(m,entropyIdx,k,j,i) = (gm1*eint)/pow(dens,gm1)*u0;
    }

  });

} // end SurfaceDamper


void FloorCooling(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;

  bool &entropy_fix_ = pmbp->pmhd->entropy_fix;
  int &nmhd  = pmbp->pmhd->nmhd;
  int &nscal = pmbp->pmhd->nscalars;
  int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;

  // MHD variables
  DvceArray5D<Real> u0_, w0_, bcc0_;
  u0_   = pmbp->pmhd->u0;
  w0_   = pmbp->pmhd->w0;
  bcc0_ = pmbp->pmhd->bcc0;

  // capture problem parameters
  Real gm1 = pp.gamma_adi - 1.0;
  auto pp_dvce = pp;

  // Cooling parameters
  const Real r_top      = pp_dvce.rmax_atm_pole;
  const Real ratio_hi   = pp_dvce.cool_floor_factor;     // no cooling above this ρ/ρ_floor
  const Real ratio_lo   = 0.5 * ratio_hi;                // full cooling below this
  const Real inv_ratio_w = 1.0 / (ratio_hi - ratio_lo);

  // par_for("floor_cool", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
  par_for("floor_cool", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
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

    Real rv, thv, phv;
    GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);

    if (rv <= r_top) return;

    // -- read current primitive state --
    // w0_(IVX..IVZ) are spatial 4-velocity components u^i (Athena convention)
    Real dens = w0_(m,IDN,k,j,i);
    Real pgas = w0_(m,IEN,k,j,i) * gm1;
    Real uu1  = w0_(m,IVX,k,j,i);
    Real uu2  = w0_(m,IVY,k,j,i);
    Real uu3  = w0_(m,IVZ,k,j,i);

    // Analytical dipole b² (matches what the runtime σ_max ceiling sees in ns_mask)
    Real emag_star = SQR(pp_dvce.B_star*pp_dvce.R_star*pp_dvce.R_star*pp_dvce.R_star);
    Real dfloor = fmax(pp_dvce.dfloor, emag_star/SQR(rv*rv*rv)/pp_dvce.sigma_max);
    Real pfloor = fmax(pp_dvce.pfloor, emag_star/SQR(rv*rv*rv)/2*pp_dvce.beta_min);

    // ρ-based discriminator: floor-like cells only
    Real rho_ratio = dens / dfloor;
    if (rho_ratio >= ratio_hi) return;   // real gas, leave alone

    Real t_smooth = fmin(1.0, fmax(0.0, (ratio_hi - rho_ratio) * inv_ratio_w));
    Real cool_factor = t_smooth*t_smooth*(3.0 - 2.0*t_smooth);

    // Pressire target
    Real p_target = pfloor;
    if (pgas <= p_target) return;

    // Hard reset toward p_target, scaled by smoothstep
    Real p_new = pgas + cool_factor * (p_target - pgas);
    Real dp    = p_new - pgas;

    // -- update primitive --
    w0_(m,IEN,k,j,i) = p_new / gm1;

    // -- increment conserved variables by the cooling delta --
    Real bx = bcc0_(m,IBX,k,j,i);
    Real by = bcc0_(m,IBY,k,j,i);
    Real bz = bcc0_(m,IBZ,k,j,i);
    Real u0_lor = sqrt(1.0 + uu1*uu1 + uu2*uu2 + uu3*uu3);
    Real b0_4   = uu1*bx + uu2*by + uu3*bz;
    Real b1_4   = (bx + b0_4*uu1) / u0_lor;
    Real b2_4   = (by + b0_4*uu2) / u0_lor;
    Real b3_4   = (bz + b0_4*uu3) / u0_lor;

    Real dwtot = (gm1+1.0)/gm1 * dp;
    u0_(m,IEN,k,j,i) += dwtot * u0_lor*u0_lor - dp;
    u0_(m,IM1,k,j,i) += dwtot * u0_lor * uu1;
    u0_(m,IM2,k,j,i) += dwtot * u0_lor * uu2;
    u0_(m,IM3,k,j,i) += dwtot * u0_lor * uu3;

    if (entropy_fix_) {
      w0_(m,entropyIdx,k,j,i) = p_new/pow(dens,gm1)/dens;
      u0_(m,entropyIdx,k,j,i) = p_new/pow(dens,gm1)*u0_lor;
    }

  });

  return;
} // end FloorCooling


// void FloorCooling(Mesh* pm, const Real bdt) {
//   MeshBlockPack *pmbp = pm->pmb_pack;
//   auto &indcs = pm->mb_indcs;
//   int is = indcs.is, js = indcs.js, ks = indcs.ks;
//   int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
//   int nmb = pmbp->nmb_thispack;
//   auto &size = pmbp->pmb->mb_size;
//
//   int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
//   int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
//   int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;
//
//   bool &entropy_fix_ = pmbp->pmhd->entropy_fix;
//   int &nmhd  = pmbp->pmhd->nmhd;
//   int &nscal = pmbp->pmhd->nscalars;
//   int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;
//
//   // MHD variables
//   DvceArray5D<Real> u0_, w0_, bcc0_;
//   u0_   = pmbp->pmhd->u0;
//   w0_   = pmbp->pmhd->w0;
//   bcc0_ = pmbp->pmhd->bcc0;
//
//   // capture problem parameters
//   Real gm1 = pp.gamma_adi - 1.0;
//   auto pp_dvce = pp;
//
//   // Cooling parameters
//   const Real r_top      = pp_dvce.rmax_atm_pole;
//   const Real T_iso      = 0.5 * pp_dvce.beta_min * pp_dvce.sigma_max;  // isothermal target
//   const Real ratio_hi   = pp_dvce.cool_floor_factor;     // no cooling above this ρ/ρ_floor
//   const Real ratio_lo   = 0.5 * ratio_hi;                // full cooling below this
//   const Real inv_ratio_w = 1.0 / (ratio_hi - ratio_lo);
//
//   // par_for("floor_cool", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
//   par_for("floor_cool", DevExeSpace(), 0,nmb-1, 0,n3m1, 0,n2m1, 0,n1m1,
//   KOKKOS_LAMBDA(int m, int k, int j, int i) {
//     Real &x1min = size.d_view(m).x1min;
//     Real &x1max = size.d_view(m).x1max;
//     Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
//
//     Real &x2min = size.d_view(m).x2min;
//     Real &x2max = size.d_view(m).x2max;
//     Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
//
//     Real &x3min = size.d_view(m).x3min;
//     Real &x3max = size.d_view(m).x3max;
//     Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
//
//     Real rv, thv, phv;
//     GetSchwarzschildCoordinates(x1v, x2v, x3v, &rv, &thv, &phv);
//
//     if (rv <= r_top) return;
//
//     // -- read current primitive state --
//     // w0_(IVX..IVZ) are spatial 4-velocity components u^i (Athena convention)
//     Real dens = w0_(m,IDN,k,j,i);
//     Real pgas = w0_(m,IEN,k,j,i) * gm1;
//     Real uu1  = w0_(m,IVX,k,j,i);
//     Real uu2  = w0_(m,IVY,k,j,i);
//     Real uu3  = w0_(m,IVZ,k,j,i);
//
//     // Analytical dipole b² (matches what the runtime σ_max ceiling sees in ns_mask)
//     Real cos2_theta = SQR(x3v) / (rv*rv);
//     Real bsq_fake   = SQR(pp_dvce.mu_dipole) * (3.0*cos2_theta + 1.0) / (rv*rv*rv*rv*rv*rv);
//     Real rho_floor  = bsq_fake / pp_dvce.sigma_max;
//
//     // ρ-based discriminator: floor-like cells only
//     Real rho_ratio = dens / fmax(rho_floor, pp_dvce.dfloor);
//     if (rho_ratio >= ratio_hi) return;   // real gas, leave alone
//
//     Real t_smooth    = fmin(1.0, fmax(0.0, (ratio_hi - rho_ratio) * inv_ratio_w));
//     Real cool_factor = t_smooth*t_smooth*(3.0 - 2.0*t_smooth);
//
//     // Isothermal target: p = ρ × T_iso (Option A)
//     Real p_target = dens * T_iso;
//     if (pgas <= p_target) return;
//
//     // Hard reset toward p_target, scaled by smoothstep
//     Real p_new = pgas + cool_factor * (p_target - pgas);
//     Real dp    = p_new - pgas;
//
//     // -- update primitive --
//     w0_(m,IEN,k,j,i) = p_new / gm1;
//
//     // -- increment conserved variables by the cooling delta --
//     Real bx = bcc0_(m,IBX,k,j,i);
//     Real by = bcc0_(m,IBY,k,j,i);
//     Real bz = bcc0_(m,IBZ,k,j,i);
//     Real u0_lor = sqrt(1.0 + uu1*uu1 + uu2*uu2 + uu3*uu3);
//     Real b0_4   = uu1*bx + uu2*by + uu3*bz;
//     Real b1_4   = (bx + b0_4*uu1) / u0_lor;
//     Real b2_4   = (by + b0_4*uu2) / u0_lor;
//     Real b3_4   = (bz + b0_4*uu3) / u0_lor;
//
//     Real dwtot = (gm1+1.0)/gm1 * dp;
//     u0_(m,IEN,k,j,i) += dwtot * u0_lor*u0_lor - dp;
//     u0_(m,IM1,k,j,i) += dwtot * u0_lor * uu1;
//     u0_(m,IM2,k,j,i) += dwtot * u0_lor * uu2;
//     u0_(m,IM3,k,j,i) += dwtot * u0_lor * uu3;
//
//     if (entropy_fix_) {
//       w0_(m,entropyIdx,k,j,i) = p_new/pow(dens,gm1)/dens;
//       u0_(m,entropyIdx,k,j,i) = p_new/pow(dens,gm1)*u0_lor;
//     }
//
//   });
//
//   return;
// } // end FloorCooling



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
