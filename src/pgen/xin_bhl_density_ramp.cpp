//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xin_bhl_density_ramp.cpp
//! \brief Restart-compatible density decline for a BHL wind onto a Kerr BH.
//!
//! Based on xin_bhl_xy_rad_corrected.cpp: +x wind, transverse +y magnetic field,
//! fixed temperature and boosted thermal radiation, and spherical flux histories.
//! The -x face injects the reservoir; the other five faces use zero-gradient gas/B
//! outflow and ambient equilibrium for incoming radiation (outgoing rays escape).
//!
//! Defaults: rho_inf starts at problem/rho0 in the ORIGINAL code units, decreases
//! logarithmically to density_ramp_final_cgs=1e-12 g/cm^3 over 30000 M, and holds
//! for 20000 M. With rho0=1 and units/density_cgs=1e-6, this is one decade per 5000 M.
//! Never change the checkpoint unit normalization or rescale its evolved cells.
//!
//! density_ramp_start defaults to the FIRST continuation checkpoint's mesh time
//! and is saved in later checkpoints; do not reset it on subsequent restarts.
//! Fresh runs start at zero. density_ramp_duration and density_ramp_hold set the
//! two intervals. density_ramp_set_tlim=true sets end=start+duration+hold.
//!
//! density_ramp_scale_b=true scales B^y with sqrt(pgas+prad), preserving the
//! original TOTAL-pressure plasma beta. False keeps the initial field amplitude. In both
//! cases wind velocity, temperature and ambient radiation intensity remain fixed.
//! Initial B^y = sqrt(2*rho0*t0/[beta_target*(1-v_wind^2)]), perpendicular to wind.
//! v_wind is physical v/c; the normal-frame velocity is gamma_inf*v_wind.
//!
//! Only boundary ghosts are forced; the checkpoint interior evolves normally.
//! density_ramp_track_floors=true scales active density/pressure/excision floors
//! with the inflow. density_ramp_local_entropy=true uses LOCAL gas density for
//! entropy floors matched to the 1e-9 and 1e-12 reference runs; dense gas retains
//! its original prescription. Saved initial floor anchors survive all restarts.
//! See docs/bhl_density_ramp.md and the partial restart input for the floor law.

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <cmath>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "tasklist/task_list.hpp"
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
static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
Real A1(struct bhl_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct bhl_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct bhl_pgen pgen, Real x1, Real x2, Real x3);

// Parameters for the BHL-wind setup (wind along +x, B along +y)
struct bhl_pgen {
  // Spacetime / excision / EOS
  Real spin;            // black hole spin a/M
  Real dexcise;         // density floor inside excision radius
  Real pexcise;         // pressure floor inside excision radius
  Real gamma_adi;       // ideal-gas adiabatic index
  Real arad;            // radiation constant (only if radiation enabled)

  // Upstream-wind state (uniform far from BH):
  //   rho = rho0,  p = rho0*t0,  uu^i = (uu_wind, 0, 0)
  // where uu_wind = gamma_inf * v_inf is the relativistic normal-frame velocity.
  Real rho0;            // uniform background rest-mass density
  Real t0;              // p/rho ("temperature"); set directly
  Real v_inf;           // physical asymptotic wind speed v_inf/c (input "v_wind")
  Real uu_wind;         // normal-frame primitive velocity uu^x = gamma_inf * v_inf

  // Uniform horizontal magnetic field B = b0 * y_hat (theta_B=90 deg).
  // Coordinate-frame B^y from A_x = b0*x3 (curl gives B^y = b0).
  // Comoving b^2 = b0^2 * (1 - v_inf^2) for B perp v.
  // b0 set from beta_target using the relativistic formula (see file header).
  Real b0;              // INITIAL coordinate-frame horizontal field amplitude
  Real density_unit;    // fixed checkpoint density unit, g/cm^3
  Real rho_final;       // final reservoir density in the original code units
  Real ramp_start;      // absolute mesh time, persisted across later restarts
  Real ramp_duration;
  Real ramp_hold;
  bool scale_b;
  Real prad0;           // fixed comoving ambient radiation pressure, E_rad/3
  Real beta_total0;     // original asymptotic (pgas+prad)/(b^2/2)

  Real InflowDensity(const Real time) const {
    if (time <= ramp_start) return rho0;
    if (time >= ramp_start + ramp_duration) return rho_final;
    const Real fraction = (time - ramp_start)/ramp_duration;
    return rho0*std::exp(fraction*std::log(rho_final/rho0));
  }

  Real InflowField(const Real time) const {
    const Real pressure = InflowDensity(time)*t0 + prad0;
    return scale_b ? b0*std::sqrt(pressure/(rho0*t0 + prad0)) : b0;
  }
};

  bhl_pgen bhl;

// ParameterInput::SetReal/GetOrAddReal in this fork serialize only six
// significant digits. Preserve the schedule's absolute timestamps exactly.
void SetRealPrecise(ParameterInput *pin, const std::string &block,
                    const std::string &name, const Real value) {
  std::ostringstream text;
  text << std::setprecision(std::numeric_limits<Real>::max_digits10) << value;
  pin->SetString(block, name, text.str());
}

struct RampFloors {
  bool track, local_entropy;
  Real dfloor0, pfloor0, dexcise0, pexcise0;
  Real sfloor0, sfloor10, sfloor20, rho10, rho20;
  Real rho_mid, rho_low, s_mid, s_low;
};
RampFloors ramp_floors;

Real ReadFloorAnchor(ParameterInput *pin, const std::string &name, Real value) {
  const Real saved = pin->GetOrAddReal("problem", "density_ramp_"+name, value);
  SetRealPrecise(pin, "problem", "density_ramp_"+name, saved);
  return saved;
}

// Convert the standalone-run entropy coefficient into the unchanged density
// unit. Between reference runs interpolate logarithmically, avoiding jumps.
Real InterpolateEntropy(const Real rho, const Real initial) {
  if (rho >= bhl.rho0) return initial;
  if (rho <= ramp_floors.rho_low) return ramp_floors.s_low;
  Real hi, lo, s_hi, s_lo;
  if (rho >= ramp_floors.rho_mid) {
    hi = bhl.rho0; lo = ramp_floors.rho_mid;
    s_hi = initial; s_lo = ramp_floors.s_mid;
  } else {
    hi = ramp_floors.rho_mid; lo = ramp_floors.rho_low;
    s_hi = ramp_floors.s_mid; s_lo = ramp_floors.s_low;
  }
  const Real fraction = std::log(rho/hi)/std::log(lo/hi);
  return s_hi*std::exp(fraction*std::log(s_lo/s_hi));
}

// Updating ParameterInput alone does not affect already-constructed physics.
// Set the active host EOS/coordinate data that future device kernels capture,
// then serialize the same values for inspection of a later checkpoint header.
void UpdateRampFloors(Mesh *pm, ParameterInput *pin, const Real time) {
  if (!ramp_floors.track) return;
  auto *pack = pm->pmb_pack;
  auto &eos = (pack->pmhd != nullptr) ? pack->pmhd->peos->eos_data
                                     : pack->phydro->peos->eos_data;
  auto &coord = pack->pcoord->coord_data;
  const Real rho = bhl.InflowDensity(time);
  const Real factor = rho/bhl.rho0;
  eos.dfloor = ramp_floors.dfloor0*factor;
  eos.pfloor = ramp_floors.pfloor0*factor;
  coord.dexcise = ramp_floors.dexcise0*factor;
  coord.pexcise = ramp_floors.pexcise0*factor;
  bhl.dexcise = coord.dexcise;
  bhl.pexcise = coord.pexcise;
  eos.bhl_local_entropy = ramp_floors.local_entropy;
  if (ramp_floors.local_entropy) {
    // Dense material retains its original entropy law while rarefied material
    // follows the low-density reference runs, irrespective of advection delay.
    eos.sfloor = ramp_floors.sfloor0;
    eos.sfloor1 = ramp_floors.sfloor10;
    eos.sfloor2 = ramp_floors.sfloor20;
    eos.rho1 = ramp_floors.rho10;
    eos.rho2 = ramp_floors.rho20;
    eos.bhl_rho_start = bhl.rho0;
    eos.bhl_rho_mid = ramp_floors.rho_mid;
    eos.bhl_rho_low = ramp_floors.rho_low;
    eos.bhl_s_mid = ramp_floors.s_mid;
    eos.bhl_s_low = ramp_floors.s_low;
  } else {
    // Optional uniform-in-time prescription. It can heat dense material still
    // in the box as the reference entropy coefficient rises; local is default.
    eos.sfloor = InterpolateEntropy(rho, ramp_floors.sfloor0);
    eos.sfloor1 = InterpolateEntropy(rho, ramp_floors.sfloor10);
    eos.sfloor2 = InterpolateEntropy(rho, ramp_floors.sfloor20);
    eos.rho1 = ramp_floors.rho10*factor;
    eos.rho2 = ramp_floors.rho20*factor;
  }
  const std::string fluid = (pack->pmhd != nullptr) ? "mhd" : "hydro";
  SetRealPrecise(pin, fluid, "dfloor", eos.dfloor);
  SetRealPrecise(pin, fluid, "pfloor", eos.pfloor);
  SetRealPrecise(pin, fluid, "sfloor", eos.sfloor);
  SetRealPrecise(pin, fluid, "sfloor1", eos.sfloor1);
  SetRealPrecise(pin, fluid, "sfloor2", eos.sfloor2);
  SetRealPrecise(pin, fluid, "rho1", eos.rho1);
  SetRealPrecise(pin, fluid, "rho2", eos.rho2);
  SetRealPrecise(pin, "coord", "dexcise", coord.dexcise);
  SetRealPrecise(pin, "coord", "pexcise", coord.pexcise);
}

} // namespace

// Prototypes for user-defined BCs and history functions
void WindBCs(Mesh *pm);
void WindFluxes(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief BHL wind onto a Kerr BH along +x.  Optional uniform horizontal B-field (+y).
//! Compile with '-D PROBLEM=xin_bhl_density_ramp' to enroll as user-specific problem generator.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xin_bhl_xy problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = WindBCs;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // Extract BH parameters
  bhl.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux = (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(bhl.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = WindFluxes;

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
    bhl.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bhl.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bhl.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    bhl.arad = pmbp->prad->arad;
  }

  // Read BHL IC parameters from input file.
  // Inputs interpreted as PHYSICAL quantities (paper / K&M 2025 convention):
  //   rho0   = rho_inf (rest-mass density)
  //   v_wind = v_inf/c (physical wind speed; internal uu^x = gamma_inf * v_inf)
  //   t0     = p/rho
  bhl.rho0  = pin->GetReal("problem", "rho0");
  bhl.t0    = pin->GetReal("problem", "t0");
  bhl.v_inf = pin->GetReal("problem", "v_wind");
  if (bhl.v_inf <= -1.0 || bhl.v_inf >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "v_wind (= v_inf/c) must be in (-1, 1); got " << bhl.v_inf << std::endl;
    exit(EXIT_FAILURE);
  }
  Real gamma_inf = 1.0 / sqrt(1.0 - bhl.v_inf * bhl.v_inf);
  bhl.uu_wind = gamma_inf * bhl.v_inf;     // relativistic normal-frame velocity

  // excision parameters
  bhl.dexcise = coord.dexcise;
  bhl.pexcise = coord.pexcise;
  bhl.prad0 = is_radiation_enabled ? bhl.arad*SQR(SQR(bhl.t0))/3.0 : 0.0;

  // Asymptotic horizontal field amplitude b0 (coordinate-frame B^y) from
  // beta_target.  MUST be computed here, BEFORE the restart return: WindBCs
  // reads bhl.b0 (as by_inj) to inject the field and runs on restart too, so
  // if this lived only in the IC block below it would be 0 after a restart
  // (-> zero field injected, growing field-free cavity at the inflow).
  if (pmbp->pmhd != nullptr) {
    Real beta_target = pin->GetOrAddReal("problem", "beta_target", 100.0);
    // GAS (comoving) plasma-beta: beta_inf = 2 p_gas / b^2_comoving, with
    // b^2_comoving = B^2 (1 - v_inf^2) for B perp v  ->  b0 = sqrt(2*p_gas/(beta*(1-v^2))).
    // Match the old pgen's INITIAL field exactly. Its beta_target denotes gas
    // beta; derive the actual total-pressure beta rather than resetting it to 10.
    Real p_gas = bhl.rho0*bhl.t0;
    if (!std::isfinite(beta_target) || beta_target <= 0.0 ||
        !std::isfinite(p_gas) || p_gas <= 0.0 ||
        !std::isfinite(bhl.prad0) || bhl.prad0 < 0.0) {
      std::cout << "### FATAL ERROR: invalid initial pressure or beta." << std::endl;
      exit(EXIT_FAILURE);
    }
    Real one_m_v2 = 1.0 - bhl.v_inf * bhl.v_inf;
    bhl.b0 = sqrt(2.0 * p_gas / (beta_target * one_m_v2));
    bhl.beta_total0 = beta_target*(1.0 + bhl.prad0/p_gas);
    SetRealPrecise(pin, "problem", "density_ramp_beta_total0", bhl.beta_total0);
  }

  // Read the schedule on BOTH fresh starts and restarts. BuildTreeFromRestart
  // has already restored mesh time. GetOrAddReal saves the first checkpoint time
  // in ParameterInput, which RestartOutput serializes in all subsequent files.
  bhl.density_unit = pin->GetReal("units", "density_cgs");
  const Real final_cgs = pin->GetOrAddReal("problem", "density_ramp_final_cgs", 1.0e-12);
  bhl.ramp_start = pin->GetOrAddReal("problem", "density_ramp_start", pmy_mesh_->time);
  SetRealPrecise(pin, "problem", "density_ramp_start", bhl.ramp_start);
  bhl.ramp_duration = pin->GetOrAddReal("problem", "density_ramp_duration", 30000.0);
  bhl.ramp_hold = pin->GetOrAddReal("problem", "density_ramp_hold", 20000.0);
  bhl.scale_b = pin->GetOrAddBoolean("problem", "density_ramp_scale_b", true);
  if (!std::isfinite(bhl.density_unit) || bhl.density_unit <= 0.0 ||
      !std::isfinite(bhl.rho0) || bhl.rho0 <= 0.0 ||
      !std::isfinite(bhl.t0) || bhl.t0 <= 0.0 ||
      !std::isfinite(final_cgs) || final_cgs <= 0.0 ||
      !std::isfinite(bhl.ramp_start) || bhl.ramp_start < 0.0 ||
      !std::isfinite(bhl.ramp_duration) || bhl.ramp_duration <= 0.0 ||
      !std::isfinite(bhl.ramp_hold) || bhl.ramp_hold < 0.0) {
    std::cout << "### FATAL ERROR: density ramp requires finite positive densities, "
              << "temperature and duration, and nonnegative start/hold times." << std::endl;
    exit(EXIT_FAILURE);
  }
  bhl.rho_final = final_cgs/bhl.density_unit;
  const Real ramp_end = bhl.ramp_start + bhl.ramp_duration + bhl.ramp_hold;
  if (!std::isfinite(bhl.rho_final) || bhl.rho_final <= 0.0 ||
      bhl.rho_final > bhl.rho0 || !std::isfinite(ramp_end)) {
    std::cout << "### FATAL ERROR: final density must be <= initial density; "
              << "density conversion and ramp end must be finite." << std::endl;
    exit(EXIT_FAILURE);
  }
  auto &eos = (pmbp->pmhd != nullptr) ? pmbp->pmhd->peos->eos_data
                                     : pmbp->phydro->peos->eos_data;
  ramp_floors.track = pin->GetOrAddBoolean("problem", "density_ramp_track_floors", true);
  if (ramp_floors.track) {
    ramp_floors.local_entropy =
        pin->GetOrAddBoolean("problem", "density_ramp_local_entropy", true);
    // Capture the ORIGINAL floor anchors once. Restarted headers contain the
    // current floors, so re-reading them as anchors would scale them twice.
    ramp_floors.dfloor0 = ReadFloorAnchor(pin, "dfloor0", eos.dfloor);
    ramp_floors.pfloor0 = ReadFloorAnchor(pin, "pfloor0", eos.pfloor);
    ramp_floors.dexcise0 = ReadFloorAnchor(pin, "dexcise0", coord.dexcise);
    ramp_floors.pexcise0 = ReadFloorAnchor(pin, "pexcise0", coord.pexcise);
    ramp_floors.sfloor0 = ReadFloorAnchor(pin, "sfloor0", eos.sfloor);
    ramp_floors.sfloor10 = ReadFloorAnchor(pin, "sfloor10", eos.sfloor1);
    ramp_floors.sfloor20 = ReadFloorAnchor(pin, "sfloor20", eos.sfloor2);
    ramp_floors.rho10 = ReadFloorAnchor(pin, "rho10", eos.rho1);
    ramp_floors.rho20 = ReadFloorAnchor(pin, "rho20", eos.rho2);
    const Real mid_cgs = pin->GetOrAddReal("problem", "density_ramp_floor_mid_cgs", 1e-9);
    const Real low_cgs = pin->GetOrAddReal("problem", "density_ramp_floor_low_cgs", 1e-12);
    const Real mid_s = pin->GetOrAddReal("problem", "density_ramp_sfloor_mid", 1e-10);
    const Real low_s = pin->GetOrAddReal("problem", "density_ramp_sfloor_low", 1e-9);
    ramp_floors.rho_mid = mid_cgs/bhl.density_unit;
    ramp_floors.rho_low = low_cgs/bhl.density_unit;
    ramp_floors.s_mid = mid_s*std::pow(ramp_floors.rho_mid, 1.0-bhl.gamma_adi);
    ramp_floors.s_low = low_s*std::pow(ramp_floors.rho_low, 1.0-bhl.gamma_adi);
    const Real anchors[] = {ramp_floors.dfloor0, ramp_floors.pfloor0,
        ramp_floors.dexcise0, ramp_floors.pexcise0, ramp_floors.sfloor0,
        ramp_floors.sfloor10, ramp_floors.sfloor20, ramp_floors.rho10,
        ramp_floors.rho20, ramp_floors.rho_mid, ramp_floors.rho_low,
        ramp_floors.s_mid, ramp_floors.s_low};
    bool valid = bhl.gamma_adi > 1.0 && ramp_floors.dfloor0 < bhl.rho0 &&
                 ramp_floors.pfloor0 < bhl.rho0*bhl.t0 &&
                 ramp_floors.rho20 > ramp_floors.rho10 &&
                 bhl.rho0 > ramp_floors.rho_mid &&
                 ramp_floors.rho_mid > ramp_floors.rho_low;
    for (const Real value : anchors) valid = valid && std::isfinite(value) && value > 0;
    if (!valid) {
      std::cout << "### FATAL ERROR: invalid density-ramp floor anchors or entropy "
                << "density ordering (start > mid > low required)." << std::endl;
      exit(EXIT_FAILURE);
    }
    // Restore floors BEFORE Driver initialization and its first C2P call.
    UpdateRampFloors(pmy_mesh_, pin, pmy_mesh_->time);
    TaskID none(0);
    Mesh *pm = pmy_mesh_;
    // before_stagen completes before reconstruction, FOFC and radiation coupling.
    pmbp->tl_map["before_stagen"]->AddTask(
        [pm, pin](Driver*, int) {
          UpdateRampFloors(pm, pin, pm->time);
          return TaskStatus::complete;
        }, none);
    // Driver increments mesh time immediately after this list, then writes
    // outputs and performs AMR. Put the endpoint floors in those checkpoints.
    pmbp->tl_map["after_timeintegrator"]->AddTask(
        [pm, pin](Driver*, int) {
          UpdateRampFloors(pm, pin, pm->time + pm->dt);
          return TaskStatus::complete;
        }, none);
  } else if (eos.dfloor >= bhl.rho_final || eos.pfloor >= bhl.rho_final*bhl.t0) {
    std::cout << "### FATAL ERROR: gas floors exceed the final inflow state in "
              << "fixed-floor mode. Enable density_ramp_track_floors or lower "
              << "the density and pressure floors." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Driver is constructed after UserProblem, so this sets its absolute tlim.
  // Disable this for short tests or to deliberately extend the final hold.
  if (pin->GetOrAddBoolean("problem", "density_ramp_set_tlim", true)) {
    SetRealPrecise(pin, "time", "tlim", ramp_end);
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(16)
              << "BHL density ramp: start=" << bhl.ramp_start
              << " duration=" << bhl.ramp_duration << " hold=" << bhl.ramp_hold
              << " end=" << ramp_end << " M; rho_cgs="
              << bhl.rho0*bhl.density_unit << " -> " << final_cgs
              << "; B=" << (bhl.scale_b ? "constant total-pressure beta" : "fixed amplitude")
              << "; initial beta_total=" << bhl.beta_total0
              << std::endl;
  }

  // Return on restart AFTER the params above: the reservoir BC and history
  // function run on restart too and read this struct, so it must be populated
  // (else the BC pins all boundary ghosts to vacuum -> global NaN).
  if (restart) return;

  // initialize primitive variables for new run ---------------------------------------
  //   uniform: rho = rho0, p = rho0*t0, uu^i = (gamma_inf*v_inf, 0, 0) outside excision
  //   excision: rho = dexcise, p = pexcise, uu^i = 0
  //   initial uniform B_y via vector potential (below)

  auto trs = bhl;
  auto &size = pmbp->pmb->mb_size;

  par_for("pgen_bhl_ic", DevExeSpace(), 0,nmb-1, ks,ke, js,je, is,ie,
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

    // Excision check: use the cell corner farthest from origin so that cells
    // with only a tiny tip inside the horizon are NOT excised.
    Real r_excise, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(trs, x1v + copysign(0.5*dx1, x1v),
                                      x2v + copysign(0.5*dx2, x2v),
                                      x3v + copysign(0.5*dx3, x3v),
                                      &r_excise, &theta_excise, &phi_excise);

    Real rho, pgas, urad = 0.0;
    Real uu1, uu2, uu3;
    if (r_excise > 1.0) {
      // Asymptotic-wind state (along +x); uu^x = gamma_inf * v_inf (normal-frame).
      rho  = trs.rho0;
      pgas = rho * trs.t0;
      if (is_radiation_enabled) urad = trs.arad * SQR(SQR(trs.t0));
      uu1 = trs.uu_wind;  uu2 = 0.0;  uu3 = 0.0;
    } else {
      // Inside horizon: apply excision floors, zero velocity
      rho  = trs.dexcise;
      pgas = trs.pexcise;
      uu1 = 0.0;  uu2 = 0.0;  uu3 = 0.0;
    }

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
    // Uniform horizontal B = b0 * y_hat (coordinate-frame B^y).  bhl.b0 was
    // already computed above (before the restart return) from beta_target:
    //   comoving b^2 = b0^2 * (1 - v_inf^2)        (B perp to v)
    //   beta_inf = p_gas / (b^2/2)                 (K&M Eq. 22-23)
    //   =>  b0 = sqrt( 2 * p_gas / (beta_target * (1 - v_inf^2)) )

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
    auto trs = bhl;

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
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bhl_pgen pgen,
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
// Vector potential for a uniform B-field along +y in Cartesian Kerr-Schild:
//   A_x = b0 * x3,  A_y = A_z = 0  =>  B = curl A = (0, b0, 0).
// Same prescription as Kim & Most 2025 (their Eq. 21 with theta_B = 90 deg):
// asymptotically uniform B, no Wald-style spin correction.

KOKKOS_INLINE_FUNCTION
Real A1(struct bhl_pgen pgen, Real x1, Real x2, Real x3) {
  return pgen.b0 * x3;
}

KOKKOS_INLINE_FUNCTION
Real A2(struct bhl_pgen pgen, Real x1, Real x2, Real x3) {
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
Real A3(struct bhl_pgen pgen, Real x1, Real x2, Real x3) {
  return 0.0;
}

} // namespace

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! \fn WindEquilibIntensity
//  \brief Coordinate-frame specific intensity (AthenaK i0 convention) of an
//  isotropic blackbody of energy density urad in the comoving frame of gas
//  moving with normal-frame velocity (uu_wind, 0, 0), evaluated for angle n at
//  the active boundary cell (m, ka, ja, ia).  This is the SAME analytic formula
//  used by the IC and the inner_x1 injection -- factored out so the 5 outflow
//  faces can inject the ambient wind-boosted equilibrium for their incoming
//  directions.  nh{0..3} are the components of discrete direction n (nh_c).
KOKKOS_INLINE_FUNCTION
Real WindEquilibIntensity(int m, int ka, int ja, int ia,
                          Real nh0, Real nh1, Real nh2, Real nh3,
                          Real x1v, Real x2v, Real x3v,
                          Real uu_wind, Real urad_inj,
                          bool is_minkowski, Real bh_spin,
                          const DvceArray6D<Real> &ntt,
                          const DvceArray6D<Real> &tetc,
                          const DvceArray6D<Real> &tcov) {
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v, x2v, x3v, is_minkowski, bh_spin, glower, gupper);
  // wind 4-velocity (normal frame), purely along x1
  Real uu1 = uu_wind;
  Real q   = glower[1][1]*uu1*uu1;
  Real uu0 = sqrt(1.0 + q);
  // boost into the local tetrad (only columns 0,1 contribute since uu2=uu3=0)
  Real ut0 = ntt(m,0,0,ka,ja,ia)*uu0 + ntt(m,0,1,ka,ja,ia)*uu1;
  Real ut1 = ntt(m,1,0,ka,ja,ia)*uu0 + ntt(m,1,1,ka,ja,ia)*uu1;
  Real ut2 = ntt(m,2,0,ka,ja,ia)*uu0 + ntt(m,2,1,ka,ja,ia)*uu1;
  Real ut3 = ntt(m,3,0,ka,ja,ia)*uu0 + ntt(m,3,1,ka,ja,ia)*uu1;
  Real un_t = ut1*nh1 + ut2*nh2 + ut3*nh3;
  Real n0_f = ut0*nh0 - un_t;                 // photon frequency in the wind frame
  Real n0  = tetc(m,0,0,ka,ja,ia);
  Real n_0 = tcov(m,0,0,ka,ja,ia)*nh0 + tcov(m,1,0,ka,ja,ia)*nh1 +
             tcov(m,2,0,ka,ja,ia)*nh2 + tcov(m,3,0,ka,ja,ia)*nh3;
  return n0*n_0*(urad_inj/(4.0*M_PI))/SQR(SQR(n0_f));
}

//! \fn WindBCs
//  \brief BHL-wind boundary conditions (wind along +x).
//         inner_x1 (-x face): INJECTION.  Dirichlet on hydro (rho_inf(t), p_inf(t),
//             uu^x=gamma_inf*v_inf), pinned face B = (0, By_inf(t), 0), thermal radiation at t0
//             boosted with the wind (same tetrad formula as IC).
//         Other 5 faces: zero-gradient outflow for hydro and face B (copy from
//             the innermost active cell).  Radiation: OUTGOING directions copy
//             (escape); INCOMING directions are set to the ambient wind-boosted
//             equilibrium at t0 via WindEquilibIntensity() (no longer zeroed).
// FIXME: Boundaries need to be adjusted for DynGRMHD

void WindBCs(Mesh *pm) {
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
  DvceArray6D<Real> tc, norm_to_tet_, tetcov_c_;
  DualArray2D<Real> nh_c_;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    nh_c_ = pm->pmb_pack->prad->nh_c;
    tc    = pm->pmb_pack->prad->tet_c;
    norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;
    tetcov_c_    = pm->pmb_pack->prad->tetcov_c;
  }

  // Geometry / coord data needed for injection face metric
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;

  // State captured by value into device lambdas
  const Real gm1     = bhl.gamma_adi - 1.0;
  // The boundary callback has no RK-stage time argument. Evaluating at mesh
  // time introduces O(dt) forcing-time error, negligible for a 30000 M ramp.
  const Real rho_inj  = bhl.InflowDensity(pm->time);
  const Real pgas_inj = rho_inj * bhl.t0;
  const Real uu_wind  = bhl.uu_wind;  // gamma_inf * v_inf (normal-frame velocity)
  const Real by_inj   = bhl.InflowField(pm->time);
  const Real urad_inj = (is_radiation_enabled) ? bhl.arad * SQR(SQR(bhl.t0)) : 0.0;

  // X1-Boundary fields:
  //   inner_x1 = injection: pin uniform horizontal B (x1f=0, x2f=by_inj, x3f=0)
  //   outer_x1 = outflow:   copy interior face values
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("bfield_x1_bc", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = 0.0;
          b0.x2f(m,k,j,is-i-1) = by_inj;
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = by_inj;}
          b0.x3f(m,k,j,is-i-1) = 0.0;
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = 0.0;}
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
  // X1 hydro BC:
  //   inner_x1 = injection: Dirichlet on (rho_inf(t), p_inf(t), uu^x=gamma_inf*v_inf, uu^y=uu^z=0)
  //   outer_x1 = outflow:   pure copy from innermost active cell
  par_for("hydro_x1_bc", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        int ig = is - i - 1;
        w0_(m,IDN,k,j,ig) = rho_inj;
        w0_(m,IEN,k,j,ig) = pgas_inj / gm1;
        w0_(m,IVX,k,j,ig) = uu_wind;
        w0_(m,IVY,k,j,ig) = 0.0;
        w0_(m,IVZ,k,j,ig) = 0.0;
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        int ig = ie + i + 1;
        w0_(m,IDN,k,j,ig) = w0_(m,IDN,k,j,ie);
        w0_(m,IEN,k,j,ig) = w0_(m,IEN,k,j,ie);
        w0_(m,IVX,k,j,ig) = w0_(m,IVX,k,j,ie);
        w0_(m,IVY,k,j,ig) = w0_(m,IVY,k,j,ie);
        w0_(m,IVZ,k,j,ig) = w0_(m,IVZ,k,j,ie);
      }
    }
  });
  if (is_radiation_enabled) {
    // inner_x1 = injection: thermal radiation at t0 boosted with wind (along +x)
    //                       Evaluated at active cell (k, j, is).
    // outer_x1 = outflow:   no incoming intensity, copy outgoing.
    par_for("rad_x1_bc", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x1v = CellCenterX(0,    indcs.nx1, x1min, x1max);  // is → local 0
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                                glower, gupper);
        Real uu1 = uu_wind, uu2 = 0.0, uu3 = 0.0;
        Real q   = glower[1][1]*uu1*uu1;
        Real uu0 = sqrt(1.0 + q);
        Real ut0 = (norm_to_tet_(m,0,0,k,j,is)*uu0 + norm_to_tet_(m,0,1,k,j,is)*uu1);
        Real ut1 = (norm_to_tet_(m,1,0,k,j,is)*uu0 + norm_to_tet_(m,1,1,k,j,is)*uu1);
        Real ut2 = (norm_to_tet_(m,2,0,k,j,is)*uu0 + norm_to_tet_(m,2,1,k,j,is)*uu1);
        Real ut3 = (norm_to_tet_(m,3,0,k,j,is)*uu0 + norm_to_tet_(m,3,1,k,j,is)*uu1);
        Real un_t = ut1*nh1 + ut2*nh2 + ut3*nh3;
        Real n0_f = ut0*nh0 - un_t;
        Real n0 = tc(m,0,0,k,j,is);
        Real n_0 = tetcov_c_(m,0,0,k,j,is)*nh0 + tetcov_c_(m,1,0,k,j,is)*nh1 +
                   tetcov_c_(m,2,0,k,j,is)*nh2 + tetcov_c_(m,3,0,k,j,is)*nh3;
        Real i_inj = n0*n_0*(urad_inj/(4.0*M_PI))/SQR(SQR(n0_f));
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i_inj;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        Real n1 = tc(m,0,1,k,j,ie)*nh0 + tc(m,1,1,k,j,ie)*nh1 + tc(m,2,1,k,j,ie)*nh2 + tc(m,3,1,k,j,ie)*nh3;
        // incoming (n1<0, into domain): ambient wind equilibrium; outgoing: copy (escape)
        Real x1v = CellCenterX(indcs.nx1-1, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real x2v = CellCenterX(j-js,        indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real x3v = CellCenterX(k-ks,        indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real i_amb = WindEquilibIntensity(m, k, j, ie, nh0,nh1,nh2,nh3, x1v,x2v,x3v,
                          uu_wind, urad_inj, coord.is_minkowski, coord.bh_spin,
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

  // X2-Boundary fields: zero-gradient outflow on both faces
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("outflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
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
  // Zero-gradient (pure copy) outflow on X2 ghost zones
  par_for("outflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        int jg = js - j - 1;
        w0_(m,IDN,k,jg,i) = w0_(m,IDN,k,js,i);
        w0_(m,IEN,k,jg,i) = w0_(m,IEN,k,js,i);
        w0_(m,IVX,k,jg,i) = w0_(m,IVX,k,js,i);
        w0_(m,IVY,k,jg,i) = w0_(m,IVY,k,js,i);
        w0_(m,IVZ,k,jg,i) = w0_(m,IVZ,k,js,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        int jg = je + j + 1;
        w0_(m,IDN,k,jg,i) = w0_(m,IDN,k,je,i);
        w0_(m,IEN,k,jg,i) = w0_(m,IEN,k,je,i);
        w0_(m,IVX,k,jg,i) = w0_(m,IVX,k,je,i);
        w0_(m,IVY,k,jg,i) = w0_(m,IVY,k,je,i);
        w0_(m,IVZ,k,jg,i) = w0_(m,IVZ,k,je,i);
      }
    }
  });
  if (is_radiation_enabled) {
    par_for("outflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        Real n2 = tc(m,0,2,k,js,i)*nh0 + tc(m,1,2,k,js,i)*nh1 + tc(m,2,2,k,js,i)*nh2 + tc(m,3,2,k,js,i)*nh3;
        // incoming (n2>0, into domain): ambient wind equilibrium; outgoing: copy (escape)
        Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real x2v = CellCenterX(0,    indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real i_amb = WindEquilibIntensity(m, k, js, i, nh0,nh1,nh2,nh3, x1v,x2v,x3v,
                          uu_wind, urad_inj, coord.is_minkowski, coord.bh_spin,
                          norm_to_tet_, tc, tetcov_c_);
        Real val = (n2 > 0) ? i_amb : i0_(m,n,k,js,i);
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        Real n2 = tc(m,0,2,k,je,i)*nh0 + tc(m,1,2,k,je,i)*nh1 + tc(m,2,2,k,je,i)*nh2 + tc(m,3,2,k,je,i)*nh3;
        // incoming (n2<0, into domain): ambient wind equilibrium; outgoing: copy (escape)
        Real x1v = CellCenterX(i-is,         indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real x2v = CellCenterX(indcs.nx2-1,  indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real x3v = CellCenterX(k-ks,         indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real i_amb = WindEquilibIntensity(m, k, je, i, nh0,nh1,nh2,nh3, x1v,x2v,x3v,
                          uu_wind, urad_inj, coord.is_minkowski, coord.bh_spin,
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

  // X3-Boundary fields: zero-gradient outflow on both faces
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("outflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
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
  // Zero-gradient (pure copy) outflow on X3 ghost zones
  par_for("outflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        int kg = ks - k - 1;
        w0_(m,IDN,kg,j,i) = w0_(m,IDN,ks,j,i);
        w0_(m,IEN,kg,j,i) = w0_(m,IEN,ks,j,i);
        w0_(m,IVX,kg,j,i) = w0_(m,IVX,ks,j,i);
        w0_(m,IVY,kg,j,i) = w0_(m,IVY,ks,j,i);
        w0_(m,IVZ,kg,j,i) = w0_(m,IVZ,ks,j,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        int kg = ke + k + 1;
        w0_(m,IDN,kg,j,i) = w0_(m,IDN,ke,j,i);
        w0_(m,IEN,kg,j,i) = w0_(m,IEN,ke,j,i);
        w0_(m,IVX,kg,j,i) = w0_(m,IVX,ke,j,i);
        w0_(m,IVY,kg,j,i) = w0_(m,IVY,ke,j,i);
        w0_(m,IVZ,kg,j,i) = w0_(m,IVZ,ke,j,i);
      }
    }
  });
  if (is_radiation_enabled) {
    par_for("outflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      Real nh0 = nh_c_.d_view(n,0);
      Real nh1 = nh_c_.d_view(n,1);
      Real nh2 = nh_c_.d_view(n,2);
      Real nh3 = nh_c_.d_view(n,3);
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        Real n3 = tc(m,0,3,ks,j,i)*nh0 + tc(m,1,3,ks,j,i)*nh1 + tc(m,2,3,ks,j,i)*nh2 + tc(m,3,3,ks,j,i)*nh3;
        // incoming (n3>0, into domain): ambient wind equilibrium; outgoing: copy (escape)
        Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real x3v = CellCenterX(0,    indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real i_amb = WindEquilibIntensity(m, ks, j, i, nh0,nh1,nh2,nh3, x1v,x2v,x3v,
                          uu_wind, urad_inj, coord.is_minkowski, coord.bh_spin,
                          norm_to_tet_, tc, tetcov_c_);
        Real val = (n3 > 0) ? i_amb : i0_(m,n,ks,j,i);
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = val;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        Real n3 = tc(m,0,3,ke,j,i)*nh0 + tc(m,1,3,ke,j,i)*nh1 + tc(m,2,3,ke,j,i)*nh2 + tc(m,3,3,ke,j,i)*nh3;
        // incoming (n3<0, into domain): ambient wind equilibrium; outgoing: copy (escape)
        Real x1v = CellCenterX(i-is,        indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        Real x2v = CellCenterX(j-js,        indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        Real x3v = CellCenterX(indcs.nx3-1, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        Real i_amb = WindEquilibIntensity(m, ke, j, i, nh0,nh1,nh2,nh3, x1v,x2v,x3v,
                          uu_wind, urad_inj, coord.is_minkowski, coord.bh_spin,
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

void WindFluxes(HistoryData *pdata, Mesh *pm) {
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

  pdata->nhist = nradii*nflux + 11;
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

  // HistoryOutput sums over ranks; contribute an equal fraction of each
  // spatially uniform reservoir diagnostic, not a mesh volume integral.
  const int offset = nradii*nflux;
  const Real rank_weight = 1.0/static_cast<Real>(global_variable::nranks);
  const Real rho_inj = bhl.InflowDensity(pm->time);
  pdata->label[offset  ] = "rho_inf";
  pdata->label[offset+1] = "rho_cgs";
  pdata->label[offset+2] = "By_inf";
  pdata->label[offset+3] = "t_ramp";
  pdata->hdata[offset  ] = rank_weight*rho_inj;
  pdata->hdata[offset+1] = rank_weight*rho_inj*bhl.density_unit;
  pdata->hdata[offset+2] = rank_weight*bhl.InflowField(pm->time);
  pdata->hdata[offset+3] = rank_weight*(pm->time - bhl.ramp_start);
  const auto &eos = (pmbp->pmhd != nullptr) ? pmbp->pmhd->peos->eos_data
                                         : pmbp->phydro->peos->eos_data;
  pdata->label[offset+4] = "dfloor";
  pdata->label[offset+5] = "pfloor";
  pdata->label[offset+6] = "sf_at_inf";
  pdata->label[offset+7] = "dexcise";
  pdata->hdata[offset+4] = rank_weight*eos.dfloor;
  pdata->hdata[offset+5] = rank_weight*eos.pfloor;
  const Real sf = (pmbp->pmhd != nullptr || eos.bhl_local_entropy) ?
                  eos.EntropyFloor(rho_inj) : eos.sfloor;
  pdata->hdata[offset+6] = rank_weight*sf;
  pdata->hdata[offset+7] = rank_weight*pmbp->pcoord->coord_data.dexcise;
  const Real pgas = rho_inj*bhl.t0;
  const Real field = bhl.InflowField(pm->time);
  pdata->label[offset+8] = "pgas_inf";
  pdata->label[offset+9] = "prad_inf";
  pdata->label[offset+10] = "beta_tot";
  pdata->hdata[offset+8] = rank_weight*pgas;
  pdata->hdata[offset+9] = rank_weight*bhl.prad0;
  pdata->hdata[offset+10] = is_mhd ? rank_weight*2.0*(pgas+bhl.prad0)/
      (SQR(field)*(1.0-SQR(bhl.v_inf))) : 0.0;

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
