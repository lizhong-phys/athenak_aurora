#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class GeodesicGrid;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks

struct RadiationTaskIDs {
  TaskID rad_irecv;
  TaskID mhd_irecv;
  TaskID hyd_irecv;
  TaskID copyu;
  TaskID rad_flux;
  TaskID mhd_flux;
  TaskID hyd_flux;
  TaskID rad_sendf;
  TaskID mhd_sendf;
  TaskID hyd_sendf;
  TaskID rad_recvf;
  TaskID mhd_recvf;
  TaskID hyd_recvf;
  TaskID rad_rkupdt;
  TaskID mhd_rkupdt;
  TaskID hyd_rkupdt;
  TaskID rad_src;
  TaskID mhd_src;
  TaskID hyd_src;
  TaskID rad_coupl;
  TaskID rad_resti;
  TaskID hyd_restu;
  TaskID mhd_restu;
  TaskID rad_sendi;
  TaskID mhd_sendu;
  TaskID hyd_sendu;
  TaskID rad_recvi;
  TaskID mhd_recvu;
  TaskID hyd_recvu;
  TaskID mhd_efld;
  TaskID mhd_sende;
  TaskID mhd_recve;
  TaskID mhd_ct;
  TaskID mhd_restb;
  TaskID mhd_sendb;
  TaskID mhd_recvb;
  TaskID bcs;
  TaskID rad_prol;
  TaskID mhd_prol;
  TaskID hyd_prol;
  TaskID mhd_c2p;
  TaskID hyd_c2p;
  TaskID rad_csend;
  TaskID mhd_csend;
  TaskID hyd_csend;
  TaskID rad_crecv;
  TaskID mhd_crecv;
  TaskID hyd_crecv;
};

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Radiation

class Radiation {
 public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // flags to denote hydro/mhd is enabled or units enabled
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool are_units_enabled;

  // Radiation coupling term parameters
  bool rad_source;          // flag to enable/disable radiation source term
  bool fixed_fluid;         // flag to enable/disable fluid integration
  bool affect_fluid;        // flag to enable/disable feedback of rad field on fluid
  Real arad;                // radiation constant
  Real kappa_a;             // constant Rosseland mean absoprtion coefficient
  Real kappa_s;             // constant scattering coefficient
  Real kappa_p;             // Planck - Rosseland mean coefficient
  bool power_opacity;       // flag to enable Kramer's law opacity for kappa_a
  bool is_compton_enabled;  // flag to enable/disable compton

  // Flags and parameters for ad hoc fixes
  bool correct_radsrc_velocity;
  // Density-drop limiter (conservative overshoot guard; requires the radiation-frame estimate,
  // i.e. correct_radsrc_velocity or limit_opacity, so E_rad_f is computed).  D=rho*u^0 is
  // conserved across the radiation source substep, so rho_post/rho_pre = W_pre/W_post.  In the
  // stiff, radiation-dominated regime (Lambda_s = (dt*sigma_s/u0)*(E_rad_f/rho_h) >
  // rad_momentum_lambda_lock) the explicit momentum kick can drive W to the ceiling and collapse
  // the density.  There we scale the WHOLE source exchange (intensities AND gas) by a scalar
  // lambda<=1 chosen so the estimated density drop is <= rad_max_density_drop -- conservative
  // (four-momentum preserving) and inactive elsewhere (physics unchanged).
  bool rad_dvlimit;
  Real rad_max_density_drop;      // f_rho: max rho_pre/rho_post per source update (>1)
  Real rad_momentum_lambda_lock;  // Lambda_lock: stiffness*dominance above which the guard acts
  bool correct_radsrc_opacity;
  Real dfloor_opacity;
  Real dens_trunc_max;
  Real tau_truncation;
  Real sigmoid_residual; // sigmoid residual must be less than 1./3
  // opacity-aware stiffness limiter (EMERGENCY STABILIZER / opacity closure, NOT a
  // physics-preserving model): on cells collapsed onto the entropy floor, cap the Planck
  // absorption (sigma_a+sigma_p, both ~T^-7/2) at the value the gas would have at T_eff =
  // T*max[1,(Xi_abs/Xi_max)^(2/7)] (capped at the virial T).  Reverses the absorption
  // density->cold->opaque->overshoot feedback.  Note Xi_abs contains dt (timestep-dependent).
  bool limit_opacity;    // enable the limiter
  Real opacity_xi_max;   // Xi_max: max proper-time absorption stiffness dt*(sig_a+sig_p)/u0
  Real opacity_tcap;     // virial ceiling on the effective opacity temperature (code units)
  // per-cell radiation-source diagnostics [nmb,17,k,j,i] (alloc iff limit_opacity,
  // correct_radsrc_velocity, or rad_dvlimit).  Opacity limiter slots 0-8: 0 on_entropy_floor,
  // 1 Xi_abs_raw, 2 Xi_abs_limited, 3 T_eff, 4 limiter_active, 5 limiter_hit_tcap, 6 sigma_a,
  // 7 sigma_p, 8 sigma_s.  Velocity-correction slots 9-14: 9 E_rad_f, 10 rho_h,
  // 11 gate(erad_f>rho+e), 12 urad_W (inferred radiation-frame Lorentz factor), 13 vrad_sq_raw
  // (pre-clamp; >v_sq_max => superluminal/corrupted), 14 rr_tet00 (tetrad radiation energy;
  // small/neg => corrupt).  Density-drop limiter slots 15-16: 15 Lambda_s (stiffness*dominance),
  // 16 lambda (source-exchange scale; <1 => guard fired, 1 => untouched).
  DvceArray5D<Real> limiter_diag;

  // radiation source term (i.e., beam)
  SourceTerms *psrc = nullptr;

  // Angular mesh
  bool rotate_geo;                    // rotate geodesic mesh
  bool angular_fluxes;                // flag to enable/disable angular fluxes
  Real n_0_floor;                     // floor on n_0
  GeodesicGrid *prgeo = nullptr;      // pointer to radiation angular mesh

  // Tetrad arrays and functions
  DualArray2D<Real> nh_c;             // normal vector computed at face center
  DualArray3D<Real> nh_f;             // normal vector computed at face edges
  DvceArray6D<Real> tet_c;            // tetrad components at cell centers
  DvceArray6D<Real> tetcov_c;         // covariant tetrad components at cell centers
  DvceArray5D<Real> tet_d1_x1f;       // tetrad components (subset) at x1f
  DvceArray5D<Real> tet_d2_x2f;       // tetrad components (subset) at x2f
  DvceArray5D<Real> tet_d3_x3f;       // tetrad components (subset) at x3f
  DvceArray6D<Real> na;               // n^a
  DvceArray6D<Real> norm_to_tet;      // used in transform b/w normal frame and tet frame
  void SetOrthonormalTetrad();

  // intensity arrays
  DvceArray5D<Real> i0;         // intensities
  DvceArray5D<Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for i
  MeshBoundaryValuesCC *pbval_i;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;         // intensity at intermediate step
  DvceFaceFld5D<Real> iflx;     // spatial fluxes on zone faces
  DvceArray5D<Real> divfa;      // angular flux divergence
  Real dtnew;

  // reconstruction method
  ReconstructionMethod recon_method;

  // container to hold names of TaskIDs
  RadiationTaskIDs id;

  // functions...
  void AssembleRadTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalculateFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus RadSrcTerms(Driver *d, int stage);
  TaskStatus RadFluidCoupling(Driver *d, int stage);
  TaskStatus RestrictI(Driver *d, int stage);
  TaskStatus SendI(Driver *d, int stage);
  TaskStatus RecvI(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
