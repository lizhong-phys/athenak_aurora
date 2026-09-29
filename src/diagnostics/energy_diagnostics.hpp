#ifndef DIAGNOSTICS_ENERGY_DIAGNOSTICS_HPP_
#define DIAGNOSTICS_ENERGY_DIAGNOSTICS_HPP_
//========================================================================================
// Passive, runtime-gated energy diagnostics for radiation GRMHD.
//
// The object is constructed only when <problem>/energy_diagnostics=true.  Its arrays are
// detached from the evolved state and are never used by fluxes, source terms, primitive
// recovery, AMR decisions, or timestep selection.
//========================================================================================

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "diagnostics/observer.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;
class Mesh;

namespace diagnostics {
// Optional end-of-run read-only state fingerprint, independent of the budget switch.
void WriteEnergyFingerprint(ParameterInput *pin, Mesh *mesh);

enum EnergyDiagIndex {
  ED_GAS_FLUX = 0,       // complete conservative MHD flux-divergence increment
  ED_GAS_HLLE,           // subset associated with the HLLE dissipative face flux
  ED_GAS_FOFC,           // subset caused by replacing high-order fluxes with FOFC fluxes
  ED_GAS_COORD,          // stationary metric/coordinate source increment
  ED_GAS_OTHER,          // explicit non-coordinate and user source increments
  ED_GAS_RAD_TOTAL,      // exact gas change across the radiation-coupling operator
  ED_GAS_RAD_ABS,        // reserved; not populated by this passive implementation
  ED_GAS_RAD_COMPT,      // reserved; not populated by this passive implementation
  ED_RAD_LIMIT_REJECT,   // reserved; rejected proposals are not instrumented
  ED_RAD_SPATIAL,        // radiation spatial-transport increment
  ED_RAD_ANGULAR,        // radiation angular-transport increment
  ED_RAD_FIX,            // positivity/excision correction in the radiation update
  ED_RAD_SOURCE_TOTAL,   // exact radiation change across the coupling operator
  ED_GAS_C2P,            // exact conserved-energy change made by C2P/floors/excision
  ED_DS_ADVECT,           // RK-integrated -dt div(F_S), auxiliary entropy face flux
  ED_DEINT_ADVECT,        // RK-integrated -dt div(F_e), auxiliary internal-energy flux
  ED_DU_ADVECT,           // RK-integrated change -dt div(u^i_face)
  ED_DPCOMP_SPATIAL,      // RK-integrated spatial pressure work -dt p div(u^i_face)
  ED_DM_ADVECT,           // actual accepted RK mass-flux increment
  ED_QRECOVERY,           // comoving thermal source from recovery corrections
  ED_QRECOVERY_ABS,       // RK-weighted absolute recovery source increments
  ED_QENT_RAD_LEGACY,     // old radiation contraction retained only for sign QA
  ED_QENT_RAD,            // comoving gas-to-radiation loss Lambda (positive = emission)
  ED_QENT_TOTAL,          // T[(D s)^{n+1}-(D s)^n-dS_advect]/dt
  ED_QENT_CENTERED,       // legacy centered-current entropy estimate retained for QA
  ED_QENT_NUM,            // entropy heating estimate, minus recovery source in budget mode
  ED_GAS_ACTUAL,         // measured gas conserved-energy change over the timestep
  ED_GAS_CLOSURE,        // actual minus independently recorded gas terms
  ED_RAD_ACTUAL,         // measured coordinate radiation-energy change over the timestep
  ED_RAD_CLOSURE,        // actual minus independently recorded radiation terms
  ED_INT_STORAGE,        // d(e u^t)/dt in Cartesian Kerr-Schild coordinates
  ED_INT_ADVECTION,      // div(e u^i), centered in time over the sampled timestep
  ED_COMPRESSION,        // -p div_4(u), centered in time over the sampled timestep
  ED_DISS_ENERGY,        // internal-energy residual minus recovery source in budget mode
  ED_THERMO_CLOSURE,     // energy-equation dissipation minus entropy dissipation
  ED_EXPANSION,          // covariant expansion theta = div_4(u)
  ED_INT_ADVECTION_CENTERED, // legacy centered div(e u^i), QA only
  ED_COMPRESSION_CENTERED,   // legacy centered -p div_4(u), QA only
  ED_EXPANSION_CENTERED,     // legacy centered div_4(u), QA only
  ED_MAG_LOSS,           // non-ideal comoving magnetic-energy loss residual
  ED_HYDRO_DISS,         // dissipation remaining after the signed magnetic residual
  ED_SHOCK_SENSOR,       // dimensionless compressive morphology score
  ED_CURRENT_SENSOR,     // dimensionless current-sheet morphology score
  ED_CONTACT_SENSOR,     // dimensionless density/contact morphology score
  ED_VORT_SENSOR,        // dimensionless vorticity score
  ED_SHEAR_SENSOR,       // dimensionless traceless-strain score
  ED_PRESSURE_JUMP,      // cell-scale relative pressure jump
  ED_FIELD_REVERSAL,     // antiparallel-field score across neighboring cells
  ED_RHO,                // primitive density for physics-first post-processing
  ED_EINT,               // comoving internal-energy density
  ED_PRESSURE,           // gas pressure
  ED_UT, ED_U1, ED_U2, ED_U3, // contravariant fluid four-velocity
  ED_B1, ED_B2, ED_B3,   // cell-centered coordinate magnetic field
  ED_BSQ,                // comoving magnetic four-vector squared
  ED_FLAGS,              // bit mask converted to Real for binary output
  ED_SAMPLE_DT, ED_SAMPLE_TIME, ED_SAMPLE_ID, ED_STENCIL_EDGE,
  // Independent EM four-momentum budget (not the centered q_mag_loss proxy).
  ED_EM_R0, ED_EM_R1, ED_EM_R2, ED_EM_R3,
  ED_MAG_BUDGET, ED_MECH_BUDGET, ED_MAG_SOURCE, ED_MAG_QUADRATURE,
  ED_MAG_POS, ED_MAG_NEG, ED_MECH_POS, ED_MECH_NEG, ED_TOTAL_POS, ED_TOTAL_NEG,
  ED_CT_CLOSURE, ED_GAS_CLOSURE_ABS, ED_THERMO_CLOSURE_ABS,
  ED_WINDOW_STEPS, ED_BUDGET_VERSION, ED_EM_BULK_WORK, ED_MAG_HEAT_INF, ED_MASS_D,
  ED_ENTROPY_LEGACY, ED_DISS_LEGACY, ED_ENTROPY_MASS_TERM,
  NENERGY_DIAG
};

enum EnergyDiagFlag : unsigned int {
  EDF_DENSITY_FLOOR = 1u << 0,
  EDF_ENERGY_FLOOR  = 1u << 1,
  EDF_VELOCITY_CEIL = 1u << 2,
  EDF_C2P_FAILURE   = 1u << 3,
  EDF_EXCISION      = 1u << 4,
  EDF_FOFC          = 1u << 5,
  EDF_RAD_FIX       = 1u << 6,
  EDF_RAD_LIMIT     = 1u << 7,
  EDF_RAD_BAD       = 1u << 8,
  EDF_C2P_REPAIR    = 1u << 10, // measured change, not a reconstructed floor reason
  EDF_NONFINITE     = 1u << 9
};

class EnergyDiagnostics {
 public:
  EnergyDiagnostics(MeshBlockPack *ppack, ParameterInput *pin);
  ~EnergyDiagnostics() { delete pbval_sidecar; }

  static const char *label[NENERGY_DIAG];

  bool recording = false;
  bool budget_enabled = false;  // second opt-in; continuous interval ledger
  DvceArray5D<Real> step;       // RK-consistent increments for the current timestep
  DvceArray5D<Real> output;     // rates/sensors from the last completed timestep
  DvceArray5D<Real> gas_initial;
  DvceArray5D<Real> rad_initial;
  DvceArray5D<Real> gas_scratch;
  DvceArray5D<Real> rad_scratch;
  DvceArray5D<Real> entropy_initial; // D*s at the beginning of the timestep
  DvceArray5D<Real> physical_initial; // detached primitive/geometry state at step start
  DvceArray5D<Real> physical_final;   // detached primitive/geometry state at step end
  DvceArray5D<Real> gas_four_scratch; // gas four-momentum before radiation coupling
  DvceArray5D<Real> ucon_scratch;     // fluid four-velocity used to contract coupling
  DvceArray5D<Real> rad_energy_flux;  // R^i_t, i=1..3, used for luminosity histories
  DvceFaceFld4D<Real> hlle_energy_flux;
  DvceFaceFld4D<Real> fofc_energy_flux;
  DvceFaceFld4D<Real> entropy_flux;   // passive entropy flux used only by diagnostics
  DvceFaceFld4D<Real> internal_energy_flux; // passive HLL flux of e u^mu
  DvceFaceFld4D<Real> four_velocity_flux;   // Riemann-face normal u^i for p dV work

  // The five sidecar face fields above are staged into this 5D array so they can
  // be passed through the SAME restriction/correction AthenaK applies to uflx at
  // static-refinement interfaces.  Without it the sidecar divergences do not close
  // on the two outermost cell layers of every block, which is the sole reason
  // stencil_edge marks them.  Variable order is fixed by SidecarVar below.
  enum SidecarVar {SC_HLLE=0, SC_FOFC, SC_ENTROPY, SC_EINT, SC_UVEL,
                  SC_EM0, SC_EM1, SC_EM2, SC_EM3, NSIDECAR};
  DvceFaceFld5D<Real> sidecar_flx;
  MeshBoundaryValuesCC *pbval_sidecar = nullptr;
  void GatherSidecarFluxes();     // 4D sidecars -> sidecar_flx
  void ScatterSidecarFluxes();    // sidecar_flx -> 4D sidecars
  TaskStatus SendSidecarFlux();
  TaskStatus RecvSidecarFlux();
  DvceArray4D<unsigned int> flags;

  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus BeginTimestep(Driver *pdriver, int stage);
  TaskStatus BeginStage(Driver *pdriver, int stage);
  TaskStatus FinalizeTimestep(Driver *pdriver, int stage);

  // Detached copies for reconstruction helpers whose APIs accept mutable Views.
  DvceArray5D<Real> reconstruction_w, reconstruction_b;
  void SnapshotReconstruction();
  void SavePrimitiveRecoveryState();
  void RecordPrimitiveRecovery();
  void SaveGasEnergy();
  void AccumulateGasEnergy(int channel);
  void FillPhysicalState(DvceArray5D<Real> &state);
  void SaveRadiationEnergy();
  void AccumulateRadiationEnergy(int channel);
  void RecordMHDFluxUpdate(Driver *pdriver, int stage);
  void RecordRadiationUpdate(Driver *pdriver, int stage);
  void SaveRadiationCouplingState();
  void RecordRadiationCoupling();
  void InitializeBudget();
  void BeginBudgetStep();
  void BeginBudgetStage(Driver *pdriver, int stage);
  void RecordEMFlux(Driver *pdriver, int stage);
  void SaveEMSourceState();
  void FinishBudgetStage();
  void RecordCT(Driver *pdriver, int stage);
  void FinalizeBudgetStep();
  void FinalizeBudgetThermodynamics();
  void PublishBudgetWindow(bool publish);

 private:
  // budget_state: initial mixed EM four-momentum (4), initial Bcc (3), initial u (4).
  // budget_delta: RK-integrated -div(F_EM)+G (4), source-induced EM change (4),
  //               actual corrected-edge CT curl increment (3).
  DvceArray5D<Real> budget_state, budget_delta, em_source_before, budget_window;
  DvceArray4D<unsigned int> window_flags;
  DvceArray5D<Real> recovery_before;
  Real window_dt_ = 0.0;
  int window_steps_ = 0;
  bool source_pending_ = false;
  MeshBlockPack *pmy_pack_;
  Real sample_dt_;
  Real next_sample_time_;
  int sample_number_ = 0;
};

} // namespace diagnostics
#endif // DIAGNOSTICS_ENERGY_DIAGNOSTICS_HPP_
