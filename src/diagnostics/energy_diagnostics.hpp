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
#include "tasklist/task_list.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace diagnostics {

enum EnergyDiagIndex {
  ED_GAS_FLUX = 0,       // complete conservative MHD flux-divergence increment
  ED_GAS_HLLE,           // subset associated with the HLLE dissipative face flux
  ED_GAS_FOFC,           // subset caused by replacing high-order fluxes with FOFC fluxes
  ED_GAS_COORD,          // stationary metric/coordinate source increment
  ED_GAS_OTHER,          // explicit non-coordinate and user source increments
  ED_GAS_RAD_TOTAL,      // exact gas change across the radiation-coupling operator
  ED_GAS_RAD_ABS,        // absorption/emission part (filled by coupling instrumentation)
  ED_GAS_RAD_COMPT,      // Compton part (filled by coupling instrumentation)
  ED_RAD_LIMIT_REJECT,   // proposed radiation exchange rejected by a limiter
  ED_RAD_SPATIAL,        // radiation spatial-transport increment
  ED_RAD_ANGULAR,        // radiation angular-transport increment
  ED_RAD_FIX,            // positivity/excision correction in the radiation update
  ED_RAD_SOURCE_TOTAL,   // exact radiation change across the coupling operator
  ED_GAS_C2P,            // exact conserved-energy change made by C2P/floors/excision
  ED_DS_ADVECT,           // RK-integrated conservative entropy-flux contribution
  ED_QENT_RAD,            // comoving radiation-to-gas energy transfer
  ED_QENT_TOTAL,          // total nonadiabatic heating from the discrete entropy residual
  ED_QENT_NUM,            // entropy heating minus radiation heating (repairs are flagged)
  ED_GAS_ACTUAL,         // measured gas conserved-energy change over the timestep
  ED_GAS_CLOSURE,        // actual minus independently recorded gas terms
  ED_RAD_ACTUAL,         // measured coordinate radiation-energy change over the timestep
  ED_RAD_CLOSURE,        // actual minus independently recorded radiation terms
  ED_SHOCK_SENSOR,       // dimensionless compressive morphology score
  ED_CURRENT_SENSOR,     // dimensionless current-sheet morphology score
  ED_CONTACT_SENSOR,     // dimensionless density/contact morphology score
  ED_VORT_SENSOR,        // dimensionless vorticity score
  ED_SHEAR_SENSOR,       // dimensionless traceless-strain score
  ED_FLAGS,              // bit mask converted to Real for binary output
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
  EDF_RAD_BAD       = 1u << 8
};

class EnergyDiagnostics {
 public:
  EnergyDiagnostics(MeshBlockPack *ppack, ParameterInput *pin);
  ~EnergyDiagnostics() = default;

  static const char *label[NENERGY_DIAG];

  bool recording = false;
  DvceArray5D<Real> step;       // RK-consistent increments for the current timestep
  DvceArray5D<Real> output;     // rates/sensors from the last completed timestep
  DvceArray5D<Real> gas_initial;
  DvceArray5D<Real> rad_initial;
  DvceArray5D<Real> gas_scratch;
  DvceArray5D<Real> rad_scratch;
  DvceArray5D<Real> entropy_initial; // D*s at the beginning of the timestep
  DvceArray5D<Real> rad_energy_flux;  // R^i_t, i=1..3, used for luminosity histories
  DvceFaceFld4D<Real> hlle_energy_flux;
  DvceFaceFld4D<Real> fofc_energy_flux;
  DvceFaceFld4D<Real> entropy_flux;   // passive entropy flux used only by diagnostics
  DvceArray4D<unsigned int> flags;

  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus BeginTimestep(Driver *pdriver, int stage);
  TaskStatus BeginStage(Driver *pdriver, int stage);
  TaskStatus FinalizeTimestep(Driver *pdriver, int stage);

  void SaveGasEnergy();
  void AccumulateGasEnergy(int channel);
  void SaveRadiationEnergy();
  void AccumulateRadiationEnergy(int channel);
  void RecordMHDFluxUpdate(Driver *pdriver, int stage);
  void RecordRadiationUpdate(Driver *pdriver, int stage);

 private:
  MeshBlockPack *pmy_pack_;
  Real sample_dt_;
  Real next_sample_time_;
  int sample_number_ = 0;
};

} // namespace diagnostics
#endif // DIAGNOSTICS_ENERGY_DIAGNOSTICS_HPP_
