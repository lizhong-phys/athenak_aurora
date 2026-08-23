//========================================================================================
//! \file energy_diagnostics.cpp
//! \brief Passive, runtime-gated radiation-GRMHD energy ledger.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "diagnostics/energy_diagnostics.hpp"
#include "globals.hpp"

namespace diagnostics {

namespace {
constexpr const char *kEnergyDiagImplementation = "physics-first-passive-v8";

enum PhysicalStateIndex {
  PS_RHO=0, PS_EINT, PS_PRESSURE, PS_ENTROPY,
  PS_UT, PS_U1, PS_U2, PS_U3,
  PS_UC0, PS_UC1, PS_UC2, PS_UC3,
  PS_B0, PS_B1, PS_B2, PS_B3, PS_BSQ,
  PS_EUT, PS_EU1, PS_EU2, PS_EU3,
  NPHYSICAL_STATE
};
}

const char *EnergyDiagnostics::label[NENERGY_DIAG] = {
  "dE_gas_flux", "dE_gas_hlle", "dE_gas_fofc", "dE_gas_coord",
  "dE_gas_other", "dE_gas_rad", "dE_gas_abs", "dE_gas_compt",
  "dE_rad_reject", "dE_rad_spatial", "dE_rad_angular", "dE_rad_fix",
  "dE_rad_source", "dE_gas_c2p", "dS_advect", "dEint_advect", "dU_advect",
  "dPcomp_spatial", "qent_rad_legacy", "qent_rad",
  "qent_total", "qent_centered", "qent_num",
  "dE_gas_actual", "dE_gas_closure", "dE_rad_actual", "dE_rad_closure",
  "q_storage", "q_advection", "q_compression", "q_diss_energy",
  "q_thermo_closure", "expansion", "q_advection_centered",
  "q_compression_centered", "expansion_centered", "q_mag_loss", "q_hydro_diss",
  "shock_sensor", "current_sensor", "contact_sensor", "vort_sensor",
  "shear_sensor", "pressure_jump", "field_reversal",
  "rho", "eint", "pressure", "ut", "u1", "u2", "u3", "B1", "B2", "B3", "bsq",
  "repair_flags"
};

EnergyDiagnostics::EnergyDiagnostics(MeshBlockPack *ppack, ParameterInput *pin) :
    step("energy_diag_step",1,1,1,1,1),
    output("energy_diag_output",1,1,1,1,1),
    gas_initial("energy_diag_gas_initial",1,1,1,1,1),
    rad_initial("energy_diag_rad_initial",1,1,1,1,1),
    gas_scratch("energy_diag_gas_scratch",1,1,1,1,1),
    rad_scratch("energy_diag_rad_scratch",1,1,1,1,1),
    entropy_initial("energy_diag_entropy_initial",1,1,1,1,1),
    physical_initial("energy_diag_physical_initial",1,1,1,1,1),
    physical_final("energy_diag_physical_final",1,1,1,1,1),
    gas_four_scratch("energy_diag_gas_four_scratch",1,1,1,1,1),
    ucon_scratch("energy_diag_ucon_scratch",1,1,1,1,1),
    rad_energy_flux("energy_diag_rad_flux",1,1,1,1,1),
    hlle_energy_flux("energy_diag_hlle_flux",1,1,1,1),
    fofc_energy_flux("energy_diag_fofc_flux",1,1,1,1),
    entropy_flux("energy_diag_entropy_flux",1,1,1,1),
    internal_energy_flux("energy_diag_internal_energy_flux",1,1,1,1),
    four_velocity_flux("energy_diag_four_velocity_flux",1,1,1,1),
    flags("energy_diag_flags",1,1,1,1),
    pmy_pack_(ppack),
    sample_dt_(pin->GetOrAddReal("problem","energy_diagnostics_dt",1.0)),
    next_sample_time_(pin->GetOrAddReal("problem","energy_diagnostics_start",
                                        ppack->pmesh->time)+sample_dt_) {
  if (ppack->pmhd == nullptr || ppack->prad == nullptr) {
    std::cout << "### FATAL ERROR: energy diagnostics currently require radiation+MHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(sample_dt_ > 0.0)) {
    std::cout << "### FATAL ERROR: <problem>/energy_diagnostics_dt must be positive"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "ENERGY_DIAG implementation=" << kEnergyDiagImplementation
              << " passive=true start=" << (next_sample_time_-sample_dt_)
              << " cadence=" << sample_dt_ << std::endl;
  }

  const int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;

  Kokkos::realloc(step,nmb,NENERGY_DIAG,n3,n2,n1);
  Kokkos::realloc(output,nmb,NENERGY_DIAG,n3,n2,n1);
  Kokkos::realloc(gas_initial,nmb,1,n3,n2,n1);
  Kokkos::realloc(rad_initial,nmb,1,n3,n2,n1);
  Kokkos::realloc(gas_scratch,nmb,1,n3,n2,n1);
  Kokkos::realloc(rad_scratch,nmb,1,n3,n2,n1);
  Kokkos::realloc(entropy_initial,nmb,1,n3,n2,n1);
  Kokkos::realloc(physical_initial,nmb,NPHYSICAL_STATE,n3,n2,n1);
  Kokkos::realloc(physical_final,nmb,NPHYSICAL_STATE,n3,n2,n1);
  Kokkos::realloc(gas_four_scratch,nmb,4,n3,n2,n1);
  Kokkos::realloc(ucon_scratch,nmb,4,n3,n2,n1);
  Kokkos::realloc(rad_energy_flux,nmb,3,n3,n2,n1);
  Kokkos::realloc(hlle_energy_flux.x1f,nmb,n3,n2,n1+1);
  Kokkos::realloc(hlle_energy_flux.x2f,nmb,n3,n2+1,n1);
  Kokkos::realloc(hlle_energy_flux.x3f,nmb,n3+1,n2,n1);
  Kokkos::realloc(fofc_energy_flux.x1f,nmb,n3,n2,n1+1);
  Kokkos::realloc(fofc_energy_flux.x2f,nmb,n3,n2+1,n1);
  Kokkos::realloc(fofc_energy_flux.x3f,nmb,n3+1,n2,n1);
  Kokkos::realloc(entropy_flux.x1f,nmb,n3,n2,n1+1);
  Kokkos::realloc(entropy_flux.x2f,nmb,n3,n2+1,n1);
  Kokkos::realloc(entropy_flux.x3f,nmb,n3+1,n2,n1);
  Kokkos::realloc(internal_energy_flux.x1f,nmb,n3,n2,n1+1);
  Kokkos::realloc(internal_energy_flux.x2f,nmb,n3,n2+1,n1);
  Kokkos::realloc(internal_energy_flux.x3f,nmb,n3+1,n2,n1);
  Kokkos::realloc(four_velocity_flux.x1f,nmb,n3,n2,n1+1);
  Kokkos::realloc(four_velocity_flux.x2f,nmb,n3,n2+1,n1);
  Kokkos::realloc(four_velocity_flux.x3f,nmb,n3+1,n2,n1);
  Kokkos::realloc(flags,nmb,n3,n2,n1);

  Kokkos::deep_copy(step,0.0);
  Kokkos::deep_copy(output,0.0);
  Kokkos::deep_copy(flags,0u);
}

void EnergyDiagnostics::AssembleTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  tl["before_timeintegrator"]->AddTask(&EnergyDiagnostics::BeginTimestep,this,none);

  // The diagnostic RK recurrence must run before any stage operator.  Register it in the
  // "before_stagen" list, which the driver runs to COMPLETION before "stagen" on every RK
  // stage (driver.cpp: before_stagen -> stagen -> after_stagen), so BeginStage precedes
  // Radiation::CopyCons and all stage operators.
  //
  // NOTE: do NOT use tl["stagen"]->InsertTask(..., copyu) here.  copyu is added with an
  // empty (`none`) dependency, and InsertTask reads old_dep = copyu's dependency = none,
  // then calls ChangeDependency(none, new_id) on every other task.  ChangeDependency tests
  // `((dep_ & id) == id)`, which is ALWAYS true for id == none, so EVERY task in the hot
  // per-stage list (including all MPI Send/Recv/boundary tasks) gets this task OR'd into
  // its dependency -- a corrupted graph that stalls multi-rank runs.  AddTask into
  // before_stagen achieves the required ordering without touching any other dependency.
  tl["before_stagen"]->AddTask(&EnergyDiagnostics::BeginStage,this,none);
  tl["after_timeintegrator"]->AddTask(&EnergyDiagnostics::FinalizeTimestep,this,none);
}

// Fill a detached state used by the physical internal- and magnetic-energy equations.
// This kernel is read-only with respect to every evolved array.  Cartesian Kerr-Schild
// has sqrt(-g)=1, so no metric determinant is required in the stored currents.
void EnergyDiagnostics::FillPhysicalState(DvceArray5D<Real> &state) {
  if (!recording) return;
  auto w = pmy_pack_->pmhd->w0;
  auto b = pmy_pack_->pmhd->bcc0;
  auto qstate = state;
  auto size = pmy_pack_->pmb->mb_size;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int is=indcs.is, js=indcs.js, ks=indcs.ks;
  const Real gamma = pmy_pack_->pmhd->peos->eos_data.gamma;
  const bool flat = pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin = pmy_pack_->pcoord->coord_data.bh_spin;
  par_for("energy_diag_physical_state",DevExeSpace(),0,nmb1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1=CellCenterX(i-is,indcs.nx1,size.d_view(m).x1min,size.d_view(m).x1max);
    const Real x2=CellCenterX(j-js,indcs.nx2,size.d_view(m).x2min,size.d_view(m).x2max);
    const Real x3=CellCenterX(k-ks,indcs.nx3,size.d_view(m).x3min,size.d_view(m).x3max);
    Real gl[4][4], gu[4][4];
    ComputeMetricAndInverse(x1,x2,x3,flat,spin,gl,gu);
    const Real rho=fmax(w(m,IDN,k,j,i),1.0e-300);
    const Real eint=fmax(w(m,IEN,k,j,i),1.0e-300);
    const Real press=fmax((gamma-1.0)*eint,1.0e-300);
    const Real v1=w(m,IVX,k,j,i), v2=w(m,IVY,k,j,i), v3=w(m,IVZ,k,j,i);
    const Real usq=gl[1][1]*v1*v1+gl[2][2]*v2*v2+gl[3][3]*v3*v3
                  +2.0*(gl[1][2]*v1*v2+gl[1][3]*v1*v3+gl[2][3]*v2*v3);
    const Real alpha=sqrt(-1.0/gu[0][0]);
    const Real lor=sqrt(1.0+fmax(usq,0.0));
    Real uc[4];
    uc[0]=lor/alpha;
    uc[1]=v1-alpha*lor*gu[0][1];
    uc[2]=v2-alpha*lor*gu[0][2];
    uc[3]=v3-alpha*lor*gu[0][3];
    Real ul[4];
    for (int a=0; a<4; ++a) {
      ul[a]=0.0;
      for (int c=0; c<4; ++c) ul[a]+=gl[a][c]*uc[c];
    }
    const Real B1=b(m,IBX,k,j,i), B2=b(m,IBY,k,j,i), B3=b(m,IBZ,k,j,i);
    Real bc[4];
    bc[0]=ul[1]*B1+ul[2]*B2+ul[3]*B3;
    bc[1]=(B1+bc[0]*uc[1])/uc[0];
    bc[2]=(B2+bc[0]*uc[2])/uc[0];
    bc[3]=(B3+bc[0]*uc[3])/uc[0];
    Real bsq=0.0;
    for (int a=0; a<4; ++a) {
      Real bl=0.0;
      for (int c=0; c<4; ++c) bl+=gl[a][c]*bc[c];
      bsq+=bl*bc[a];
    }
    qstate(m,PS_RHO,k,j,i)=rho;
    qstate(m,PS_EINT,k,j,i)=eint;
    qstate(m,PS_PRESSURE,k,j,i)=press;
    qstate(m,PS_ENTROPY,k,j,i)=(log(press)-gamma*log(rho))/(gamma-1.0);
    qstate(m,PS_UT,k,j,i)=uc[0];
    qstate(m,PS_U1,k,j,i)=uc[1]; qstate(m,PS_U2,k,j,i)=uc[2];
    qstate(m,PS_U3,k,j,i)=uc[3];
    qstate(m,PS_UC0,k,j,i)=ul[0]; qstate(m,PS_UC1,k,j,i)=ul[1];
    qstate(m,PS_UC2,k,j,i)=ul[2]; qstate(m,PS_UC3,k,j,i)=ul[3];
    qstate(m,PS_B0,k,j,i)=bc[0]; qstate(m,PS_B1,k,j,i)=bc[1];
    qstate(m,PS_B2,k,j,i)=bc[2]; qstate(m,PS_B3,k,j,i)=bc[3];
    qstate(m,PS_BSQ,k,j,i)=bsq;
    qstate(m,PS_EUT,k,j,i)=eint*uc[0];
    qstate(m,PS_EU1,k,j,i)=eint*uc[1];
    qstate(m,PS_EU2,k,j,i)=eint*uc[2];
    qstate(m,PS_EU3,k,j,i)=eint*uc[3];
  });
}

TaskStatus EnergyDiagnostics::BeginTimestep(Driver *pdriver, int stage) {
  // Match Driver's 32-bit output-time comparison.  The ledger is intentionally active
  // only on a step whose end time crosses a requested diagnostic output time.
  const Real end_time = pmy_pack_->pmesh->time + pmy_pack_->pmesh->dt;
  recording = (static_cast<float>(end_time) >=
               static_cast<float>(next_sample_time_));
  if (!recording) return TaskStatus::complete;
  if (global_variable::my_rank == 0) {
    std::cout << "ENERGY_DIAG sample_begin=" << (sample_number_+1)
              << " step_time=[" << pmy_pack_->pmesh->time << "," << end_time
              << "] target=" << next_sample_time_ << std::endl;
  }
  auto gas = pmy_pack_->pmhd->u0;
  auto prim = pmy_pack_->pmhd->w0;
  auto initial = gas_initial;
  auto sinitial = entropy_initial;
  const Real gamma = pmy_pack_->pmhd->peos->eos_data.gamma;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_begin_gas",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    initial(m,0,k,j,i) = gas(m,IEN,k,j,i);
    const Real rho=fmax(prim(m,IDN,k,j,i),1.0e-300);
    const Real p=fmax((gamma-1.0)*prim(m,IEN,k,j,i),1.0e-300);
    const Real s=(log(p)-gamma*log(rho))/(gamma-1.0);
    sinitial(m,0,k,j,i)=gas(m,IDN,k,j,i)*s;
  });

  FillPhysicalState(physical_initial);

  SaveRadiationEnergy();
  Kokkos::deep_copy(DevExeSpace(),rad_initial,rad_scratch);
  return TaskStatus::complete;
}

void EnergyDiagnostics::SaveRadiationCouplingState() {
  if (!recording) return;
  auto u=pmy_pack_->pmhd->u0;
  auto before=gas_four_scratch;
  auto usave=ucon_scratch;
  auto qstate=physical_final;
  // Refresh four-velocity from the primitives immediately before this source operator.
  FillPhysicalState(physical_final);
  auto &indcs=pmy_pack_->pmesh->mb_indcs;
  const int nmb1=pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_rad_before",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    before(m,0,k,j,i)=u(m,IEN,k,j,i);
    before(m,1,k,j,i)=u(m,IM1,k,j,i);
    before(m,2,k,j,i)=u(m,IM2,k,j,i);
    before(m,3,k,j,i)=u(m,IM3,k,j,i);
    usave(m,0,k,j,i)=qstate(m,PS_UT,k,j,i);
    usave(m,1,k,j,i)=qstate(m,PS_U1,k,j,i);
    usave(m,2,k,j,i)=qstate(m,PS_U2,k,j,i);
    usave(m,3,k,j,i)=qstate(m,PS_U3,k,j,i);
  });
  SaveGasEnergy();
  SaveRadiationEnergy();
}

void EnergyDiagnostics::RecordRadiationCoupling() {
  if (!recording) return;
  auto u=pmy_pack_->pmhd->u0;
  auto before=gas_four_scratch;
  auto usave=ucon_scratch;
  auto d=step;
  auto &indcs=pmy_pack_->pmesh->mb_indcs;
  const int nmb1=pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_rad_after",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    const Real de=u(m,IEN,k,j,i)-before(m,0,k,j,i);
    const Real dm1=u(m,IM1,k,j,i)-before(m,1,k,j,i);
    const Real dm2=u(m,IM2,k,j,i)-before(m,2,k,j,i);
    const Real dm3=u(m,IM3,k,j,i)-before(m,3,k,j,i);
    // The GRMHD conserved components are T^t_mu (covariant second index):
    // IEN=T^t_t+D and IMi=T^t_i.  Therefore u^mu Delta(T^t_mu) has PLUS signs
    // for all four components.  The previous minus-spatial contraction is kept as
    // a detached QA channel so the sign correction is directly auditable.
    d(m,ED_QENT_RAD_LEGACY,k,j,i)+=usave(m,0,k,j,i)*de-usave(m,1,k,j,i)*dm1
                                   -usave(m,2,k,j,i)*dm2-usave(m,3,k,j,i)*dm3;
    d(m,ED_QENT_RAD,k,j,i)+=usave(m,0,k,j,i)*de+usave(m,1,k,j,i)*dm1
                            +usave(m,2,k,j,i)*dm2+usave(m,3,k,j,i)*dm3;
  });
  AccumulateGasEnergy(ED_GAS_RAD_TOTAL);
  AccumulateRadiationEnergy(ED_RAD_SOURCE_TOTAL);
}

TaskStatus EnergyDiagnostics::BeginStage(Driver *pdriver, int stage) {
  if (!recording) return TaskStatus::complete;
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(),step,0.0);
    Kokkos::deep_copy(DevExeSpace(),flags,0u);
  } else {
    const Real gam0 = pdriver->gam0[stage-1];
    auto d = step;
    auto &indcs = pmy_pack_->pmesh->mb_indcs;
    const int nmb1 = pmy_pack_->nmb_thispack-1;
    par_for("energy_diag_rk_scale",DevExeSpace(),0,nmb1,0,ED_QENT_RAD,
            indcs.ks,indcs.ke,indcs.js,indcs.je,indcs.is,indcs.ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      d(m,n,k,j,i) *= gam0;
    });
  }
  Kokkos::deep_copy(DevExeSpace(),hlle_energy_flux.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),hlle_energy_flux.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),hlle_energy_flux.x3f,0.0);
  Kokkos::deep_copy(DevExeSpace(),fofc_energy_flux.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),fofc_energy_flux.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),fofc_energy_flux.x3f,0.0);
  Kokkos::deep_copy(DevExeSpace(),entropy_flux.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),entropy_flux.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),entropy_flux.x3f,0.0);
  Kokkos::deep_copy(DevExeSpace(),internal_energy_flux.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),internal_energy_flux.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),internal_energy_flux.x3f,0.0);
  Kokkos::deep_copy(DevExeSpace(),four_velocity_flux.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),four_velocity_flux.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),four_velocity_flux.x3f,0.0);
  return TaskStatus::complete;
}

void EnergyDiagnostics::SaveGasEnergy() {
  if (!recording) return;
  auto gas = pmy_pack_->pmhd->u0;
  auto scratch = gas_scratch;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_save_gas",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    scratch(m,0,k,j,i) = gas(m,IEN,k,j,i);
  });
}

void EnergyDiagnostics::AccumulateGasEnergy(int channel) {
  if (!recording) return;
  auto gas = pmy_pack_->pmhd->u0;
  auto scratch = gas_scratch;
  auto d = step;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_accum_gas",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    d(m,channel,k,j,i) += gas(m,IEN,k,j,i)-scratch(m,0,k,j,i);
  });
}

void EnergyDiagnostics::SaveRadiationEnergy() {
  if (!recording) return;
  auto prad = pmy_pack_->prad;
  auto i0 = prad->i0;
  auto omega = prad->prgeo->solid_angles;
  const int nang = prad->prgeo->nangles;
  auto scratch = rad_scratch;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_save_rad",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real e = 0.0;
    for (int n=0; n<nang; ++n) e += i0(m,n,k,j,i)*omega.d_view(n);
    scratch(m,0,k,j,i) = e;
  });
}

void EnergyDiagnostics::AccumulateRadiationEnergy(int channel) {
  if (!recording) return;
  auto prad = pmy_pack_->prad;
  auto i0 = prad->i0;
  auto omega = prad->prgeo->solid_angles;
  const int nang = prad->prgeo->nangles;
  auto scratch = rad_scratch;
  auto d = step;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("energy_diag_accum_rad",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real e = 0.0;
    for (int n=0; n<nang; ++n) e += i0(m,n,k,j,i)*omega.d_view(n);
    d(m,channel,k,j,i) += e-scratch(m,0,k,j,i);
  });
}

void EnergyDiagnostics::RecordMHDFluxUpdate(Driver *pdriver, int stage) {
  if (!recording) return;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const Real beta_dt = pdriver->beta[stage-1]*pmy_pack_->pmesh->dt;
  auto flx = pmy_pack_->pmhd->uflx;
  auto hll = hlle_energy_flux;
  auto fof = fofc_energy_flux;
  auto sflx = entropy_flux;
  auto eflx = internal_energy_flux;
  auto uflx = four_velocity_flux;
  auto size = pmy_pack_->pmb->mb_size;
  auto w = pmy_pack_->pmhd->w0;
  const Real gm1 = pmy_pack_->pmhd->peos->eos_data.gamma-1.0;
  auto d = step;
  par_for("energy_diag_mhd_flux",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real total = (flx.x1f(m,IEN,k,j,i+1)-flx.x1f(m,IEN,k,j,i))/size.d_view(m).dx1;
    Real dhll = (hll.x1f(m,k,j,i+1)-hll.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real dfof = (fof.x1f(m,k,j,i+1)-fof.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real divs = (sflx.x1f(m,k,j,i+1)-sflx.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real dive = (eflx.x1f(m,k,j,i+1)-eflx.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real divu = (uflx.x1f(m,k,j,i+1)-uflx.x1f(m,k,j,i))/size.d_view(m).dx1;
    if (multi_d) {
      total += (flx.x2f(m,IEN,k,j+1,i)-flx.x2f(m,IEN,k,j,i))/size.d_view(m).dx2;
      dhll += (hll.x2f(m,k,j+1,i)-hll.x2f(m,k,j,i))/size.d_view(m).dx2;
      dfof += (fof.x2f(m,k,j+1,i)-fof.x2f(m,k,j,i))/size.d_view(m).dx2;
      divs += (sflx.x2f(m,k,j+1,i)-sflx.x2f(m,k,j,i))/size.d_view(m).dx2;
      dive += (eflx.x2f(m,k,j+1,i)-eflx.x2f(m,k,j,i))/size.d_view(m).dx2;
      divu += (uflx.x2f(m,k,j+1,i)-uflx.x2f(m,k,j,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      total += (flx.x3f(m,IEN,k+1,j,i)-flx.x3f(m,IEN,k,j,i))/size.d_view(m).dx3;
      dhll += (hll.x3f(m,k+1,j,i)-hll.x3f(m,k,j,i))/size.d_view(m).dx3;
      dfof += (fof.x3f(m,k+1,j,i)-fof.x3f(m,k,j,i))/size.d_view(m).dx3;
      divs += (sflx.x3f(m,k+1,j,i)-sflx.x3f(m,k,j,i))/size.d_view(m).dx3;
      dive += (eflx.x3f(m,k+1,j,i)-eflx.x3f(m,k,j,i))/size.d_view(m).dx3;
      divu += (uflx.x3f(m,k+1,j,i)-uflx.x3f(m,k,j,i))/size.d_view(m).dx3;
    }
    d(m,ED_GAS_FLUX,k,j,i) += -beta_dt*total;
    d(m,ED_GAS_HLLE,k,j,i) += -beta_dt*dhll;
    d(m,ED_GAS_FOFC,k,j,i) += -beta_dt*dfof;
    d(m,ED_DS_ADVECT,k,j,i) += -beta_dt*divs;
    d(m,ED_DEINT_ADVECT,k,j,i) += -beta_dt*dive;
    d(m,ED_DU_ADVECT,k,j,i) += -beta_dt*divu;
    d(m,ED_DPCOMP_SPATIAL,k,j,i) += -beta_dt*gm1*w(m,IEN,k,j,i)*divu;
  });
}

void EnergyDiagnostics::RecordRadiationUpdate(Driver *pdriver, int stage) {
  if (!recording) return;
  auto prad = pmy_pack_->prad;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const bool angular = prad->angular_fluxes;
  const Real gam0 = pdriver->gam0[stage-1];
  const Real gam1 = pdriver->gam1[stage-1];
  const Real beta_dt = pdriver->beta[stage-1]*pmy_pack_->pmesh->dt;
  const int nang = prad->prgeo->nangles;
  auto i0 = prad->i0;
  auto i1 = prad->i1;
  auto flx = prad->iflx;
  auto divfa = prad->divfa;
  auto omega = prad->prgeo->solid_angles;
  auto size = pmy_pack_->pmb->mb_size;
  auto before = rad_scratch;
  auto d = step;
  auto flg = flags;
  par_for("energy_diag_rad_update",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real e_after=0.0, e_i1=0.0, spatial=0.0, angular_inc=0.0;
    for (int n=0; n<nang; ++n) {
      const Real om = omega.d_view(n);
      e_after += i0(m,n,k,j,i)*om;
      e_i1 += i1(m,n,k,j,i)*om;
      Real div = (flx.x1f(m,n,k,j,i+1)-flx.x1f(m,n,k,j,i))/size.d_view(m).dx1;
      if (multi_d) div += (flx.x2f(m,n,k,j+1,i)-flx.x2f(m,n,k,j,i))/size.d_view(m).dx2;
      if (three_d) div += (flx.x3f(m,n,k+1,j,i)-flx.x3f(m,n,k,j,i))/size.d_view(m).dx3;
      spatial += -beta_dt*div*om;
      if (angular) angular_inc += -beta_dt*divfa(m,n,k,j,i)*om;
    }
    const Real total = e_after-gam0*before(m,0,k,j,i)-gam1*e_i1;
    d(m,ED_RAD_SPATIAL,k,j,i) += spatial;
    d(m,ED_RAD_ANGULAR,k,j,i) += angular_inc;
    const Real fix=total-spatial-angular_inc;
    d(m,ED_RAD_FIX,k,j,i) += fix;
    if (fabs(fix) > 1.0e-12*(fabs(total)+fabs(spatial)+fabs(angular_inc)+1.0e-30)) {
      flg(m,k,j,i) |= EDF_RAD_FIX;
    }
  });
}

TaskStatus EnergyDiagnostics::FinalizeTimestep(Driver *pdriver, int stage) {
  if (!recording) return TaskStatus::complete;
  FillPhysicalState(physical_final);
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const Real dt = pmy_pack_->pmesh->dt;
  const Real gamma = pmy_pack_->pmhd->peos->eos_data.gamma;
  auto u = pmy_pack_->pmhd->u0;
  auto w = pmy_pack_->pmhd->w0;
  auto b = pmy_pack_->pmhd->bcc0;
  auto i0 = pmy_pack_->prad->i0;
  auto omega = pmy_pack_->prad->prgeo->solid_angles;
  auto nh = pmy_pack_->prad->nh_c;
  auto tet = pmy_pack_->prad->tet_c;
  const int nang = pmy_pack_->prad->prgeo->nangles;
  auto size = pmy_pack_->pmb->mb_size;
  auto initial_g = gas_initial;
  auto initial_r = rad_initial;
  auto initial_s = entropy_initial;
  auto phys0 = physical_initial;
  auto phys1 = physical_final;
  auto d = step;
  auto out = output;
  auto rflux = rad_energy_flux;
  auto flg = flags;
  const bool flat = pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin = pmy_pack_->pcoord->coord_data.bh_spin;

  par_for("energy_diag_finalize",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real erad=0.0, rf1=0.0, rf2=0.0, rf3=0.0;
    const Real n0 = tet(m,0,0,k,j,i);
    for (int n=0; n<nang; ++n) {
      Real nc1=0.0,nc2=0.0,nc3=0.0;
      for (int a=0; a<4; ++a) {
        nc1 += tet(m,a,1,k,j,i)*nh.d_view(n,a);
        nc2 += tet(m,a,2,k,j,i)*nh.d_view(n,a);
        nc3 += tet(m,a,3,k,j,i)*nh.d_view(n,a);
      }
      const Real iom = i0(m,n,k,j,i)*omega.d_view(n);
      erad += iom;
      rf1 += nc1*iom/n0;
      rf2 += nc2*iom/n0;
      rf3 += nc3*iom/n0;
    }
    rflux(m,0,k,j,i)=rf1;
    rflux(m,1,k,j,i)=rf2;
    rflux(m,2,k,j,i)=rf3;

    const Real gas_actual = u(m,IEN,k,j,i)-initial_g(m,0,k,j,i);
    const Real gas_sum = d(m,ED_GAS_FLUX,k,j,i)+d(m,ED_GAS_COORD,k,j,i)
                       + d(m,ED_GAS_OTHER,k,j,i)+d(m,ED_GAS_RAD_TOTAL,k,j,i)
                       + d(m,ED_GAS_C2P,k,j,i);
    const Real rad_actual = erad-initial_r(m,0,k,j,i);
    const Real rad_sum = d(m,ED_RAD_SPATIAL,k,j,i)+d(m,ED_RAD_ANGULAR,k,j,i)
                       + d(m,ED_RAD_FIX,k,j,i)+d(m,ED_RAD_SOURCE_TOTAL,k,j,i);

    for (int n=0; n<=ED_QENT_RAD; ++n) out(m,n,k,j,i)=d(m,n,k,j,i)/dt;

    // Physical internal-energy and entropy-current budget.  Start/end currents are
    // centered over the actual solver timestep, which is much shorter than the output
    // cadence.  This avoids differencing widely separated full dumps.
    const Real dx1=size.d_view(m).dx1, dx2=size.d_view(m).dx2, dx3=size.d_view(m).dx3;
    Real div_e0=(phys0(m,PS_EU1,k,j,i+1)-phys0(m,PS_EU1,k,j,i-1))/(2.0*dx1);
    Real div_e1=(phys1(m,PS_EU1,k,j,i+1)-phys1(m,PS_EU1,k,j,i-1))/(2.0*dx1);
    Real div_u0=(phys0(m,PS_U1,k,j,i+1)-phys0(m,PS_U1,k,j,i-1))/(2.0*dx1);
    Real div_u1=(phys1(m,PS_U1,k,j,i+1)-phys1(m,PS_U1,k,j,i-1))/(2.0*dx1);
    Real div_s0=(phys0(m,PS_RHO,k,j,i+1)*phys0(m,PS_ENTROPY,k,j,i+1)
                    *phys0(m,PS_U1,k,j,i+1)
                 -phys0(m,PS_RHO,k,j,i-1)*phys0(m,PS_ENTROPY,k,j,i-1)
                    *phys0(m,PS_U1,k,j,i-1))/(2.0*dx1);
    Real div_s1=(phys1(m,PS_RHO,k,j,i+1)*phys1(m,PS_ENTROPY,k,j,i+1)
                    *phys1(m,PS_U1,k,j,i+1)
                 -phys1(m,PS_RHO,k,j,i-1)*phys1(m,PS_ENTROPY,k,j,i-1)
                    *phys1(m,PS_U1,k,j,i-1))/(2.0*dx1);
    if (multi_d) {
      div_e0+=(phys0(m,PS_EU2,k,j+1,i)-phys0(m,PS_EU2,k,j-1,i))/(2.0*dx2);
      div_e1+=(phys1(m,PS_EU2,k,j+1,i)-phys1(m,PS_EU2,k,j-1,i))/(2.0*dx2);
      div_u0+=(phys0(m,PS_U2,k,j+1,i)-phys0(m,PS_U2,k,j-1,i))/(2.0*dx2);
      div_u1+=(phys1(m,PS_U2,k,j+1,i)-phys1(m,PS_U2,k,j-1,i))/(2.0*dx2);
      div_s0+=(phys0(m,PS_RHO,k,j+1,i)*phys0(m,PS_ENTROPY,k,j+1,i)
                  *phys0(m,PS_U2,k,j+1,i)
               -phys0(m,PS_RHO,k,j-1,i)*phys0(m,PS_ENTROPY,k,j-1,i)
                  *phys0(m,PS_U2,k,j-1,i))/(2.0*dx2);
      div_s1+=(phys1(m,PS_RHO,k,j+1,i)*phys1(m,PS_ENTROPY,k,j+1,i)
                  *phys1(m,PS_U2,k,j+1,i)
               -phys1(m,PS_RHO,k,j-1,i)*phys1(m,PS_ENTROPY,k,j-1,i)
                  *phys1(m,PS_U2,k,j-1,i))/(2.0*dx2);
    }
    if (three_d) {
      div_e0+=(phys0(m,PS_EU3,k+1,j,i)-phys0(m,PS_EU3,k-1,j,i))/(2.0*dx3);
      div_e1+=(phys1(m,PS_EU3,k+1,j,i)-phys1(m,PS_EU3,k-1,j,i))/(2.0*dx3);
      div_u0+=(phys0(m,PS_U3,k+1,j,i)-phys0(m,PS_U3,k-1,j,i))/(2.0*dx3);
      div_u1+=(phys1(m,PS_U3,k+1,j,i)-phys1(m,PS_U3,k-1,j,i))/(2.0*dx3);
      div_s0+=(phys0(m,PS_RHO,k+1,j,i)*phys0(m,PS_ENTROPY,k+1,j,i)
                  *phys0(m,PS_U3,k+1,j,i)
               -phys0(m,PS_RHO,k-1,j,i)*phys0(m,PS_ENTROPY,k-1,j,i)
                  *phys0(m,PS_U3,k-1,j,i))/(2.0*dx3);
      div_s1+=(phys1(m,PS_RHO,k+1,j,i)*phys1(m,PS_ENTROPY,k+1,j,i)
                  *phys1(m,PS_U3,k+1,j,i)
               -phys1(m,PS_RHO,k-1,j,i)*phys1(m,PS_ENTROPY,k-1,j,i)
                  *phys1(m,PS_U3,k-1,j,i))/(2.0*dx3);
    }
    const Real p0=phys0(m,PS_PRESSURE,k,j,i), p1=phys1(m,PS_PRESSURE,k,j,i);
    const Real rho0=phys0(m,PS_RHO,k,j,i), rho1=phys1(m,PS_RHO,k,j,i);
    const Real dut=(phys1(m,PS_UT,k,j,i)-phys0(m,PS_UT,k,j,i))/dt;
    const Real expansion_centered=dut+0.5*(div_u0+div_u1);
    const Real qstore=(phys1(m,PS_EUT,k,j,i)-phys0(m,PS_EUT,k,j,i))/dt;
    const Real qadv_centered=0.5*(div_e0+div_e1);
    const Real qcomp_centered=-0.5*(p0+p1)*dut-0.5*(p0*div_u0+p1*div_u1);
    // Primary finite-volume terms use the same reconstructed face states, HLL wave fan,
    // FOFC replacement, RK beta weights, and timestep as the production MHD update.
    // dEint_advect=-integral dt div(F_e), so its corresponding rate has a minus sign.
    const Real qadv=-d(m,ED_DEINT_ADVECT,k,j,i)/dt;
    const Real divu_face=-d(m,ED_DU_ADVECT,k,j,i)/dt;
    const Real expansion=dut+divu_face;
    // Spatial p dV work is accumulated stage-by-stage with the Riemann-face velocity.
    // The temporal part has no face flux and is endpoint-centered over this solver step.
    const Real qcomp=d(m,ED_DPCOMP_SPATIAL,k,j,i)/dt-0.5*(p0+p1)*dut;

    const Real s_final=phys1(m,PS_ENTROPY,k,j,i);
    const Real delta_ds=u(m,IDN,k,j,i)*s_final-initial_s(m,0,k,j,i);
    const Real temperature=0.5*(p0/rho0+p1/rho1);
    // dS_advect is the RK-weighted conservative change predicted by the actual face
    // entropy fluxes, including the final HLLE states and any FOFC replacement:
    //   dS_advect = -dt div(F_S).
    // Therefore actual minus advected entropy is delta(D s)-dS_advect.  This uses the
    // solver's discrete transport operator instead of an unrelated centered stencil.
    const Real qent_total=temperature*(delta_ds-d(m,ED_DS_ADVECT,k,j,i))/dt;
    const Real qent_centered=temperature*(delta_ds
                                         +0.5*dt*(div_s0+div_s1))/dt;
    out(m,ED_QENT_TOTAL,k,j,i)=qent_total;
    out(m,ED_QENT_CENTERED,k,j,i)=qent_centered;
    // ED_QENT_RAD is Lambda_{gas->radiation}: u.e evolves T^t_t+D, so a positive
    // conserved-energy increment corresponds to gas energy loss.  Recover irreversible
    // heating by adding that radiative loss back to the gas entropy change.
    out(m,ED_QENT_NUM,k,j,i)=qent_total+out(m,ED_QENT_RAD,k,j,i);
    out(m,ED_GAS_ACTUAL,k,j,i)=gas_actual/dt;
    out(m,ED_GAS_CLOSURE,k,j,i)=(gas_actual-gas_sum)/dt;
    out(m,ED_RAD_ACTUAL,k,j,i)=rad_actual/dt;
    out(m,ED_RAD_CLOSURE,k,j,i)=(rad_actual-rad_sum)/dt;

    const Real qdiss_e=qstore+qadv-qcomp+out(m,ED_QENT_RAD,k,j,i);
    out(m,ED_INT_STORAGE,k,j,i)=qstore;
    out(m,ED_INT_ADVECTION,k,j,i)=qadv;
    out(m,ED_COMPRESSION,k,j,i)=qcomp;
    out(m,ED_DISS_ENERGY,k,j,i)=qdiss_e;
    out(m,ED_THERMO_CLOSURE,k,j,i)=qdiss_e-out(m,ED_QENT_NUM,k,j,i);
    out(m,ED_EXPANSION,k,j,i)=expansion;
    out(m,ED_INT_ADVECTION_CENTERED,k,j,i)=qadv_centered;
    out(m,ED_COMPRESSION_CENTERED,k,j,i)=qcomp_centered;
    out(m,ED_EXPANSION_CENTERED,k,j,i)=expansion_centered;

    // Covariant magnetic-energy identity for ideal GRMHD:
    //   u.d(b^2/2) + b^2 theta - b^mu b^nu nabla_mu u_nu = 0.
    // A negative residual is effective non-ideal magnetic-energy loss.  All derivatives
    // are time-centered over this sampled timestep; the stationary CKS metric has det g=-1.
    Real dmag[4]={0.0,0.0,0.0,0.0};
    dmag[0]=0.5*(phys1(m,PS_BSQ,k,j,i)-phys0(m,PS_BSQ,k,j,i))/dt;
    dmag[1]=((phys0(m,PS_BSQ,k,j,i+1)-phys0(m,PS_BSQ,k,j,i-1))
            +(phys1(m,PS_BSQ,k,j,i+1)-phys1(m,PS_BSQ,k,j,i-1)))/(8.0*dx1);
    if (multi_d) {
      dmag[2]=((phys0(m,PS_BSQ,k,j+1,i)-phys0(m,PS_BSQ,k,j-1,i))
              +(phys1(m,PS_BSQ,k,j+1,i)-phys1(m,PS_BSQ,k,j-1,i)))/(8.0*dx2);
    }
    if (three_d) {
      dmag[3]=((phys0(m,PS_BSQ,k+1,j,i)-phys0(m,PS_BSQ,k-1,j,i))
              +(phys1(m,PS_BSQ,k+1,j,i)-phys1(m,PS_BSQ,k-1,j,i)))/(8.0*dx3);
    }
    Real uavg[4], bavg[4], ulavg[4];
    for (int a=0; a<4; ++a) {
      uavg[a]=0.5*(phys0(m,PS_UT+a,k,j,i)+phys1(m,PS_UT+a,k,j,i));
      bavg[a]=0.5*(phys0(m,PS_B0+a,k,j,i)+phys1(m,PS_B0+a,k,j,i));
      ulavg[a]=0.5*(phys0(m,PS_UC0+a,k,j,i)+phys1(m,PS_UC0+a,k,j,i));
    }
    Real ducov[4][4]={{0.0}};
    for (int a=0; a<4; ++a) {
      ducov[0][a]=(phys1(m,PS_UC0+a,k,j,i)-phys0(m,PS_UC0+a,k,j,i))/dt;
      ducov[1][a]=((phys0(m,PS_UC0+a,k,j,i+1)-phys0(m,PS_UC0+a,k,j,i-1))
                  +(phys1(m,PS_UC0+a,k,j,i+1)-phys1(m,PS_UC0+a,k,j,i-1)))/(4.0*dx1);
      if (multi_d) {
        ducov[2][a]=((phys0(m,PS_UC0+a,k,j+1,i)-phys0(m,PS_UC0+a,k,j-1,i))
                    +(phys1(m,PS_UC0+a,k,j+1,i)-phys1(m,PS_UC0+a,k,j-1,i)))/(4.0*dx2);
      }
      if (three_d) {
        ducov[3][a]=((phys0(m,PS_UC0+a,k+1,j,i)-phys0(m,PS_UC0+a,k-1,j,i))
                    +(phys1(m,PS_UC0+a,k+1,j,i)-phys1(m,PS_UC0+a,k-1,j,i)))/(4.0*dx3);
      }
    }
    const Real x1=CellCenterX(i-indcs.is,indcs.nx1,size.d_view(m).x1min,size.d_view(m).x1max);
    const Real x2=CellCenterX(j-indcs.js,indcs.nx2,size.d_view(m).x2min,size.d_view(m).x2max);
    const Real x3=CellCenterX(k-indcs.ks,indcs.nx3,size.d_view(m).x3min,size.d_view(m).x3max);
    Real gl[4][4], gu[4][4], dg1[4][4], dg2[4][4], dg3[4][4];
    ComputeMetricAndInverse(x1,x2,x3,flat,spin,gl,gu);
    ComputeMetricDerivatives(x1,x2,x3,flat,spin,dg1,dg2,dg3);
    Real dg[4][4][4]={{{0.0}}};
    for (int a=0; a<4; ++a) for (int c=0; c<4; ++c) {
      dg[1][a][c]=dg1[a][c]; dg[2][a][c]=dg2[a][c]; dg[3][a][c]=dg3[a][c];
    }
    Real partial=0.0, connection=0.0;
    for (int mu=0; mu<4; ++mu) for (int nu=0; nu<4; ++nu) {
      partial+=bavg[mu]*bavg[nu]*ducov[mu][nu];
      for (int lam=0; lam<4; ++lam) {
        Real christ=0.0;
        for (int sig=0; sig<4; ++sig) {
          christ+=0.5*gu[lam][sig]
                     *(dg[mu][sig][nu]+dg[nu][sig][mu]-dg[sig][mu][nu]);
        }
        connection+=bavg[mu]*bavg[nu]*christ*ulavg[lam];
      }
    }
    Real material_mag=0.0;
    for (int mu=0; mu<4; ++mu) material_mag+=uavg[mu]*dmag[mu];
    const Real bsqavg=0.5*(phys0(m,PS_BSQ,k,j,i)+phys1(m,PS_BSQ,k,j,i));
    const Real ideal_mag_resid=material_mag+bsqavg*expansion-(partial-connection);
    const Real qmagloss=-ideal_mag_resid;
    out(m,ED_MAG_LOSS,k,j,i)=qmagloss;
    out(m,ED_HYDRO_DISS,k,j,i)=qdiss_e-qmagloss;

    Real dv[3][3]={{0.0}};
    dv[0][0]=(w(m,IVX,k,j,i+1)-w(m,IVX,k,j,i-1))/(2.0*dx1);
    dv[1][0]=(w(m,IVY,k,j,i+1)-w(m,IVY,k,j,i-1))/(2.0*dx1);
    dv[2][0]=(w(m,IVZ,k,j,i+1)-w(m,IVZ,k,j,i-1))/(2.0*dx1);
    Real dr1=(w(m,IDN,k,j,i+1)-w(m,IDN,k,j,i-1))/(2.0*dx1);
    Real dp1=(w(m,IEN,k,j,i+1)-w(m,IEN,k,j,i-1))*(gamma-1.0)/(2.0*dx1);
    Real j1=0.0,j2=-(b(m,IBZ,k,j,i+1)-b(m,IBZ,k,j,i-1))/(2.0*dx1);
    Real j3=(b(m,IBY,k,j,i+1)-b(m,IBY,k,j,i-1))/(2.0*dx1);
    Real grad_r2=dr1*dr1, grad_p2=dp1*dp1;
    if (multi_d) {
      dv[0][1]=(w(m,IVX,k,j+1,i)-w(m,IVX,k,j-1,i))/(2.0*dx2);
      dv[1][1]=(w(m,IVY,k,j+1,i)-w(m,IVY,k,j-1,i))/(2.0*dx2);
      dv[2][1]=(w(m,IVZ,k,j+1,i)-w(m,IVZ,k,j-1,i))/(2.0*dx2);
      Real dr2=(w(m,IDN,k,j+1,i)-w(m,IDN,k,j-1,i))/(2.0*dx2);
      Real dp2=(w(m,IEN,k,j+1,i)-w(m,IEN,k,j-1,i))*(gamma-1.0)/(2.0*dx2);
      grad_r2+=dr2*dr2; grad_p2+=dp2*dp2;
      j1+=(b(m,IBZ,k,j+1,i)-b(m,IBZ,k,j-1,i))/(2.0*dx2);
      j3-=(b(m,IBX,k,j+1,i)-b(m,IBX,k,j-1,i))/(2.0*dx2);
    }
    if (three_d) {
      dv[0][2]=(w(m,IVX,k+1,j,i)-w(m,IVX,k-1,j,i))/(2.0*dx3);
      dv[1][2]=(w(m,IVY,k+1,j,i)-w(m,IVY,k-1,j,i))/(2.0*dx3);
      dv[2][2]=(w(m,IVZ,k+1,j,i)-w(m,IVZ,k-1,j,i))/(2.0*dx3);
      Real dr3=(w(m,IDN,k+1,j,i)-w(m,IDN,k-1,j,i))/(2.0*dx3);
      Real dp3=(w(m,IEN,k+1,j,i)-w(m,IEN,k-1,j,i))*(gamma-1.0)/(2.0*dx3);
      grad_r2+=dr3*dr3; grad_p2+=dp3*dp3;
      j1-=(b(m,IBY,k+1,j,i)-b(m,IBY,k-1,j,i))/(2.0*dx3);
      j2+=(b(m,IBX,k+1,j,i)-b(m,IBX,k-1,j,i))/(2.0*dx3);
    }
    const Real h=fmax(dx1,fmax(dx2,dx3));
    const Real divv=dv[0][0]+dv[1][1]+dv[2][2];
    const Real rho=fmax(fabs(w(m,IDN,k,j,i)),1.0e-30);
    const Real p=fmax(fabs((gamma-1.0)*w(m,IEN,k,j,i)),1.0e-30);
    const Real bmag=sqrt(SQR(b(m,IBX,k,j,i))+SQR(b(m,IBY,k,j,i))+SQR(b(m,IBZ,k,j,i)));
    const Real cs=sqrt(fmax(gamma*p/(rho+gamma*p/(gamma-1.0)),1.0e-30));
    out(m,ED_SHOCK_SENSOR,k,j,i)=fmax(-expansion,0.0)*h/cs;
    out(m,ED_CURRENT_SENSOR,k,j,i)=sqrt(j1*j1+j2*j2+j3*j3)*h/(bmag+sqrt(p)+1.0e-30);
    const Real gradr=sqrt(grad_r2)*h/rho;
    const Real gradp=sqrt(grad_p2)*h/p;
    out(m,ED_CONTACT_SENSOR,k,j,i)=gradr/(1.0+gradp);
    const Real om1=dv[2][1]-dv[1][2];
    const Real om2=dv[0][2]-dv[2][0];
    const Real om3=dv[1][0]-dv[0][1];
    Real shear2=0.0;
    for (int a=0; a<3; ++a) {
      for (int c=0; c<3; ++c) {
        Real sij=0.5*(dv[a][c]+dv[c][a]);
        if (a==c) sij-=divv/3.0;
        shear2+=sij*sij;
      }
    }
    out(m,ED_VORT_SENSOR,k,j,i)=sqrt(om1*om1+om2*om2+om3*om3)*h/cs;
    out(m,ED_SHEAR_SENSOR,k,j,i)=sqrt(2.0*shear2)*h/cs;
    Real pjump=fmax(fabs(phys1(m,PS_PRESSURE,k,j,i+1)-phys1(m,PS_PRESSURE,k,j,i-1)),0.0);
    if (multi_d) pjump=fmax(pjump,fabs(phys1(m,PS_PRESSURE,k,j+1,i)-phys1(m,PS_PRESSURE,k,j-1,i)));
    if (three_d) pjump=fmax(pjump,fabs(phys1(m,PS_PRESSURE,k+1,j,i)-phys1(m,PS_PRESSURE,k-1,j,i)));
    out(m,ED_PRESSURE_JUMP,k,j,i)=pjump/(2.0*p);
    Real reversal=0.0;
    Real bm1=b(m,IBX,k,j,i-1), bm2=b(m,IBY,k,j,i-1), bm3=b(m,IBZ,k,j,i-1);
    Real bp1=b(m,IBX,k,j,i+1), bp2=b(m,IBY,k,j,i+1), bp3=b(m,IBZ,k,j,i+1);
    reversal=fmax(reversal,-(bm1*bp1+bm2*bp2+bm3*bp3)/
                           (sqrt(bm1*bm1+bm2*bm2+bm3*bm3)*
                            sqrt(bp1*bp1+bp2*bp2+bp3*bp3)+1.0e-30));
    if (multi_d) {
      bm1=b(m,IBX,k,j-1,i); bm2=b(m,IBY,k,j-1,i); bm3=b(m,IBZ,k,j-1,i);
      bp1=b(m,IBX,k,j+1,i); bp2=b(m,IBY,k,j+1,i); bp3=b(m,IBZ,k,j+1,i);
      reversal=fmax(reversal,-(bm1*bp1+bm2*bp2+bm3*bp3)/
                             (sqrt(bm1*bm1+bm2*bm2+bm3*bm3)*
                              sqrt(bp1*bp1+bp2*bp2+bp3*bp3)+1.0e-30));
    }
    if (three_d) {
      bm1=b(m,IBX,k-1,j,i); bm2=b(m,IBY,k-1,j,i); bm3=b(m,IBZ,k-1,j,i);
      bp1=b(m,IBX,k+1,j,i); bp2=b(m,IBY,k+1,j,i); bp3=b(m,IBZ,k+1,j,i);
      reversal=fmax(reversal,-(bm1*bp1+bm2*bp2+bm3*bp3)/
                             (sqrt(bm1*bm1+bm2*bm2+bm3*bm3)*
                              sqrt(bp1*bp1+bp2*bp2+bp3*bp3)+1.0e-30));
    }
    out(m,ED_FIELD_REVERSAL,k,j,i)=fmax(reversal,0.0);
    out(m,ED_RHO,k,j,i)=phys1(m,PS_RHO,k,j,i);
    out(m,ED_EINT,k,j,i)=phys1(m,PS_EINT,k,j,i);
    out(m,ED_PRESSURE,k,j,i)=phys1(m,PS_PRESSURE,k,j,i);
    out(m,ED_UT,k,j,i)=phys1(m,PS_UT,k,j,i);
    out(m,ED_U1,k,j,i)=phys1(m,PS_U1,k,j,i);
    out(m,ED_U2,k,j,i)=phys1(m,PS_U2,k,j,i);
    out(m,ED_U3,k,j,i)=phys1(m,PS_U3,k,j,i);
    out(m,ED_B1,k,j,i)=b(m,IBX,k,j,i);
    out(m,ED_B2,k,j,i)=b(m,IBY,k,j,i);
    out(m,ED_B3,k,j,i)=b(m,IBZ,k,j,i);
    out(m,ED_BSQ,k,j,i)=phys1(m,PS_BSQ,k,j,i);
    out(m,ED_FLAGS,k,j,i)=static_cast<Real>(flg(m,k,j,i));
  });
  ++sample_number_;
  const Real end_time = pmy_pack_->pmesh->time + pmy_pack_->pmesh->dt;
  // Normally one interval is crossed (dt_sim << sample_dt).  Jump arithmetically if a
  // deliberately coarse test step crosses several targets; do not iterate over each one.
  const Real crossed = std::floor((end_time-next_sample_time_)/sample_dt_)+1.0;
  next_sample_time_ += std::max(1.0,crossed)*sample_dt_;
  while (static_cast<float>(next_sample_time_) <= static_cast<float>(end_time)) {
    next_sample_time_ += sample_dt_;
  }
  if (global_variable::my_rank == 0) {
    std::cout << "ENERGY_DIAG sample_complete=" << sample_number_
              << " next_target=" << next_sample_time_ << std::endl;
  }
  return TaskStatus::complete;
}

} // namespace diagnostics
