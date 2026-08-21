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
#include "diagnostics/energy_diagnostics.hpp"
#include "globals.hpp"

namespace diagnostics {

const char *EnergyDiagnostics::label[NENERGY_DIAG] = {
  "dE_gas_flux", "dE_gas_hlle", "dE_gas_fofc", "dE_gas_coord",
  "dE_gas_other", "dE_gas_rad", "dE_gas_abs", "dE_gas_compt",
  "dE_rad_reject", "dE_rad_spatial", "dE_rad_angular", "dE_rad_fix",
  "dE_rad_source", "dE_gas_c2p", "dS_advect", "qent_rad",
  "qent_total", "qent_num",
  "dE_gas_actual", "dE_gas_closure", "dE_rad_actual", "dE_rad_closure",
  "shock_sensor", "current_sensor", "contact_sensor", "vort_sensor",
  "shear_sensor", "repair_flags"
};

EnergyDiagnostics::EnergyDiagnostics(MeshBlockPack *ppack, ParameterInput *pin) :
    step("energy_diag_step",1,1,1,1,1),
    output("energy_diag_output",1,1,1,1,1),
    gas_initial("energy_diag_gas_initial",1,1,1,1,1),
    rad_initial("energy_diag_rad_initial",1,1,1,1,1),
    gas_scratch("energy_diag_gas_scratch",1,1,1,1,1),
    rad_scratch("energy_diag_rad_scratch",1,1,1,1,1),
    entropy_initial("energy_diag_entropy_initial",1,1,1,1,1),
    rad_energy_flux("energy_diag_rad_flux",1,1,1,1,1),
    hlle_energy_flux("energy_diag_hlle_flux",1,1,1,1),
    fofc_energy_flux("energy_diag_fofc_flux",1,1,1,1),
    entropy_flux("energy_diag_entropy_flux",1,1,1,1),
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
  Kokkos::realloc(flags,nmb,n3,n2,n1);

  Kokkos::deep_copy(step,0.0);
  Kokkos::deep_copy(output,0.0);
  Kokkos::deep_copy(flags,0u);
}

void EnergyDiagnostics::AssembleTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  tl["before_timeintegrator"]->AddTask(&EnergyDiagnostics::BeginTimestep,this,none);

  // The diagnostic RK recurrence must be applied before any stage operator. Insert it
  // before Radiation::CopyCons, the first task for radiation-MHD evolution.
  TaskID first = pmy_pack_->prad->id.copyu;
  tl["stagen"]->InsertTask(&EnergyDiagnostics::BeginStage,this,none,first);
  tl["after_timeintegrator"]->AddTask(&EnergyDiagnostics::FinalizeTimestep,this,none);
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

  SaveRadiationEnergy();
  Kokkos::deep_copy(DevExeSpace(),rad_initial,rad_scratch);
  return TaskStatus::complete;
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
  auto size = pmy_pack_->pmb->mb_size;
  auto d = step;
  par_for("energy_diag_mhd_flux",DevExeSpace(),0,nmb1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real total = (flx.x1f(m,IEN,k,j,i+1)-flx.x1f(m,IEN,k,j,i))/size.d_view(m).dx1;
    Real dhll = (hll.x1f(m,k,j,i+1)-hll.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real dfof = (fof.x1f(m,k,j,i+1)-fof.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real divs = (sflx.x1f(m,k,j,i+1)-sflx.x1f(m,k,j,i))/size.d_view(m).dx1;
    if (multi_d) {
      total += (flx.x2f(m,IEN,k,j+1,i)-flx.x2f(m,IEN,k,j,i))/size.d_view(m).dx2;
      dhll += (hll.x2f(m,k,j+1,i)-hll.x2f(m,k,j,i))/size.d_view(m).dx2;
      dfof += (fof.x2f(m,k,j+1,i)-fof.x2f(m,k,j,i))/size.d_view(m).dx2;
      divs += (sflx.x2f(m,k,j+1,i)-sflx.x2f(m,k,j,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      total += (flx.x3f(m,IEN,k+1,j,i)-flx.x3f(m,IEN,k,j,i))/size.d_view(m).dx3;
      dhll += (hll.x3f(m,k+1,j,i)-hll.x3f(m,k,j,i))/size.d_view(m).dx3;
      dfof += (fof.x3f(m,k+1,j,i)-fof.x3f(m,k,j,i))/size.d_view(m).dx3;
      divs += (sflx.x3f(m,k+1,j,i)-sflx.x3f(m,k,j,i))/size.d_view(m).dx3;
    }
    d(m,ED_GAS_FLUX,k,j,i) += -beta_dt*total;
    d(m,ED_GAS_HLLE,k,j,i) += -beta_dt*dhll;
    d(m,ED_GAS_FOFC,k,j,i) += -beta_dt*dfof;
    d(m,ED_DS_ADVECT,k,j,i) += -beta_dt*divs;
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
  auto d = step;
  auto out = output;
  auto rflux = rad_energy_flux;
  auto flg = flags;

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
    const Real rho_s=fmax(w(m,IDN,k,j,i),1.0e-300);
    const Real p_s=fmax((gamma-1.0)*w(m,IEN,k,j,i),1.0e-300);
    const Real s_final=(log(p_s)-gamma*log(rho_s))/(gamma-1.0);
    const Real ds_cons=u(m,IDN,k,j,i)*s_final-initial_s(m,0,k,j,i)
                      -d(m,ED_DS_ADVECT,k,j,i);
    const Real qent_total=(p_s/rho_s)*ds_cons/dt;
    out(m,ED_QENT_TOTAL,k,j,i)=qent_total;
    out(m,ED_QENT_NUM,k,j,i)=qent_total-out(m,ED_QENT_RAD,k,j,i);
    out(m,ED_GAS_ACTUAL,k,j,i)=gas_actual/dt;
    out(m,ED_GAS_CLOSURE,k,j,i)=(gas_actual-gas_sum)/dt;
    out(m,ED_RAD_ACTUAL,k,j,i)=rad_actual/dt;
    out(m,ED_RAD_CLOSURE,k,j,i)=(rad_actual-rad_sum)/dt;

    const Real dx1=size.d_view(m).dx1, dx2=size.d_view(m).dx2, dx3=size.d_view(m).dx3;
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
    out(m,ED_SHOCK_SENSOR,k,j,i)=fmax(-divv,0.0)*h/cs;
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
