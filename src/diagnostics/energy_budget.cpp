// Passive finite-volume electromagnetic ledger. No array in this file feeds evolution.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "diagnostics/energy_diagnostics.hpp"
#include "diagnostics/em_budget.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "globals.hpp"

namespace diagnostics {
void WriteEnergyFingerprint(ParameterInput *pin, Mesh *mesh) {
  // Only enrolled by this pgen's explicit energy_fingerprint switch. No diagnostic
  // object or extra solver calls are needed in the diagnostics-OFF comparison run.
  auto *pack=mesh->pmb_pack;
  const auto ix=mesh->mb_indcs;
  std::uint64_t h1=14695981039346656037ull, h2=0, count=0;
  auto add=[&](Real value) {
    unsigned char bytes[sizeof(Real)];
    std::memcpy(bytes,&value,sizeof(Real));
    for (unsigned char byte:bytes) {h1^=byte; h1*=1099511628211ull;}
    std::uint64_t bits=0;
    std::memcpy(&bits,&value,sizeof(Real));
    bits^=++count*0x9e3779b97f4a7c15ull;
    bits=(bits^(bits>>30))*0xbf58476d1ce4e5b9ull;
    bits=(bits^(bits>>27))*0x94d049bb133111ebull;
    h2+=bits^(bits>>31);
  };
  auto cc=[&](const DvceArray5D<Real> &device) {
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device);
    for (int m=0; m<pack->nmb_thispack; ++m) for (int n=0; n<device.extent_int(1); ++n)
      for (int k=ix.ks; k<=ix.ke; ++k) for (int j=ix.js; j<=ix.je; ++j)
        for (int i=ix.is; i<=ix.ie; ++i) add(host(m,n,k,j,i));
  };
  auto fc=[&](const DvceArray4D<Real> &device,int dir) {
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device);
    for (int m=0; m<pack->nmb_thispack; ++m)
      for (int k=ix.ks; k<=ix.ke+(dir==3); ++k)
        for (int j=ix.js; j<=ix.je+(dir==2); ++j)
          for (int i=ix.is; i<=ix.ie+(dir==1); ++i) add(host(m,k,j,i));
  };
  if (pack->pmhd==nullptr || pack->prad==nullptr)
    Kokkos::abort("Energy fingerprint requires radiation MHD");
  cc(pack->pmhd->u0);
  fc(pack->pmhd->b0.x1f,1); fc(pack->pmhd->b0.x2f,2); fc(pack->pmhd->b0.x3f,3);
  cc(pack->prad->i0);
  std::ostringstream name;
  name<<"fingerprint_rank_"<<std::setw(8)<<std::setfill('0')<<global_variable::my_rank<<".json";
  std::ofstream f(name.str());
  if (!f) Kokkos::abort("Cannot write passive state fingerprint");
  f<<std::setprecision(17)<<"{\"time\":"<<mesh->time<<",\"cycle\":"<<mesh->ncycle
   <<",\"dt\":"<<mesh->dt<<",\"values\":"<<count<<",\"hash\":\""
   <<std::hex<<std::setw(16)<<std::setfill('0')<<h1<<std::setw(16)<<h2<<"\"}\n";
}

namespace {
KOKKOS_INLINE_FUNCTION
void CellEM(const DvceArray5D<Real> &w, const DvceArray5D<Real> &bcc,
            const DualArray1D<RegionSize> &size, const RegionIndcs &ix,
            bool flat, Real spin, int m, int k, int j, int i,
            Real g[4][4], Real gi[4][4], Real u[4], Real b[4], Real &b2,
            Real t[4][4]) {
  const Real x=CellCenterX(i-ix.is,ix.nx1,size.d_view(m).x1min,size.d_view(m).x1max);
  const Real y=CellCenterX(j-ix.js,ix.nx2,size.d_view(m).x2min,size.d_view(m).x2max);
  const Real z=CellCenterX(k-ix.ks,ix.nx3,size.d_view(m).x3min,size.d_view(m).x3max);
  ComputeMetricAndInverse(x,y,z,flat,spin,g,gi);
  Real wp[3]={w(m,IVX,k,j,i),w(m,IVY,k,j,i),w(m,IVZ,k,j,i)};
  Real B[3]={bcc(m,IBX,k,j,i),bcc(m,IBY,k,j,i),bcc(m,IBZ,k,j,i)};
  EMState(wp,B,g,gi,u,b,b2,t);
}
}

void EnergyDiagnostics::InitializeBudget() {
  const int nm=output.extent_int(0), nk=output.extent_int(2);
  const int nj=output.extent_int(3), ni=output.extent_int(4);
  Kokkos::realloc(budget_state,nm,11,nk,nj,ni);
  Kokkos::realloc(budget_delta,nm,11,nk,nj,ni);
  Kokkos::realloc(em_source_before,nm,4,nk,nj,ni);
  Kokkos::realloc(budget_window,nm,NENERGY_DIAG,nk,nj,ni);
  Kokkos::realloc(window_flags,nm,nk,nj,ni);
  Kokkos::deep_copy(budget_window,0.0);
  Kokkos::deep_copy(window_flags,0u);
}

void EnergyDiagnostics::BeginBudgetStep() {
  auto w=pmy_pack_->pmhd->w0, bcc=pmy_pack_->pmhd->bcc0;
  auto size=pmy_pack_->pmb->mb_size;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool flat=pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin=pmy_pack_->pcoord->coord_data.bh_spin;
  auto state=budget_state;
  par_for("budget_initial",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real g[4][4],gi[4][4],u[4],b[4],b2,t[4][4];
    CellEM(w,bcc,size,ix,flat,spin,m,k,j,i,g,gi,u,b,b2,t);
    for (int n=0; n<4; ++n) {state(m,n,k,j,i)=t[0][n]; state(m,7+n,k,j,i)=u[n];}
    for (int n=0; n<3; ++n) state(m,4+n,k,j,i)=bcc(m,n,k,j,i);
  });
}

void EnergyDiagnostics::BeginBudgetStage(Driver *driver,int stage) {
  if (stage==1) {
    Kokkos::deep_copy(DevExeSpace(),budget_delta,0.0);
  } else {
    auto d=budget_delta;
    const Real weight=driver->gam0[stage-1];
    const auto ix=pmy_pack_->pmesh->mb_indcs;
    par_for("budget_rk",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,0,10,
            ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
    KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {d(m,n,k,j,i)*=weight;});
  }
  // GatherSidecarFluxes overwrites only the five historical scalar fields.
  Kokkos::deep_copy(DevExeSpace(),sidecar_flx.x1f,0.0);
  Kokkos::deep_copy(DevExeSpace(),sidecar_flx.x2f,0.0);
  Kokkos::deep_copy(DevExeSpace(),sidecar_flx.x3f,0.0);
}

void EnergyDiagnostics::RecordEMFlux(Driver *driver,int stage) {
  auto w=pmy_pack_->pmhd->w0, bcc=pmy_pack_->pmhd->bcc0;
  auto size=pmy_pack_->pmb->mb_size;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool flat=pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin=pmy_pack_->pcoord->coord_data.bh_spin;
  const bool md=pmy_pack_->pmesh->multi_d, td=pmy_pack_->pmesh->three_d;
  const Real h=driver->beta[stage-1]*pmy_pack_->pmesh->dt;
  auto f=sidecar_flx;
  auto d=budget_delta;
  par_for("budget_flux_geometry",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real g[4][4],gi[4][4],u[4],b[4],b2,t[4][4];
    CellEM(w,bcc,size,ix,flat,spin,m,k,j,i,g,gi,u,b,b2,t);
    const auto sz=size.d_view(m);
    const Real x=CellCenterX(i-ix.is,ix.nx1,sz.x1min,sz.x1max);
    const Real y=CellCenterX(j-ix.js,ix.nx2,sz.x2min,sz.x2max);
    const Real z=CellCenterX(k-ix.ks,ix.nx3,sz.x3min,sz.x3max);
    Real dg[3][4][4];
    ComputeMetricDerivatives(x,y,z,flat,spin,dg[0],dg[1],dg[2]);
    for (int n=0; n<4; ++n) {
      Real div=(f.x1f(m,SC_EM0+n,k,j,i+1)-f.x1f(m,SC_EM0+n,k,j,i))/sz.dx1;
      if (md) div+=(f.x2f(m,SC_EM0+n,k,j+1,i)-f.x2f(m,SC_EM0+n,k,j,i))/sz.dx2;
      if (td) div+=(f.x3f(m,SC_EM0+n,k+1,j,i)-f.x3f(m,SC_EM0+n,k,j,i))/sz.dx3;
      Real geom=0.0;
      if (n>0) for (int a=0; a<4; ++a) for (int c=0; c<4; ++c)
        geom+=0.5*(b2*u[a]*u[c]+0.5*b2*gi[a][c]-b[a]*b[c])*dg[n-1][a][c];
      d(m,n,k,j,i)+=h*(geom-div);
    }
  });
}

void EnergyDiagnostics::SaveEMSourceState() {
  // Called only AFTER radiation's existing pre-source primitive recovery.
  source_pending_=true;
  auto w=pmy_pack_->pmhd->w0, bcc=pmy_pack_->pmhd->bcc0;
  auto size=pmy_pack_->pmb->mb_size;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool flat=pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin=pmy_pack_->pcoord->coord_data.bh_spin;
  auto before=em_source_before;
  par_for("budget_source_before",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real g[4][4],gi[4][4],u[4],b[4],b2,t[4][4];
    CellEM(w,bcc,size,ix,flat,spin,m,k,j,i,g,gi,u,b,b2,t);
    for (int n=0; n<4; ++n) before(m,n,k,j,i)=t[0][n];
  });
}

void EnergyDiagnostics::FinishBudgetStage() {
  if (!budget_enabled || !recording || !source_pending_) return;
  // AFTER the solver's normal end-stage recovery: no stale primitives, no extra C2P.
  // Includes any post-source repairs. Flagged cells remain separately identifiable.
  source_pending_=false;
  auto w=pmy_pack_->pmhd->w0, bcc=pmy_pack_->pmhd->bcc0;
  auto size=pmy_pack_->pmb->mb_size;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool flat=pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin=pmy_pack_->pcoord->coord_data.bh_spin;
  auto before=em_source_before, d=budget_delta;
  par_for("budget_source_after",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real g[4][4],gi[4][4],u[4],b[4],b2,t[4][4];
    CellEM(w,bcc,size,ix,flat,spin,m,k,j,i,g,gi,u,b,b2,t);
    for (int n=0; n<4; ++n) d(m,4+n,k,j,i)+=t[0][n]-before(m,n,k,j,i);
  });
}

void EnergyDiagnostics::RecordCT(Driver *driver,int stage) {
  if (!budget_enabled || !recording) return;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool md=pmy_pack_->pmesh->multi_d, td=pmy_pack_->pmesh->three_d;
  auto size=pmy_pack_->pmb->mb_size;
  auto e=pmy_pack_->pmhd->efld;
  auto d=budget_delta;
  const Real h=driver->beta[stage-1]*pmy_pack_->pmesh->dt;
  // Sum the actual corrected-edge curls at both faces of each cell. This checks
  // RK/CT bookkeeping, not equality of the HLL stress split to a CT Poynting flux.
  par_for("budget_ct",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    const auto sz=size.d_view(m);
    Real db[3]={0.0};
    for (int s=0; s<2; ++s) {
      if (md) db[0]-=(e.x3e(m,k,j+1,i+s)-e.x3e(m,k,j,i+s))/sz.dx2;
      if (td) db[0]+=(e.x2e(m,k+1,j,i+s)-e.x2e(m,k,j,i+s))/sz.dx3;
      db[1]+=(e.x3e(m,k,j+s,i+1)-e.x3e(m,k,j+s,i))/sz.dx1;
      if (td) db[1]-=(e.x1e(m,k+1,j+s,i)-e.x1e(m,k,j+s,i))/sz.dx3;
      db[2]-=(e.x2e(m,k+s,j,i+1)-e.x2e(m,k+s,j,i))/sz.dx1;
      if (md) db[2]+=(e.x1e(m,k+s,j+1,i)-e.x1e(m,k+s,j,i))/sz.dx2;
    }
    for (int n=0; n<3; ++n) d(m,8+n,k,j,i)+=0.5*h*db[n];
  });
}

void EnergyDiagnostics::FinalizeBudgetStep() {
  auto w=pmy_pack_->pmhd->w0, bcc=pmy_pack_->pmhd->bcc0;
  auto size=pmy_pack_->pmb->mb_size;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  const bool flat=pmy_pack_->pcoord->coord_data.is_minkowski;
  const Real spin=pmy_pack_->pcoord->coord_data.bh_spin, dt=pmy_pack_->pmesh->dt;
  auto initial=budget_state, d=budget_delta, out=output;
  par_for("budget_finalize",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real g[4][4],gi[4][4],u[4],b[4],b2,t[4][4];
    CellEM(w,bcc,size,ix,flat,spin,m,k,j,i,g,gi,u,b,b2,t);
    Real heat=0.0, source=0.0, quadrature=0.0, ua[4], norm=0.0, ul0=0.0;
    for (int n=0; n<4; ++n) ua[n]=0.5*(u[n]+initial(m,7+n,k,j,i));
    for (int n=0; n<4; ++n) for (int l=0; l<4; ++l) norm-=g[n][l]*ua[n]*ua[l];
    for (int n=0; n<4; ++n) {ua[n]/=sqrt(norm); ul0+=g[0][n]*ua[n];}
    for (int n=0; n<4; ++n) {
      const Real r=(t[0][n]-initial(m,n,k,j,i)-d(m,n,k,j,i)-d(m,4+n,k,j,i))/dt;
      out(m,ED_EM_R0+n,k,j,i)=r;
      heat+=ua[n]*r;
      source+=ua[n]*d(m,4+n,k,j,i)/dt;
      quadrature+=0.5*(u[n]-initial(m,7+n,k,j,i))*r;
    }
    const Real total=out(m,ED_DISS_ENERGY,k,j,i), mech=total-heat;
    out(m,ED_MAG_BUDGET,k,j,i)=heat;
    out(m,ED_MECH_BUDGET,k,j,i)=mech;
    out(m,ED_MAG_SOURCE,k,j,i)=source;
    out(m,ED_MAG_QUADRATURE,k,j,i)=fabs(quadrature);
    // -C_0 = R_0 = (-u_0) q + bulk work. These two fields are coordinate
    // Killing-energy rates, NOT additional comoving heat sources.
    out(m,ED_MAG_HEAT_INF,k,j,i)=-ul0*heat;
    out(m,ED_EM_BULK_WORK,k,j,i)=out(m,ED_EM_R0,k,j,i)+ul0*heat;
    out(m,ED_MAG_POS,k,j,i)=fmax(heat,0.0);
    out(m,ED_MAG_NEG,k,j,i)=fmax(-heat,0.0);
    out(m,ED_MECH_POS,k,j,i)=fmax(mech,0.0);
    out(m,ED_MECH_NEG,k,j,i)=fmax(-mech,0.0);
    out(m,ED_TOTAL_POS,k,j,i)=fmax(total,0.0);
    out(m,ED_TOTAL_NEG,k,j,i)=fmax(-total,0.0);
    Real err=0.0, scale=1.0e-30;
    for (int n=0; n<3; ++n) {
      const Real bend=bcc(m,n,k,j,i), bstart=initial(m,4+n,k,j,i);
      err+=fabs(bend-bstart-d(m,8+n,k,j,i));
      scale+=fabs(bend)+fabs(bstart)+fabs(d(m,8+n,k,j,i));
    }
    out(m,ED_CT_CLOSURE,k,j,i)=err/scale;
    out(m,ED_GAS_CLOSURE_ABS,k,j,i)=fabs(out(m,ED_GAS_CLOSURE,k,j,i));
    out(m,ED_THERMO_CLOSURE_ABS,k,j,i)=fabs(out(m,ED_THERMO_CLOSURE,k,j,i));
  });
}

void EnergyDiagnostics::PublishBudgetWindow(bool publish) {
  const Real dt=pmy_pack_->pmesh->dt;
  window_dt_+=dt;
  ++window_steps_;
  const Real duration=window_dt_, steps=window_steps_;
  const Real time=pmy_pack_->pmesh->time+dt, id=sample_number_+1;
  auto out=output, sum=budget_window;
  auto wf=window_flags, fl=flags;
  const auto ix=pmy_pack_->pmesh->mb_indcs;
  par_for("budget_window",DevExeSpace(),0,pmy_pack_->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    wf(m,k,j,i)|=fl(m,k,j,i);
    for (int n=0; n<NENERGY_DIAG; ++n) {
      // CT is a worst-step relative check; flags are an OR, not an average.
      if (n==ED_CT_CLOSURE) sum(m,n,k,j,i)=fmax(sum(m,n,k,j,i),out(m,n,k,j,i));
      else sum(m,n,k,j,i)+=dt*out(m,n,k,j,i);
      if (publish) {
        out(m,n,k,j,i)=sum(m,n,k,j,i)/(n==ED_CT_CLOSURE ? 1.0 : duration);
        sum(m,n,k,j,i)=0.0;
      }
    }
    if (publish) {
      out(m,ED_FLAGS,k,j,i)=static_cast<Real>(wf(m,k,j,i));
      wf(m,k,j,i)=0u;
      out(m,ED_SAMPLE_DT,k,j,i)=duration;
      out(m,ED_SAMPLE_TIME,k,j,i)=time;
      out(m,ED_SAMPLE_ID,k,j,i)=id;
      out(m,ED_WINDOW_STEPS,k,j,i)=steps;
      out(m,ED_BUDGET_VERSION,k,j,i)=1.0;
    }
  });
  if (publish) {window_dt_=0.0; window_steps_=0;}
}
} // namespace diagnostics
