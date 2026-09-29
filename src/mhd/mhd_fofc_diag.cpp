// Passive observation of the ORIGINAL FOFC operator; never replaces the solver.
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "coordinates/cell_locations.hpp"
#include "diagnostics/energy_diagnostics.hpp"
#include "diagnostics/em_budget.hpp"

namespace mhd {
void MHD::SnapshotFOFCEnergy() {
  auto *diag=pmy_pack->penergy_diag;
  if (diag==nullptr || !diag->recording) return;
  diagnostics::ObserverPhase phase;
  const auto ix=pmy_pack->pmesh->mb_indcs;
  for (int dir=1; dir<=3; ++dir) {
    if (dir==2 && !pmy_pack->pmesh->multi_d) continue;
    if (dir==3 && !pmy_pack->pmesh->three_d) continue;
    const auto flux=diagnostics::ReadOnly(dir==1 ? uflx.x1f : (dir==2 ? uflx.x2f : uflx.x3f));
    const auto before=dir==1 ? diag->fofc_energy_flux.x1f :
                      (dir==2 ? diag->fofc_energy_flux.x2f : diag->fofc_energy_flux.x3f);
    par_for("diagnostic_fofc_before",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
            ix.ks,ix.ke+(dir==3),ix.js,ix.je+(dir==2),ix.is,ix.ie+(dir==1),
    KOKKOS_LAMBDA(int m,int k,int j,int i) { before(m,k,j,i)=flux(m,IEN,k,j,i); });
  }
}

void MHD::RecordFOFCDiagnostics() {
  auto *diag=pmy_pack->penergy_diag;
  if (diag==nullptr || !diag->recording) return;
  diagnostics::ObserverPhase phase;
  const auto ix=pmy_pack->pmesh->mb_indcs;
  const auto w=diagnostics::ReadOnly(w0), b=diagnostics::ReadOnly(bcc0);
  const auto ff=diagnostics::ReadOnly(fofc);
  const bool use_ff=use_fofc;
  const bool excise=pmy_pack->pcoord->coord_data.bh_excise;
  const bool em=diag->budget_enabled;
  const auto excision=pmy_pack->pcoord->excision_flux;
  const auto size=pmy_pack->pmb->mb_size;
  const auto coord=pmy_pack->pcoord->coord_data;
  const auto eos=peos->eos_data;
  const auto flags=diag->flags;
  par_for("diagnostic_fofc_flags",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
          ix.ks,ix.ke,ix.js,ix.je,ix.is,ix.ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    if (use_ff && ff(m,k,j,i)) flags(m,k,j,i)|=diagnostics::EDF_FOFC;
    if (excise && excision(m,k,j,i)) flags(m,k,j,i)|=diagnostics::EDF_EXCISION;
  });
  // One writer per face, selected by either adjacent production flag. No duplicate
  // += operations from neighboring flagged cells, and no second floor-test/C2P.
  for (int dir=1; dir<=3; ++dir) {
    if (dir==2 && !pmy_pack->pmesh->multi_d) continue;
    if (dir==3 && !pmy_pack->pmesh->three_d) continue;
    const int ivy=1+dir%3, ivz=1+(dir+1)%3;
    const int iby=ivy-1, ibz=ivz-1;
    const auto flux=diagnostics::ReadOnly(dir==1 ? uflx.x1f : (dir==2 ? uflx.x2f : uflx.x3f));
    const auto bf=diagnostics::ReadOnly(dir==1 ? b0.x1f : (dir==2 ? b0.x2f : b0.x3f));
    const auto delta=dir==1 ? diag->fofc_energy_flux.x1f :
                     (dir==2 ? diag->fofc_energy_flux.x2f : diag->fofc_energy_flux.x3f);
    const auto sf=dir==1 ? diag->entropy_flux.x1f :
                  (dir==2 ? diag->entropy_flux.x2f : diag->entropy_flux.x3f);
    const auto ef=dir==1 ? diag->internal_energy_flux.x1f :
                  (dir==2 ? diag->internal_energy_flux.x2f : diag->internal_energy_flux.x3f);
    const auto vf=dir==1 ? diag->four_velocity_flux.x1f :
                  (dir==2 ? diag->four_velocity_flux.x2f : diag->four_velocity_flux.x3f);
    const auto emflux=dir==1 ? diag->sidecar_flx.x1f :
                      (dir==2 ? diag->sidecar_flx.x2f : diag->sidecar_flx.x3f);
    par_for("diagnostic_fofc_faces",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
            ix.ks,ix.ke+(dir==3),ix.js,ix.je+(dir==2),ix.is,ix.ie+(dir==1),
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      delta(m,k,j,i)=flux(m,IEN,k,j,i)-delta(m,k,j,i);
      const int il=i-(dir==1), jl=j-(dir==2), kl=k-(dir==3);
      const bool corrected=(use_ff && (ff(m,kl,jl,il)||ff(m,k,j,i))) ||
                           (excise && (excision(m,kl,jl,il)||excision(m,k,j,i)));
      if (!corrected) return;
      MHDPrim1D left, right;
      left.d=w(m,IDN,kl,jl,il); right.d=w(m,IDN,k,j,i);
      left.e=w(m,IEN,kl,jl,il); right.e=w(m,IEN,k,j,i);
      left.vx=w(m,dir,kl,jl,il); right.vx=w(m,dir,k,j,i);
      left.vy=w(m,ivy,kl,jl,il); right.vy=w(m,ivy,k,j,i);
      left.vz=w(m,ivz,kl,jl,il); right.vz=w(m,ivz,k,j,i);
      left.by=b(m,iby,kl,jl,il); right.by=b(m,iby,k,j,i);
      left.bz=b(m,ibz,kl,jl,il); right.bz=b(m,ibz,k,j,i);
      const Real mass=flux(m,IDN,k,j,i);  // actual accepted mass flux, not recomputed
      const Real rho=mass>=0.0 ? left.d : right.d;
      const Real eint=mass>=0.0 ? left.e : right.e;
      const Real entropy=(log(fmax((eos.gamma-1.0)*eint,1.0e-300))-
                          eos.gamma*log(fmax(rho,1.0e-300)))/(eos.gamma-1.0);
      sf(m,k,j,i)=mass*entropy;
      ef(m,k,j,i)=mass*eint/fmax(rho,1.0e-300);
      vf(m,k,j,i)=mass/fmax(rho,1.0e-300);
      if (em) {
        const auto sz=size.d_view(m);
        const Real x=dir==1 ? LeftEdgeX(i-ix.is,ix.nx1,sz.x1min,sz.x1max) :
                             CellCenterX(i-ix.is,ix.nx1,sz.x1min,sz.x1max);
        const Real y=dir==2 ? LeftEdgeX(j-ix.js,ix.nx2,sz.x2min,sz.x2max) :
                             CellCenterX(j-ix.js,ix.nx2,sz.x2min,sz.x2max);
        const Real z=dir==3 ? LeftEdgeX(k-ix.ks,ix.nx3,sz.x3min,sz.x3max) :
                             CellCenterX(k-ix.ks,ix.nx3,sz.x3min,sz.x3max);
        Real f[4];
        diagnostics::EMFluxLLF(left,right,bf(m,k,j,i),x,y,z,dir,coord,eos,f);
        for (int n=0; n<4; ++n)
          emflux(m,diagnostics::EnergyDiagnostics::SC_EM0+n,k,j,i)=f[n];
      }
    });
  }
}
} // namespace mhd
