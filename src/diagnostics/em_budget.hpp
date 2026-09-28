#ifndef DIAGNOSTICS_EM_BUDGET_HPP_
#define DIAGNOSTICS_EM_BUDGET_HPP_

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace diagnostics {
// Read-only ideal-MHD electromagnetic stress tensor. CKS has sqrt(-g)=1.
// w is Athena's normal-frame spatial four-velocity, NOT coordinate dx/dt.
KOKKOS_INLINE_FUNCTION
void EMState(const Real w[3], const Real B[3], const Real g[4][4],
             const Real gi[4][4], Real u[4], Real b[4], Real &bsq,
             Real mixed[4][4]) {
  Real uu=0.0;
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
    uu+=g[i+1][j+1]*w[i]*w[j];
  const Real alpha=sqrt(-1.0/gi[0][0]), lor=sqrt(1.0+uu);
  u[0]=lor/alpha;
  for (int i=1; i<4; ++i) u[i]=w[i-1]-alpha*lor*gi[0][i];
  Real ul[4]={0.0}, bl[4]={0.0};
  for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) ul[i]+=g[i][j]*u[j];
  b[0]=ul[1]*B[0]+ul[2]*B[1]+ul[3]*B[2];
  for (int i=1; i<4; ++i) b[i]=(B[i-1]+b[0]*u[i])/u[0];
  for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) bl[i]+=g[i][j]*b[j];
  bsq=0.0;
  for (int i=0; i<4; ++i) bsq+=bl[i]*b[i];
  for (int i=0; i<4; ++i) for (int j=0; j<4; ++j)
    mixed[i][j]=bsq*u[i]*ul[j]-b[i]*bl[j]+(i==j ? 0.5*bsq : 0.0);
}

// FOFC uses LLF, not HLLE. Reconstruct its EM part with the identical first-order
// states and wave-speed prescription; production fluxes are not touched.
template <typename Prim, typename EOS, typename Coord>
KOKKOS_INLINE_FUNCTION
void EMFluxLLF(const Prim &wl, const Prim &wr, Real bx, Real x, Real y, Real z,
               int dir, const Coord &coord, const EOS &eos, Real flux[4]) {
  const int j=1+dir%3, k=1+(dir+1)%3;
  Real g[4][4], gi[4][4];
  ComputeMetricAndInverse(x,y,z,coord.is_minkowski,coord.bh_spin,g,gi);
  Real wL[3], wR[3], BL[3], BR[3];
  wL[dir-1]=wl.vx; wL[j-1]=wl.vy; wL[k-1]=wl.vz;
  wR[dir-1]=wr.vx; wR[j-1]=wr.vy; wR[k-1]=wr.vz;
  BL[dir-1]=BR[dir-1]=bx;
  BL[j-1]=wl.by; BL[k-1]=wl.bz; BR[j-1]=wr.by; BR[k-1]=wr.bz;
  Real ul[4], ur[4], bl[4], br[4], b2l, b2r, tl[4][4], tr[4][4];
  EMState(wL,BL,g,gi,ul,bl,b2l,tl);
  EMState(wR,BR,g,gi,ur,br,b2r,tr);
  Real lp, lm, rp, rm;
  eos.IdealGRMHDFastSpeeds(wl.d,eos.IdealGasPressure(wl.e),ul[0],ul[dir],b2l,
                          gi[0][0],gi[0][dir],gi[dir][dir],lp,lm);
  eos.IdealGRMHDFastSpeeds(wr.d,eos.IdealGasPressure(wr.e),ur[0],ur[dir],b2r,
                          gi[0][0],gi[0][dir],gi[dir][dir],rp,rm);
  const Real speed=fmax(fmax(lp,rp),-fmin(lm,rm));
  for (int n=0; n<4; ++n)
    flux[n]=0.5*(tl[dir][n]+tr[dir][n]-speed*(tr[0][n]-tl[0][n]));
}
} // namespace diagnostics
#endif
