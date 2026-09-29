#ifndef DIAGNOSTICS_THERMODYNAMIC_BUDGET_HPP_
#define DIAGNOSTICS_THERMODYNAMIC_BUDGET_HPP_
#include <cmath>
#include "athena.hpp"
namespace diagnostics {
// Logarithmic mean, continuous at a=b. Detached diagnostics only.
KOKKOS_INLINE_FUNCTION
Real ThermoLogMean(Real a, Real b) {
  if (!(a>0.0 && b>0.0)) return NAN;
  const Real mean=0.5*a+0.5*b, z=(0.5*b-0.5*a)/mean, z2=z*z;
  if (fabs(z)<1.0e-3) return mean/(1.0+z2/3.0+z2*z2/5.0+z2*z2*z2/7.0);
  return (b-a)/(log(b)-log(a));
}
struct ThermoTimeWeights { Real temperature, pressure, mass; };
// Exact finite-increment Gibbs identity for an ideal gas:
// Delta(e ut) + p_* Delta(ut) = T_* Delta(D s) + mu_* Delta(D), D=rho ut.
// Fixes temporal quadrature, NOT the independent spatial transport defect.
KOKKOS_INLINE_FUNCTION
ThermoTimeWeights ThermoWeights(Real a0, Real a1, Real d0, Real d1,
                               Real ut0, Real ut1, Real s0, Real s1, Real gamma) {
  const Real al=ThermoLogMean(a0,a1), dl=ThermoLogMean(d0,d1);
  const Real wl=ThermoLogMean(ut0,ut1);
  const Real t=(gamma-1.0)*al/(0.5*d0+0.5*d1);
  return {t,(gamma-1.0)*al/wl,gamma*al/dl-t*(0.5*s0+0.5*s1)};
}
} // namespace diagnostics
#endif
