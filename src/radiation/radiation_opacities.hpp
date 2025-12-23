#ifndef RADIATION_RADIATION_OPACITIES_HPP_
#define RADIATION_RADIATION_OPACITIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (pow_opacity) {  // power law opacity (accounting for diff b/w Ross & Planck)
    Real power_law = (dens*density_scale)*pow(gm1*mu/(temp*temperature_scale), 3.5);
    Real k_a_r = rosseland_coef * power_law;
    Real k_a_p = planck_minus_rosseland_coef * power_law;
    sigma_a = dens*k_a_r*density_scale*length_scale;
    sigma_p = dens*k_a_p*density_scale*length_scale;
    sigma_s = dens*k_s  *density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_p = dens*k_p*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}



//========================================================================================
//=============================== Multi-Frequency Radiation ==============================
//========================================================================================
// TODO: add comments and clean code below

// a^3 / (exp(a) - 1)
KOKKOS_INLINE_FUNCTION
Real FFPlanckMeanDenom(const Real a){
  if (a == 0) return 0;
  return a*SQR(a)/(exp(a)-1.);
}

// d^4/da^4\left( a^3 / (exp(a) - 1) \right)
// a > 0
KOKKOS_INLINE_FUNCTION
Real d4FFPlanckMeanDenom(const Real a){
  Real e  = exp(a);
  Real em = e-1.;

  Real a2 = SQR(a);
  Real a3 = a*a2;

  Real e2 = SQR(e);
  Real e3 = e*e2;
  Real e4 = e*e3;

  Real em2 = SQR(em);
  Real em3 = em*em2;
  Real em4 = em*em3;
  Real em5 = em*em4;

  Real c3 = (-e/em2 + 14*e2/em3 - 36*e3/em4 + 24*e4/em5);
  Real c2 = (-e/em2 +  6*e2/em3 -  6*e3/em4) * 12;
  Real c1 = (-e/em2 +  2*e2/em3) * 36;
  Real c0 = (-e/em2) * 24;

  return c3*a3 + c2*a2 + c1*a + c0;
}

// a^4*exp(a) / (exp(a) - 1)^2
KOKKOS_INLINE_FUNCTION
Real RossMeanNumer(const Real a){
  if (a == 0) return 0;
  return SQR(SQR(a))*exp(a)/SQR(exp(a)-1.);
}

// d^4/da^4\left( a^4*exp(a) / (exp(a) - 1)^2 \right)
// a > 0
KOKKOS_INLINE_FUNCTION
Real d4RossMeanNumer(const Real a){
  Real e  = exp(a);
  Real em = e-1.;

  Real a2 = SQR(a);
  Real a3 = a*a2;
  Real a4 = a*a3;

  Real e2 = SQR(e);
  Real e3 = e*e2;
  Real e4 = e*e3;
  Real e5 = e*e4;

  Real em2 = SQR(em);
  Real em3 = em*em2;
  Real em4 = em*em3;
  Real em5 = em*em4;
  Real em6 = em*em5;

  Real c4 = e/em2 - 8*e2/em3;
  c4 += (-2*e/em3 +  6*e2/em4) * 6*e;
  c4 += (-2*e/em3 + 18*e2/em4 - 24*e3/em5) * 4*e;
  c4 += (-2*e/em3 + 42*e2/em4 - 144*e3/em5 + 120*e4/em6) * e;

  Real c3 = e/em2 -  6*e2/em3;
  c3 += (-2*e/em3 +  6*e2/em4) * 3*e;
  c3 += (-2*e/em3 + 18*e2/em4 - 24*e3/em5) * e;
  c3 *= 16;

  Real c2 = e/em2 - 4*e2/em3;
  c2 += (-2*e/em3 + 6*e2/em4) * e;
  c2 *= 72;

  Real c1 = 96 * (e/em2 - 2*e2/em3);
  Real c0 = 24 * e/em2;

  return c4*a4 + c3*a3 + c2*a2 + c1*a + c0;
}

// a^7*exp(2*a) / (exp(a) - 1)^3
KOKKOS_INLINE_FUNCTION
Real RossMeanDenom(const Real a){
  if (a == 0) return 0;
  Real a2 = SQR(a);
  Real a7 = SQR(a2)*a2*a;
  Real exp_a_m1 = exp(a)-1.;
  return a7*exp(2*a)/SQR(exp_a_m1)/exp_a_m1;
}

// d^4/da^4\left( a^7*exp(2*a) / (exp(a) - 1)^3 \right)
// a > 0
KOKKOS_INLINE_FUNCTION
Real d4RossMeanDenom(const Real a){
  Real e  = exp(a);
  Real em = e-1.;

  Real a3 = a*SQR(a);
  Real a4 = a*a3;
  Real a5 = a*a4;
  Real a6 = a*a5;
  Real a7 = a*a6;

  Real e2 = SQR(e);
  Real e3 = e*e2;
  Real e4 = e*e3;
  Real e5 = e*e4;
  Real e6 = e*e5;

  Real em2 = SQR(em);
  Real em3 = em*em2;
  Real em4 = em*em3;
  Real em5 = em*em4;
  Real em6 = em*em5;
  Real em7 = em*em6;

  Real c7 = 16*e2/em3 - 96*e3/em4;
  c7 += (-3*e/em4 + 12*e2/em5) * 24*e2;
  c7 += (-3*e/em4 + 36*e2/em5 -  60*e3/em6) * 8*e2;
  c7 += (-3*e/em4 + 84*e2/em5 - 360*e3/em6 + 360*e4/em7) * e2;

  Real c6 = 8*e2/em3 - 36*e3/em4;
  c6 += (-3*e/em4 + 12*e2/em5) * 6*e2;
  c6 += (-3*e/em4 + 36*e2/em5 - 60*e3/em6) * e2;
  c6 *= 28;

  Real c5 = 4*e2/em3 - 12*e3/em4;
  c5 += (-3*e/em4 + 12*e2/em5) * e2;
  c5 *= 252;

  Real c4 = 840 * (2*e2/em3 - 3*e3/em4);
  Real c3 = 840 * (e2/em3);

  return c7*a7 + c6*a6 + c5*a5 + c4*a4 + c3*a3;
}

// Numerical integration
// Simpson's 3/8 rule
KOKKOS_INLINE_FUNCTION
Real SimpsonIntegration(const Real al, const Real ar, Real err, int select) {
  int num_itr_max = 30; // corresponding to 7e-46 factor reduction in step size estimation

  // estimate step size
  Real num_div = 3;
  Real h = (ar-al)/num_div;
  Real a0 = al;
  Real a1 = al + h;
  Real a2 = ar - h;
  Real a3 = ar;
  Real d4f0, d4f1, d4f2, d4f3;
  if (select == 2) { // d^4/da^4\left( a^7*exp(2*a) / (exp(a) - 1)^3 \right)
    d4f0 = d4RossMeanDenom(a0);
    d4f1 = d4RossMeanDenom(a1);
    d4f2 = d4RossMeanDenom(a2);
    d4f3 = d4RossMeanDenom(a3);
  } else if (select == 1) { // d^4/da^4\left( a^4*exp(a) / (exp(a) - 1)^2 \right)
    d4f0 = d4RossMeanNumer(a0);
    d4f1 = d4RossMeanNumer(a1);
    d4f2 = d4RossMeanNumer(a2);
    d4f3 = d4RossMeanNumer(a3);
  } else { // select == 0, d^4/da^4\left( a^3 / (exp(a) - 1) \right)
    d4f0 = d4FFPlanckMeanDenom(a0);
    d4f1 = d4FFPlanckMeanDenom(a1);
    d4f2 = d4FFPlanckMeanDenom(a2);
    d4f3 = d4FFPlanckMeanDenom(a3);
  }
  Real d4f = fmax(fmax(fabs(d4f0), fabs(d4f1)), fmax(fabs(d4f2), fabs(d4f3)));
  for (int n=0; n<num_itr_max; n++) {
    if (3./80*h*SQR(SQR(h))*d4f > err) break;
    num_div *= 2;
    h = (ar-al)/num_div;
  }

  // Simpson's 3/8 rule
  Real ret = 0;
  for (int n=0; n<num_div/3; n++) { // for each segment
    a0 = al + n*(3*h);
    a1 = a0 + h;
    a2 = a1 + h;
    a3 = a2 + h;
    Real f0, f1, f2, f3;
    if (select == 2) { // a^7*exp(2*a) / (exp(a) - 1)^3
      f0 = RossMeanDenom(a0);
      f1 = RossMeanDenom(a1);
      f2 = RossMeanDenom(a2);
      f3 = RossMeanDenom(a3);
    } else if (select == 1) { // a^4*exp(a) / (exp(a) - 1)^2
      f0 = RossMeanNumer(a0);
      f1 = RossMeanNumer(a1);
      f2 = RossMeanNumer(a2);
      f3 = RossMeanNumer(a3);
    } else { // select == 0, a^3 / (exp(a) - 1)
      f0 = FFPlanckMeanDenom(a0);
      f1 = FFPlanckMeanDenom(a1);
      f2 = FFPlanckMeanDenom(a2);
      f3 = FFPlanckMeanDenom(a3);
    }
    ret += 3./8 * h * (f0 + 3*f1 + 3*f2 + f3);
  } // endfor n

  return ret;
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
