//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source.cpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"   // TransformToSRMHD, SingleC2P_IdealSRMHD (C2P W-limiter trial)
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

//----------------------------------------------------------------------------------------
//! C2P-based backtracking limiter.  A radiation<->gas exchange dU is admissible only if the REAL
//! GRMHD C2P (TransformToSRMHD -> SingleC2P_IdealSRMHD) recovers a physical, sub-W_limit state on
//! U_old + lambda*dU.  No proxy for the recovered W is used.  MHD-only (uses SingleC2P_IdealSRMHD).

// Sentinel W for a nonfinite trial C2P: > any physical W_limit yet finite, so it is rejected via
// W>w_limit and never seeds a NaN (fmax(NaN,0)=0 would otherwise masquerade as a healthy W=1).
namespace { constexpr Real kRadWBad = 1.0e30; }

//! Raw metric-correct Lorentz factor (BEFORE any gamma_max ceiling) + C2P status flags for a trial
//! gas conserved state with unchanged cell-centered B.  u is a LOCAL copy (C2P may floor it).
KOKKOS_INLINE_FUNCTION
Real RadTrialW(Real d, Real m1, Real m2, Real m3, Real e, Real bx, Real by, Real bz,
               Real glower[][4], Real gupper[][4], const EOS_Data &eos,
               bool &cfail, bool &dfl, bool &efl) {
  MHDCons1D u;
  u.d = d; u.mx = m1; u.my = m2; u.mz = m3; u.e = e; u.bx = bx; u.by = by; u.bz = bz;
  MHDCons1D u_sr;
  Real s2, b2, rpar;
  TransformToSRMHD(u, glower, gupper, s2, b2, rpar, u_sr);
  HydPrim1D w;
  int iter = 0;
  cfail = false; dfl = false; efl = false;
  SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, w, dfl, efl, cfail, iter);
  if (cfail || !isfinite(w.vx) || !isfinite(w.vy) || !isfinite(w.vz)) { return kRadWBad; }
  Real tmp = glower[1][1]*SQR(w.vx) + glower[2][2]*SQR(w.vy) + glower[3][3]*SQR(w.vz)
           + 2.0*(glower[1][2]*w.vx*w.vy + glower[1][3]*w.vx*w.vz + glower[2][3]*w.vy*w.vz);
  if (!isfinite(tmp)) { return kRadWBad; }
  return sqrt(1.0 + fmax(tmp, 0.0));               // raw W (pre-gamma_max), metric-correct
}

//! Admissible iff C2P succeeds, W finite and <= w_limit, and NO new density/energy floor is
//! introduced vs the lambda=0 baseline (b_df/b_ef).  A cell already on a floor stays admissible.
KOKKOS_INLINE_FUNCTION
bool RadAdmissible(Real lam, Real d, Real m1, Real m2, Real m3, Real e,
                   Real dm1, Real dm2, Real dm3, Real de,
                   Real bx, Real by, Real bz, Real glower[][4], Real gupper[][4],
                   const EOS_Data &eos, Real w_limit, bool b_df, bool b_ef) {
  bool cf, df, ef;
  Real W = RadTrialW(d, m1 + lam*dm1, m2 + lam*dm2, m3 + lam*dm3, e + lam*de,
                     bx, by, bz, glower, gupper, eos, cf, df, ef);
  if (cf || !isfinite(W) || W > w_limit) return false;
  if (df && !b_df) return false;                  // newly triggered density floor
  if (ef && !b_ef) return false;                  // newly triggered energy/entropy floor
  return true;
}

//! Largest lambda in [0,1] whose trial C2P is admissible.  The lambda=0 baseline floor flags
//! (b_df/b_ef) and the FULL lambda=1 trial (W1, cf1, df1, ef1) are computed ONCE by the caller and
//! passed in (2 C2P per substep on the common path, not 3).  If lambda=1 is rejected, halve to
//! bracket then bisect.  W(lambda) is not monotonic, so this is conservative backtracking.  Returns
//! 0 (freeze the whole exchange) if nothing short of no-exchange is admissible.
KOKKOS_INLINE_FUNCTION
Real RadBacktrackLambda(Real d, Real m1, Real m2, Real m3, Real e,
                        Real dm1, Real dm2, Real dm3, Real de,
                        Real bx, Real by, Real bz, Real glower[][4], Real gupper[][4],
                        const EOS_Data &eos, Real w_limit, bool b_df, bool b_ef,
                        Real W1, bool cf1, bool df1, bool ef1) {
  if (!(cf1 || !isfinite(W1) || W1 > w_limit || (df1 && !b_df) || (ef1 && !b_ef))) return 1.0;
  Real lam = 0.5, lam_acc = 0.0, lam_rej = 1.0;
  for (int it = 0; it < 8; ++it) {
    if (RadAdmissible(lam, d,m1,m2,m3,e, dm1,dm2,dm3,de, bx,by,bz,
                      glower,gupper,eos,w_limit,b_df,b_ef)) { lam_acc = lam; break; }
    lam_rej = lam; lam *= 0.5;
  }
  if (lam_acc <= 0.0) return 0.0;                  // fail closed
  Real lo = lam_acc, hi = lam_rej;                // [accepted, rejected]
  for (int it = 0; it < 16; ++it) {
    Real mid = 0.5*(lo + hi);
    if (RadAdmissible(mid, d,m1,m2,m3,e, dm1,dm2,dm3,de, bx,by,bz,
                      glower,gupper,eos,w_limit,b_df,b_ef)) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::RadFluidCoupling(Driver *pdriver, int stage)
//! \brief Add implicit radiation-fluid source terms.  Based on @c-white and @yanfeij's
//! gr_rad branch, radiation/coupling/emission.cpp commit be7f84565b.

TaskStatus Radiation::RadFluidCoupling(Driver *pdriver, int stage) {
  // Return if radiation source term disabled
  if (!(rad_source)) {
    return TaskStatus::complete;
  }

  // Extract indices, size data, hydro/mhd/units flags, and coupling flags
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = prgeo->nangles - 1;
  auto &size = pmy_pack->pmb->mb_size;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &are_units_enabled_ = are_units_enabled;
  bool &is_compton_enabled_ = is_compton_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  bool &affect_fluid_ = affect_fluid;

  // Extract flags and parameters for ad hoc fixes
  bool &correct_radsrc_velocity_ = correct_radsrc_velocity;
  bool &correct_radsrc_opacity_  = correct_radsrc_opacity;
  Real &dfloor_op   = dfloor_opacity;
  Real &dtrunc_max  = dens_trunc_max;
  Real &tau_trunc   = tau_truncation;
  Real &sigmoid_res = sigmoid_residual;

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  Real &n_0_floor_ = n_0_floor;

  // Extract radiation constant and units
  Real &arad_ = arad;
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  Real inv_t_electron_ = 1.0;
  if (are_units_enabled_) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
    inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
  }

  // Extract adiabatic index
  Real gm1, dfloor, v_sq_max;
  if (is_hydro_enabled_) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
    v_sq_max = 1.-1./SQR(pmy_pack->phydro->peos->eos_data.gamma_max);
    dfloor = pmy_pack->phydro->peos->eos_data.dfloor;
  } else if (is_mhd_enabled_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
    v_sq_max = 1.-1./SQR(pmy_pack->pmhd->peos->eos_data.gamma_max);
    dfloor = pmy_pack->pmhd->peos->eos_data.dfloor;
  }

  // C2P-based W limiter: flag + params + the EOS used by the trial C2P (MHD-only), captured by
  // value into the device kernel.  With rad_wlimit_ == false this whole path compiles to nothing.
  bool rad_wlimit_ = rad_wlimit && is_mhd_enabled_;
  Real fw_    = rad_wlimit_fw;
  Real whard_ = rad_w_hard;
  EOS_Data eos_ = (is_mhd_enabled_) ? pmy_pack->pmhd->peos->eos_data
                                    : (is_hydro_enabled_ ? pmy_pack->phydro->peos->eos_data
                                                         : EOS_Data());

  // Extract radiation, radiation frame, and radiation angular mesh data
  auto &i0_ = i0;
  Real &kappa_a_ = kappa_a;
  Real &kappa_s_ = kappa_s;
  Real &kappa_p_ = kappa_p;
  bool &power_opacity_ = power_opacity;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;

  // Extract hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }

  // Extract timestep
  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Call ConsToPrim over active zones prior to source term application
  DvceArray5D<Real> bcc0_;
  if (!(fixed_fluid_)) {
    if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      bcc0_ = pmy_pack->pmhd->bcc0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is,ie,js,je,ks,ke);
    }
  }

  // compute implicit source term
  par_for("radiation_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // compute metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);

    // fluid state
    Real &wdn = w0_(m,IDN,k,j,i);
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real &wen = w0_(m,IEN,k,j,i);

    // derived quantities
    Real pgas = gm1*wen;
    Real tgas = pgas/wdn;
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;

    // compute sigma_cold
    Real sigma_cold = 0.0;
    if (correct_radsrc_opacity_ && is_mhd_enabled_) {
      Real u1 = wvx - alpha * gamma * gupper[0][1];
      Real u2 = wvy - alpha * gamma * gupper[0][2];
      Real u3 = wvz - alpha * gamma * gupper[0][3];

      // lower vector indices
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // calculate 4-magnetic field
      auto &bccx = bcc0_(m,IBX,k,j,i);
      auto &bccy = bcc0_(m,IBY,k,j,i);
      auto &bccz = bcc0_(m,IBZ,k,j,i);
      Real b0_ = u_1*bccx + u_2*bccy + u_3*bccz;
      Real b1_ = (bccx + b0_ * u1) / u0;
      Real b2_ = (bccy + b0_ * u2) / u0;
      Real b3_ = (bccz + b0_ * u3) / u0;

      // lower vector indices
      Real b_0 = glower[0][0]*b0_ + glower[0][1]*b1_ + glower[0][2]*b2_ + glower[0][3]*b3_;
      Real b_1 = glower[1][0]*b0_ + glower[1][1]*b1_ + glower[1][2]*b2_ + glower[1][3]*b3_;
      Real b_2 = glower[2][0]*b0_ + glower[2][1]*b1_ + glower[2][2]*b2_ + glower[2][3]*b3_;
      Real b_3 = glower[3][0]*b0_ + glower[3][1]*b1_ + glower[3][2]*b2_ + glower[3][3]*b3_;
      Real b_sq = b0_*b_0 + b1_*b_1 + b2_*b_2 + b3_*b_3;

      sigma_cold = b_sq/wdn;
    } // endif (correct_radsrc_opacity_ && is_mhd_enabled_)

    // set opacities
    Real sigma_a, sigma_s, sigma_p;
    OpacityFunction(wdn, density_scale_,
                    tgas, temperature_scale_,
                    length_scale_, gm1, mean_mol_weight_,
                    power_opacity_, rosseland_coef_, planck_minus_rosseland_coef_,
                    kappa_a_, kappa_s_, kappa_p_,
                    sigma_a, sigma_s, sigma_p);

    // correct the density used for opacity setup
    Real wdn_opacity = fmax(wdn-dfloor, dfloor_op);
    if (correct_radsrc_opacity_) {
      if (excision_flux_(m,k,j,i)) {
        wdn_opacity = dfloor_op;
      } else {
      Real delta_l = fmax(fmax(size.d_view(m).dx1, size.d_view(m).dx2), size.d_view(m).dx3);
      Real dtrunc = fmax(0.0, sigma_cold)*tau_trunc / (kappa_s_*delta_l);
      dtrunc = fmin(dtrunc_max, fmax(dfloor, dtrunc)); // dfloor <= dtrunc <= dtrunc_max
      Real fac_trunc = dtrunc / dfloor;
      Real wid_trunc = 0.5*log10(fac_trunc) / log(1./sigmoid_res - 1.);
      Real wdn_real = fmax(wdn-dfloor, dfloor_op);
      Real del_reduce = log10(dfloor) - log10(dfloor_op);

      Real fac_inv = 1.0;
      if (fabs(fac_trunc-1) > 1e-12) {
        fac_inv = 1.0 + exp( -1./wid_trunc * ( log10(wdn_real) - (log10(dfloor) + 0.5*log10(fac_trunc)) ) );
      }

      Real lg_rho_op = log10(wdn_real) - (1.-1./fac_inv) * del_reduce;
      wdn_opacity = pow(10.0, lg_rho_op);
      } // endelse

      // apply the reduced density
      sigma_a *= wdn_opacity/wdn;
      sigma_s *= wdn_opacity/wdn;
      sigma_p *= wdn_opacity/wdn;
    } // endif correct_radsrc_opacity_

    Real dtcsiga = dt_*sigma_a;
    Real dtcsigs = dt_*sigma_s;
    Real dtcsigp = dt_*sigma_p;
    Real dtaucsiga = dtcsiga/u0;
    Real dtaucsigs = dtcsigs/u0;
    Real dtaucsigp = dtcsigp/u0;

    // compute fluid velocity in tetrad frame
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

    // coordinate component n^0
    Real n0 = tt(m,0,0,k,j,i);

    // Correct velocity in radiation-dominated regime (for details, see White 2023)
    if (correct_radsrc_velocity_) {
      // NOTE(@lzhang): In radiation-dominated regime, the gas-radiation momentum
      // coupling without accounting the change of the gas velocity can
      // result in an overestimated high gas temperature because the overestimated
      // velocity leads to the low density. This turns out to be extremely dangerous
      // since it can generate enormous radiation through the thermal coupling or
      // Compton process.

      // calculate radiation energy density in fluid frame
      Real erad_f_ = 0.0;
      Real omega_hat_tot = 0.0; Real omega_cm_tot = 0.0;
      for (int n=0; n<=nang1; ++n) {
        // compute coordinate normal components
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);

        // compute quantites in fluid frame
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));

        Real i0_cm = i0_(m,n,k,j,i)/(n0*n_0)*SQR(SQR(n0_cm));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        omega_hat_tot += solid_angles_.d_view(n);
        omega_cm_tot += omega_cm;
        erad_f_ += i0_cm*omega_cm;
      }
      erad_f_ *= omega_hat_tot/omega_cm_tot;

      // apply velocity correction if radiation is dominated
      if (erad_f_ > wdn + wen) {
        // compute radiation moments in terad frame
        Real rr_tet00 = 0.0;
        Real rr_tet01 = 0.0; Real rr_tet02 = 0.0; Real rr_tet03 = 0.0;
        Real rr_tet11 = 0.0; Real rr_tet22 = 0.0; Real rr_tet33 = 0.0;
        Real rr_tet12 = 0.0; Real rr_tet13 = 0.0; Real rr_tet23 = 0.0;
        for (int n=0; n<=nang1; ++n) {
          // tetrad normal components
          Real nh0 = nh_c_.d_view(n,0);
          Real nh1 = nh_c_.d_view(n,1);
          Real nh2 = nh_c_.d_view(n,2);
          Real nh3 = nh_c_.d_view(n,3);

          // coordinate normal components
          Real n_0 = tc(m,0,0,k,j,i)*nh0 + tc(m,1,0,k,j,i)*nh1 + tc(m,2,0,k,j,i)*nh2 + tc(m,3,0,k,j,i)*nh3;
          Real n_1 = tc(m,0,1,k,j,i)*nh0 + tc(m,1,1,k,j,i)*nh1 + tc(m,2,1,k,j,i)*nh2 + tc(m,3,1,k,j,i)*nh3;
          Real n_2 = tc(m,0,2,k,j,i)*nh0 + tc(m,1,2,k,j,i)*nh1 + tc(m,2,2,k,j,i)*nh2 + tc(m,3,2,k,j,i)*nh3;
          Real n_3 = tc(m,0,3,k,j,i)*nh0 + tc(m,1,3,k,j,i)*nh1 + tc(m,2,3,k,j,i)*nh2 + tc(m,3,3,k,j,i)*nh3;

          // radiation moments in terad frame
          rr_tet00 += (        i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet01 += (    nh1*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet02 += (    nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet03 += (    nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet11 += (nh1*nh1*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet22 += (nh2*nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet33 += (nh3*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet12 += (nh1*nh2*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet13 += (nh1*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
          rr_tet23 += (nh2*nh3*i0_(m,n,k,j,i)/(n0*n_0)*solid_angles_.d_view(n));
        } // endfor n

        // calculate radiation velocity in tetrad frame
        Real vrad_tet1 = rr_tet01 / rr_tet00;
        Real vrad_tet2 = rr_tet02 / rr_tet00;
        Real vrad_tet3 = rr_tet03 / rr_tet00;
        Real vrad_sq = SQR(vrad_tet1) + SQR(vrad_tet2) + SQR(vrad_tet3);
        if (vrad_sq > v_sq_max) {
          Real ratio = sqrt(v_sq_max / vrad_sq);
          vrad_tet1 *= ratio;
          vrad_tet2 *= ratio;
          vrad_tet3 *= ratio;
          vrad_sq = v_sq_max;
        }
        Real urad_tet0 = 1.0 / sqrt(1.0 - vrad_sq);
        Real urad_tet1 = urad_tet0 * vrad_tet1;
        Real urad_tet2 = urad_tet0 * vrad_tet2;
        Real urad_tet3 = urad_tet0 * vrad_tet3;

        // calculate current fluid momentum
        Real wgas = wdn + wen + pgas;
        Real mgas_tet1 = wgas * u_tet[0] * u_tet[1];
        Real mgas_tet2 = wgas * u_tet[0] * u_tet[2];
        Real mgas_tet3 = wgas * u_tet[0] * u_tet[3];

        // calculate fluid momentum if accelerated to radiation frame
        Real mgas_rad_tet1 = wgas * urad_tet0 * urad_tet1;
        Real mgas_rad_tet2 = wgas * urad_tet0 * urad_tet2;
        Real mgas_rad_tet3 = wgas * urad_tet0 * urad_tet3;

        // calculate the gas-radiation coupling force
        // Use the same local inverse-length coefficients as the source update. These
        // include unit conversion, power-law opacity, and opacity correction.
        Real chi_p = sigma_a + sigma_p;
        Real chi_s = sigma_s;
        Real chi_a = sigma_a + sigma_s;
        Real emissivity = chi_p*arad_*SQR(SQR(tgas)) + chi_s*erad_f_;
        if (is_compton_enabled_) {
          Real trad = sqrt(sqrt(erad_f_/arad_));
          emissivity += chi_s*4*(tgas-trad)*inv_t_electron_*erad_f_;
        }
        Real gg_tet1 = -emissivity*u_tet[1] - chi_a*(-u_tet[0]*rr_tet01 + u_tet[1]*rr_tet11 + u_tet[2]*rr_tet12 + u_tet[3]*rr_tet13);
        Real gg_tet2 = -emissivity*u_tet[2] - chi_a*(-u_tet[0]*rr_tet02 + u_tet[1]*rr_tet12 + u_tet[2]*rr_tet22 + u_tet[3]*rr_tet23);
        Real gg_tet3 = -emissivity*u_tet[3] - chi_a*(-u_tet[0]*rr_tet03 + u_tet[1]*rr_tet13 + u_tet[2]*rr_tet23 + u_tet[3]*rr_tet33);

        // estimate change in fluid momentum from source terms
        Real dmgas_tet1 = gg_tet1 * dt_ / u_tet[0];
        Real dmgas_tet2 = gg_tet2 * dt_ / u_tet[0];
        Real dmgas_tet3 = gg_tet3 * dt_ / u_tet[0];

        // estimate new fluid velocity
        Real frac1 = (mgas_rad_tet1==mgas_tet1) ? 0.0 : dmgas_tet1 / (mgas_rad_tet1 - mgas_tet1);
        Real frac2 = (mgas_rad_tet2==mgas_tet2) ? 0.0 : dmgas_tet2 / (mgas_rad_tet2 - mgas_tet2);
        Real frac3 = (mgas_rad_tet3==mgas_tet3) ? 0.0 : dmgas_tet3 / (mgas_rad_tet3 - mgas_tet3);
        frac1 = fmin(fmax(frac1, 0.0), 1.0);
        frac2 = fmin(fmax(frac2, 0.0), 1.0);
        frac3 = fmin(fmax(frac3, 0.0), 1.0);
        u_tet[1] = (1.0-frac1)*u_tet[1] + frac1*urad_tet1;
        u_tet[2] = (1.0-frac2)*u_tet[2] + frac2*urad_tet2;
        u_tet[3] = (1.0-frac3)*u_tet[3] + frac3*urad_tet3;
        u_tet[0] = sqrt(1.0 + SQR(u_tet[1]) + SQR(u_tet[2]) + SQR(u_tet[3]));
      } // endif (erad_f_ > wdn + wen)
    } // endif (correct_radsrc_velocity_)

    // Calculate polynomial coefficients
    Real wght_sum = 0.0;
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    // denom == (1 - suma3), accumulated DIRECTLY as a sum of positive terms to avoid the
    // catastrophic cancellation "1 - suma3" when scattering is enormous (suma3 -> 1).
    Real denom = 0.0;
    for (int n=0; n<=nang1; ++n) {
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                 tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                    u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
      Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
      Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
      Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
      Real vncsigma2 = n0_cm*vncsigma;
      Real ir_weight = intensity_cm*omega_cm;
      wght_sum += omega_cm;
      suma1 += omega_cm*vncsigma2;
      suma2 += ir_weight*n0*vncsigma;
      denom += omega_cm*(n0 + (dtcsiga + dtcsigp)*n0_cm)*vncsigma;   // == (1 - suma3) per angle
    }
    suma1 /= wght_sum;
    suma2 /= wght_sum;
    denom /= wght_sum;                     // == 1 - suma3, cancellation-free
    suma1 *= (dtcsiga + dtcsigp);

    // Reject the source update ONLY if the implicit denominator is degenerate (non-finite or
    // non-positive) -- leaves the cell unchanged.  A small-but-positive denom is PHYSICAL (stiff
    // scattering: denom ~ 1/(1+dt*sigma_s)); the direct sum recovers it where 1-suma3 rounded to 0.
    bool badcell = (!isfinite(denom) || (denom <= 0.0));

    // compute coefficients
    Real coef[2] = {0.0, 0.0};
    Real tgasnew = tgas;
    // Fraction of the absorption exchange actually accepted this cell (0 if skipped / failed closed,
    // <1 if backtracked).  Compton uses tgas + lam_abs*(tgasnew - tgas), not the full lambda=1 T.
    Real lam_abs = 0.0;
    if (!(badcell)) {
      coef[1] = (dtaucsiga+dtaucsigp-(dtaucsiga+dtaucsigp)*suma1/denom)*arad_*gm1/wdn;
      coef[0] = -tgas-(dtaucsiga+dtaucsigp)*suma2*gm1/(wdn*denom);

      // Calculate new gas temperature
      if (fabs(coef[1]) > 1.0e-20) {
        bool flag = FourthPolyRoot(coef[1], coef[0], tgasnew);
        if (!(flag) || !(isfinite(tgasnew))) {
          badcell = true;
          tgasnew = tgas;
        }
      } else {
        tgasnew = -coef[0];
      }
    }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      Real jr_cm = (suma1*emission + suma2)/denom;

      // ---- pass A: accumulate the FULL proposed exchange dU = m_old - m_new, WITHOUT writing i0.
      //      Flag nonfinite i0_upd/dU so a degenerate update fails CLOSED (no NaN, no 0*NaN). ----
      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      bool bad_exchange = false;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
        Real i0_old = i0_(m,n,k,j,i);
        m_old[0] += (    i0_old    *solid_angles_.d_view(n));
        m_old[1] += (n_1*i0_old/n_0*solid_angles_.d_view(n));
        m_old[2] += (n_2*i0_old/n_0*solid_angles_.d_view(n));
        m_old[3] += (n_3*i0_old/n_0*solid_angles_.d_view(n));
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_old/(n0*n_0))*SQR(SQR(n0_cm));
        Real vncsigma2 = n0_cm/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm + (dtcsiga+dtcsigp)*emission
                      - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
        Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
        bad_exchange = bad_exchange || !isfinite(i0_upd);
        m_new[0] += (    i0_upd    *solid_angles_.d_view(n));
        m_new[1] += (n_1*i0_upd/n_0*solid_angles_.d_view(n));
        m_new[2] += (n_2*i0_upd/n_0*solid_angles_.d_view(n));
        m_new[3] += (n_3*i0_upd/n_0*solid_angles_.d_view(n));
      }
      Real dE = m_old[0]-m_new[0], dM1 = m_old[1]-m_new[1],
           dM2 = m_old[2]-m_new[2], dM3 = m_old[3]-m_new[3];
      if (!isfinite(dE)||!isfinite(dM1)||!isfinite(dM2)||!isfinite(dM3)) bad_exchange = true;

      // ---- decide lambda: fail closed if pass A invalid; else backtrack via the REAL C2P ----
      Real lam;
      if (bad_exchange) {
        lam = 0.0;
      } else {
        lam = 1.0;
        if (affect_fluid_ && rad_wlimit_) {
          Real w_limit = fmax(gamma, fmin(whard_, fw_*gamma));
          Real D = u0_(m,IDN,k,j,i), M1 = u0_(m,IM1,k,j,i), M2 = u0_(m,IM2,k,j,i),
               M3 = u0_(m,IM3,k,j,i), E = u0_(m,IEN,k,j,i);
          Real bx = bcc0_(m,IBX,k,j,i), by = bcc0_(m,IBY,k,j,i), bz = bcc0_(m,IBZ,k,j,i);
          bool cf0,df0,ef0,cf1,df1,ef1;
          RadTrialW(D, M1, M2, M3, E, bx,by,bz, glower,gupper,eos_, cf0,df0,ef0);          // baseline
          Real W1 = RadTrialW(D, M1+dM1, M2+dM2, M3+dM3, E+dE, bx,by,bz,                   // lambda=1
                              glower,gupper,eos_, cf1,df1,ef1);
          lam = RadBacktrackLambda(D, M1, M2, M3, E, dM1, dM2, dM3, dE, bx,by,bz,
                                   glower,gupper,eos_, w_limit, df0,ef0, W1,cf1,df1,ef1);
          // HARD backstop: never let an out-of-range/nonfinite lambda multiply dU; fail closed if
          // the baseline C2P (cf0) was untrustworthy.
          if (cf0 || !isfinite(lam) || lam < 0.0 || lam > 1.0) { lam = 0.0; bad_exchange = true; }
        }
      }
      lam_abs = bad_exchange ? 0.0 : lam;   // accepted absorption fraction (for Compton's tgas)

      // ---- pass B: apply the lambda-blended exchange (SKIP entirely if bad_exchange) ----
      if (!bad_exchange) {
        for (int n=0; n<=nang1; ++n) {
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real i0_old = i0_(m,n,k,j,i);
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real intensity_cm = 4.0*M_PI*(i0_old/(n0*n_0))*SQR(SQR(n0_cm));
          Real vncsigma2 = n0_cm/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm + (dtcsiga+dtcsigp)*emission
                        - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
          Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
          i0_(m,n,k,j,i) = i0_old + lam*(i0_upd - i0_old);
          if (excise) {
            bool apply_excision = (rad_mask_(m,k,j,i) ||
                                   (!(is_compton_enabled_) && fabs(n_0) < n_0_floor_));
            if (apply_excision) { i0_(m,n,k,j,i) = 0.0; }
          }
        }
        if (affect_fluid_) {
          u0_(m,IEN,k,j,i) += lam*dE;
          u0_(m,IM1,k,j,i) += lam*dM1;
          u0_(m,IM2,k,j,i) += lam*dM2;
          u0_(m,IM3,k,j,i) += lam*dM3;
        }
      }
    }

    // compton scattering
    if (is_compton_enabled_) {
      // use the CONSISTENT partially-updated gas temperature: only lam_abs of the absorption
      // exchange was accepted, so blend rather than taking the full lambda=1 solution tgasnew.
      tgas = tgas + lam_abs*(tgasnew - tgas);

      // compute polynomial coefficients using partially updated gas temp and intensity
      suma1 = 0.0;
      Real jr_cm = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                   tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real wght_cm = solid_angles_.d_view(n)/SQR(n0_cm)/wght_sum;
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real ir_weight = intensity_cm*wght_cm;
        jr_cm += ir_weight;
        suma1 += (n0_cm/n0)*4.0*dtcsigs*inv_t_electron_*wght_cm;
      }
      suma2 = 4.0*dtaucsigs*inv_t_electron_*gm1/wdn;

      // compute partially updated radiation temperature
      Real trad = sqrt(sqrt(jr_cm/arad_));
      const bool temp_equil = (fabs(trad - tgas) < 1.0e-12);

      // Calculate new gas temperature due to Compton
      Real tradnew = trad;
      badcell = false;
      if (!(temp_equil)) {
        coef[1] = (1.0 + suma2*jr_cm)/(suma1*jr_cm)*arad_;
        coef[0] = -(1.0 + suma2*jr_cm)/suma1 - tgas;
        bool flag = FourthPolyRoot(coef[1], coef[0], tradnew);
        if (!(flag) || !(isfinite(tradnew))) {
          badcell = true;
        }
      }

      // Update the specific intensity
      if (!(badcell) && !(temp_equil)) {
        // Compute updated gas temperature
        tgasnew = (arad_*SQR(SQR(tradnew)) - jr_cm)/(suma1*jr_cm) + tradnew;

        // ---- pass A: full Compton exchange dU = m_old - m_new, WITHOUT writing i0 ----
        Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
        bool bad_exchange = false;
        for (int n=0; n<=nang1; ++n) {
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);
          Real i0_old = i0_(m,n,k,j,i);
          m_old[0] += (    i0_old    *solid_angles_.d_view(n));
          m_old[1] += (n_1*i0_old/n_0*solid_angles_.d_view(n));
          m_old[2] += (n_2*i0_old/n_0*solid_angles_.d_view(n));
          m_old[3] += (n_3*i0_old/n_0*solid_angles_.d_view(n));
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
          Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
          bad_exchange = bad_exchange || !isfinite(i0_upd);
          m_new[0] += (    i0_upd    *solid_angles_.d_view(n));
          m_new[1] += (n_1*i0_upd/n_0*solid_angles_.d_view(n));
          m_new[2] += (n_2*i0_upd/n_0*solid_angles_.d_view(n));
          m_new[3] += (n_3*i0_upd/n_0*solid_angles_.d_view(n));
        }
        Real dE = m_old[0]-m_new[0], dM1 = m_old[1]-m_new[1],
             dM2 = m_old[2]-m_new[2], dM3 = m_old[3]-m_new[3];
        if (!isfinite(dE)||!isfinite(dM1)||!isfinite(dM2)||!isfinite(dM3)) bad_exchange = true;

        // ---- decide lambda (same W_limit as absorption; cumulative bound over both substeps) ----
        Real lam;
        if (bad_exchange) {
          lam = 0.0;
        } else {
          lam = 1.0;
          if (affect_fluid_ && rad_wlimit_) {
            Real w_limit = fmax(gamma, fmin(whard_, fw_*gamma));
            Real D = u0_(m,IDN,k,j,i), M1 = u0_(m,IM1,k,j,i), M2 = u0_(m,IM2,k,j,i),
                 M3 = u0_(m,IM3,k,j,i), E = u0_(m,IEN,k,j,i);
            Real bx = bcc0_(m,IBX,k,j,i), by = bcc0_(m,IBY,k,j,i), bz = bcc0_(m,IBZ,k,j,i);
            bool cf0,df0,ef0,cf1,df1,ef1;
            RadTrialW(D, M1, M2, M3, E, bx,by,bz, glower,gupper,eos_, cf0,df0,ef0);
            Real W1 = RadTrialW(D, M1+dM1, M2+dM2, M3+dM3, E+dE, bx,by,bz,
                                glower,gupper,eos_, cf1,df1,ef1);
            lam = RadBacktrackLambda(D, M1, M2, M3, E, dM1, dM2, dM3, dE, bx,by,bz,
                                     glower,gupper,eos_, w_limit, df0,ef0, W1,cf1,df1,ef1);
            if (cf0 || !isfinite(lam) || lam < 0.0 || lam > 1.0) { lam = 0.0; bad_exchange = true; }
          }
        }

        // ---- pass B: apply the lambda-blended exchange (SKIP entirely if bad_exchange) ----
        if (!bad_exchange) {
          for (int n=0; n<=nang1; ++n) {
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                     + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            Real i0_old = i0_(m,n,k,j,i);
            Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                          u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
            Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
            Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
            i0_(m,n,k,j,i) = i0_old + lam*(i0_upd - i0_old);
            if (excise) {
              if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
            }
          }
          if (affect_fluid_) {
            u0_(m,IEN,k,j,i) += lam*dE;
            u0_(m,IM1,k,j,i) += lam*dM1;
            u0_(m,IM2,k,j,i) += lam*dM2;
            u0_(m,IM3,k,j,i) += lam*dM3;
          }
        }
      } else {
        // NOTE(@pdmullen): At this point, it is possible that excision has not been
        // entirely applied if Compton is enabled and a badcell or temperature equilibrium
        // was encountered.. apply excision
        if (excise) {
          for (int n=0; n<=nang1; ++n) {
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+
                       tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)+
                       tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+
                       tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

} // namespace radiation
