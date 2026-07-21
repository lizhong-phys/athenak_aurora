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
  bool &limit_opacity_ = limit_opacity;
  Real &xi_max_ = opacity_xi_max;
  Real &tcap_   = opacity_tcap;
  bool &rad_dvlimit_ = rad_dvlimit;
  Real &lambda_lock_   = rad_momentum_lambda_lock;
  Real &rad_dom_lock_  = rad_dominance_lock;
  Real &fw_            = rad_max_w_increase;
  Real &w_hard_        = rad_w_hard;

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

  // Extract adiabatic index + floors (entropy floor params feed the opacity limiter)
  Real gm1, dfloor, v_sq_max, gamma_max_ = 1.0e30;
  Real sfloor_ = 1.0e-30, sfloor1_ = 1.0e-30, sfloor2_ = 1.0e-30, rho1_ = 10.0, rho2_ = 1.0e3;
  if (is_hydro_enabled_) {
    auto &ed = pmy_pack->phydro->peos->eos_data;
    gm1 = ed.gamma - 1.0;  gamma_max_ = ed.gamma_max;  v_sq_max = 1.-1./SQR(gamma_max_);
    dfloor = ed.dfloor;
    sfloor_ = ed.sfloor;  sfloor1_ = ed.sfloor1;  sfloor2_ = ed.sfloor2;
    rho1_ = ed.rho1;  rho2_ = ed.rho2;
  } else if (is_mhd_enabled_) {
    auto &ed = pmy_pack->pmhd->peos->eos_data;
    gm1 = ed.gamma - 1.0;  gamma_max_ = ed.gamma_max;  v_sq_max = 1.-1./SQR(gamma_max_);
    dfloor = ed.dfloor;
    sfloor_ = ed.sfloor;  sfloor1_ = ed.sfloor1;  sfloor2_ = ed.sfloor2;
    rho1_ = ed.rho1;  rho2_ = ed.rho2;
  }

  // Extract radiation, radiation frame, and radiation angular mesh data
  auto &i0_ = i0;
  auto ldiag = limiter_diag;   // per-cell limiter diagnostics (written only if limit_opacity)
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

    // --- opacity-aware stiffness limiter (EMERGENCY STABILIZER / opacity closure) ---
    // On cells collapsed onto the density-dependent entropy floor, the Planck absorption
    // (sigma_a+sigma_p ~ rho^2 T^-7/2) blows up and drives the density->cold->opaque->
    // overshoot feedback.  Cap it at the opacity the gas would have at
    //   T_eff = tgas * max[1, (Xi_abs/Xi_max)^(2/7)]   (capped at the virial tcap),
    //   Xi_abs = dt*(sigma_a+sigma_p)/u0  (proper-time absorption stiffness).
    // Rescale sigma_a,sigma_p (both ~T^-7/2); leave sigma_s (scattering) unchanged.
    // NOTE: Xi_abs contains dt -> this is a timestep-dependent opacity closure, NOT a
    // physical heating model (real heating is applied separately in C2P).  It also breaks
    // exact opacity/source-function consistency while active (emission still uses tgas).
    if (limit_opacity_ && power_opacity_) {   // 2/7 scaling is only valid for Kramers opacity
      Real xi_abs = dt_*(sigma_a + sigma_p)/u0;           // raw absorption stiffness
      Real lg_sfl = log10(sfloor1_) + (log10(wdn)-log10(rho1_))
                  *(log10(sfloor2_)-log10(sfloor1_))/(log10(rho2_)-log10(rho1_));
      Real sfloor_local = fmax(sfloor_, pow(10.0, lg_sfl));
      Real t_entropy = sfloor_local*pow(wdn, gm1);        // entropy-floor T (~rho^0.928)
      bool on_floor = (tgas < 1.02*t_entropy);
      Real t_eff = tgas; bool active = false; bool hit_tcap = false;
      if (isfinite(xi_abs) && on_floor && (xi_abs > xi_max_)) {
        Real t_req = tgas*pow(xi_abs/xi_max_, 2.0/7.0);   // > tgas (since xi_abs > xi_max)
        hit_tcap = (t_req > tcap_);
        // clamp into [tgas, tcap] so we can only LOWER opacity (fac<=1), never raise it,
        // even if tcap is misconfigured below tgas.
        t_eff = fmax(tgas, fmin(t_req, tcap_));
        Real fac = pow(tgas/t_eff, 3.5);                  // <= 1
        sigma_a *= fac;
        sigma_p *= fac;
        active = true;
      }
      // record per-cell diagnostics (array is allocated iff limit_opacity)
      ldiag(m,0,k,j,i) = on_floor ? 1.0 : 0.0;
      ldiag(m,1,k,j,i) = xi_abs;                          // raw
      ldiag(m,2,k,j,i) = dt_*(sigma_a + sigma_p)/u0;      // limited
      ldiag(m,3,k,j,i) = t_eff;
      ldiag(m,4,k,j,i) = active   ? 1.0 : 0.0;
      ldiag(m,5,k,j,i) = hit_tcap ? 1.0 : 0.0;
      ldiag(m,6,k,j,i) = sigma_a;
      ldiag(m,7,k,j,i) = sigma_p;
      ldiag(m,8,k,j,i) = sigma_s;
    }

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

    // Density-drop limiter state for this cell.  dvlimit_lambda: exchange scale, 1 = untouched
    // (normal case => source update below is bit-for-bit unchanged), <1 only in the pathological
    // radiation-dominated stiff regime.  dvlimit_gate: whether the stiffness*dominance gate is
    // open (lambda then decided from the ACTUAL exchange in the coupling loop).  dvlimit_lambda_s:
    // recorded Lambda_s (diag slot 15; 0 when the radiation-dominance gate below did not fire).
    Real dvlimit_lambda = 1.0;         // scale for the absorption/scattering exchange
    Real dvlimit_lambda_c = 1.0;       // scale for the Compton exchange (shares the SAME budget)
    Real dvlimit_budget_sq = 0.0;      // |S|^2 that recovers W = W_limit (max allowed post |S|^2)
    bool dvlimit_gate = false;
    Real dvlimit_lambda_s = 0.0;       // diag: Lambda_s (stiffness*dominance)
    Real dvlimit_wpost = 0.0;          // diag: recovered W_post at lambda=1
    Real dvlimit_wlim  = 0.0;          // diag: W_limit

    // Radiation-frame estimate + velocity correction (White 2023).  The frame estimate and
    // its diagnostics (limiter_diag slots 9-14) are computed whenever the diag array exists
    // (limit_opacity || correct_radsrc_velocity || rad_dvlimit), so a correct_radsrc_velocity=
    // false CONTROL run still records what the correction WOULD do.  The actual velocity change
    // is applied ONLY when correct_radsrc_velocity (it may HURT -- test on/off separately).
    if (correct_radsrc_velocity_ || limit_opacity_ || rad_dvlimit_) {
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
        // Defensive: skip grazing angles (|n_0| < n_0_floor) from the frame moments.  Their
        // 1/n_0 weight diverges; RKUpdate already zeros these intensities in transport, but a
        // zero over a near-zero n_0 can still give 0/0=NaN here.  This is a guard, NOT a claim
        // that grazing angles caused any particular frame value.
        if (fabs(n_0) < n_0_floor_) { continue; }

        // compute quantites in fluid frame
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));

        Real i0_cm = i0_(m,n,k,j,i)/(n0*n_0)*SQR(SQR(n0_cm));
        Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
        omega_hat_tot += solid_angles_.d_view(n);
        omega_cm_tot += omega_cm;
        erad_f_ += i0_cm*omega_cm;
      }
      // guard the normalization: if every angle was skipped/degenerate, treat as no radiation
      erad_f_ = (omega_cm_tot > 0.0) ? erad_f_*omega_hat_tot/omega_cm_tot : 0.0;

      // diagnostics for the velocity correction (recorded into limiter_diag slots 9-14)
      Real rho_h = wdn + wen + pgas;
      bool vc_gate = (erad_f_ > wdn + wen);
      Real d_uradW = -1.0, d_vradsq = -1.0, d_rr00 = -1.0;

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
          // Defensive: skip grazing angles (|n_0| < n_0_floor) -- see the erad_f_ loop above.
          // Guard against 0/0 from a transport-zeroed intensity over a near-zero n_0; not a
          // claim that these caused the observed vrad^2 (which may be physical for radiation
          // isotropic in a fast-moving gas frame).
          if (fabs(n_0) < n_0_floor_) { continue; }

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

        // calculate radiation velocity in tetrad frame.  Guard against a degenerate radiation
        // energy (rr_tet00 <= 0 or non-finite): the frame is then undefined, so fall back to the
        // gas rest frame (vrad = 0 => no velocity relaxation) rather than divide by it.
        bool bad_frame = !(rr_tet00 > 0.0) || !isfinite(rr_tet00);
        Real vrad_tet1 = 0.0, vrad_tet2 = 0.0, vrad_tet3 = 0.0, vrad_sq = 0.0;
        if (!bad_frame) {
          vrad_tet1 = rr_tet01 / rr_tet00;
          vrad_tet2 = rr_tet02 / rr_tet00;
          vrad_tet3 = rr_tet03 / rr_tet00;
          vrad_sq = SQR(vrad_tet1) + SQR(vrad_tet2) + SQR(vrad_tet3);
        }
        d_vradsq = vrad_sq;                 // RAW (pre-clamp) radiation-frame velocity^2
        d_rr00 = rr_tet00;                  // tetrad radiation energy (grazing-angle check)
        if (vrad_sq > v_sq_max) {
          Real ratio = sqrt(v_sq_max / vrad_sq);
          vrad_tet1 *= ratio;
          vrad_tet2 *= ratio;
          vrad_tet3 *= ratio;
          vrad_sq = v_sq_max;
        }
        Real urad_tet0 = 1.0 / sqrt(1.0 - vrad_sq);
        d_uradW = urad_tet0;                // inferred radiation-frame Lorentz factor
        Real urad_tet1 = urad_tet0 * vrad_tet1;
        Real urad_tet2 = urad_tet0 * vrad_tet2;
        Real urad_tet3 = urad_tet0 * vrad_tet3;

        // Density-drop limiter and velocity correction (both need the gas enthalpy; the
        // correction additionally needs the tetrad momentum + radiation force).
        if (correct_radsrc_velocity_ || rad_dvlimit_) {
          Real wgas = wdn + wen + pgas;

          // ---- density-drop limiter: only OPEN the gate here (lambda is decided in the -----
          // coupling loop below from the ACTUAL implicit moment exchange, with the FINAL
          // post-correction u_tet, so the explicit force estimate is not used).  Gate:
          // Lambda_s = (dt*sigma_s/u0)*(E_rad_f/rho_h) = stiffness * radiation dominance.
          if (rad_dvlimit_) {
            Real r_rad = erad_f_ / fmax(wgas, 1.0e-300);      // radiation dominance E_rad_f/(rho h)
            dvlimit_lambda_s = dtaucsigs * r_rad;             // Lambda_s = (dt sigma_s/u0) R_rad
            dvlimit_gate = (r_rad > rad_dom_lock_) && (dvlimit_lambda_s > lambda_lock_);
          }

          // ---- velocity correction: implicit relaxation of the fluid velocity toward the ----
          // radiation rest frame (urad), applied only if enabled.  Replaces the White (2023)
          // componentwise frac blend, which relaxes each axis by a different, force-based,
          // sign-clampable fraction and so leaves u_tet != urad -- the residual, times the huge
          // E_rad, is the scattering momentum kick that overshoots the gas to the W-ceiling.
          //
          // The radiation drag obeys dv/dt = -k (v - v_rad); its EXACT implicit-in-dt solution
          // moves v a fraction theta = dt*k/(1+dt*k) of the way to v_rad, with a SINGLE scalar
          // theta in [0,1] (no per-axis sign traps).  The proper-time drag stiffness dt*k is
          // exactly Lambda_s = (dt*sigma_s/u0)*(E_rad_f/rho_h).  Stiff cell (Lambda_s>>1) =>
          // theta->1 => u_tet->urad => the comoving frame IS the radiation flux-frame, so the
          // scattering feedback (m_old-m_new) vanishes and the gas coasts at its bounded
          // pre-radiation velocity instead of being flung to gamma_max.  Weak coupling
          // (Lambda_s<<1) => theta->0 => unchanged.
          if (correct_radsrc_velocity_) {
            Real lambda_s = dtaucsigs * erad_f_ / fmax(wgas, 1.0e-300);
            Real theta = lambda_s / (1.0 + lambda_s);       // implicit relaxation fraction, 0..1
            u_tet[1] = (1.0-theta)*u_tet[1] + theta*urad_tet1;
            u_tet[2] = (1.0-theta)*u_tet[2] + theta*urad_tet2;
            u_tet[3] = (1.0-theta)*u_tet[3] + theta*urad_tet3;
            u_tet[0] = sqrt(1.0 + SQR(u_tet[1]) + SQR(u_tet[2]) + SQR(u_tet[3]));
            // urad is already sub-gamma_max (vrad_sq clamp), and a convex blend of two
            // sub-gamma_max states is sub-gamma_max, so no further cap is needed; keep a guard.
            if (u_tet[0] > gamma_max_) {
              Real rescale = sqrt((SQR(gamma_max_) - 1.0)/fmax(SQR(u_tet[0]) - 1.0, 1.0e-30));
              u_tet[1] *= rescale;
              u_tet[2] *= rescale;
              u_tet[3] *= rescale;
              u_tet[0] = gamma_max_;
            }
          } // endif (correct_radsrc_velocity_)
        } // endif (correct_radsrc_velocity_ || rad_dvlimit_)
      } // endif (erad_f_ > wdn + wen)
      // record velocity-correction diagnostics (array allocated iff limit_opacity,
      // correct_radsrc_velocity, or rad_dvlimit; d_* are -1 when the gate did not fire)
      ldiag(m, 9,k,j,i) = erad_f_;
      ldiag(m,10,k,j,i) = rho_h;
      ldiag(m,11,k,j,i) = vc_gate ? 1.0 : 0.0;
      ldiag(m,12,k,j,i) = d_uradW;      // inferred radiation-frame W (=gamma_max if clamped)
      ldiag(m,13,k,j,i) = d_vradsq;     // raw radiation vrad^2 (>v_sq_max => superluminal)
      ldiag(m,14,k,j,i) = d_rr00;       // tetrad radiation energy (small/neg => corrupt)
    } // endif (correct_radsrc_velocity_ || limit_opacity_)


    // Calculate polynomial coefficients
    Real wght_sum = 0.0;
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    // denom == (1 - suma3), accumulated DIRECTLY as a sum of positive terms to avoid the
    // catastrophic cancellation "1 - suma3" when scattering is enormous (suma3 -> 1).
    // Algebra: 1 - (dtcsigs-dtcsigp)<n0_cm/(n0+(dtcsiga+dtcsigs)n0_cm)>
    //        = <(n0 + (dtcsiga+dtcsigp)n0_cm)/(n0+(dtcsiga+dtcsigs)n0_cm)>.
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
      denom += omega_cm*(n0 + (dtcsiga + dtcsigp)*n0_cm)*vncsigma;
    }
    suma1 /= wght_sum;
    suma2 /= wght_sum;
    denom /= wght_sum;                     // == 1 - suma3, cancellation-free
    suma1 *= (dtcsiga + dtcsigp);

    // Reject the source update ONLY if the implicit denominator is degenerate (non-finite
    // or non-positive) -- leaves the cell unchanged; the failure tracker will flag it.  A
    // small-but-positive denom is PHYSICAL: for enormous scattering (dt*sigma_s ~ 1e17),
    // denom ~ 1/(1+dt*sigma_s) ~ 3.5e-18 is the correct stiff value, which the direct sum
    // recovers accurately (the old 1-suma3 rounded it to 0).  Keep it and SOLVE the update;
    // any genuine overflow downstream is caught by the FourthPolyRoot isfinite guard.
    bool badcell = (!isfinite(denom) || (denom <= 0.0));

    // compute coefficients
    Real coef[2] = {0.0, 0.0};
    Real tgasnew = tgas;
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

      // ---- velocity (W) limiter: scale the exchange to bound the RECOVERED post-coupling W ---
      // Preview the full (lambda=1) update to get the true momentum exchange dS_i=(m_old-m_new)_i
      // with the FINAL post-correction u_tet, WITHOUT writing i0.  Then bound the cold-gas
      // recovered W (see below) at W_limit by scaling the WHOLE exchange by lambda -- this caps
      // the density drop (rho_pre/rho_post = W_post/W_pre) and the Lorentz factor together.  Runs
      // only where the radiation-dominance + stiffness gate fired (lambda=1, untouched, else).
      if (dvlimit_gate) {
        Real dm1 = 0.0, dm2 = 0.0, dm3 = 0.0;   // actual (m_old - m_new)_i
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
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real intensity_cm = 4.0*M_PI*(i0_old/(n0*n_0))*SQR(SQR(n0_cm));
          Real vncsigma2 = n0_cm/(n0 + (dtcsiga + dtcsigs)*n0_cm);
          Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm + (dtcsiga+dtcsigp)*emission
                        - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
          Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
          Real diff = (i0_old - i0_upd)/n_0*solid_angles_.d_view(n);   // (old-new)/n_0*dOmega
          dm1 += n_1*diff;  dm2 += n_2*diff;  dm3 += n_3*diff;
        }
        // Inverse SPATIAL 3-metric gamma^{ij} = g^{ij} - g^{0i}g^{0j}/g^{00} (positive-definite;
        // the bare spatial block g^{ij} of the 4-inverse is NOT, and near the horizon can make
        // the momentum norm underestimate or go <=0).
        Real ig00  = 1.0/gupper[0][0];
        Real gam11 = gupper[1][1] - gupper[0][1]*gupper[0][1]*ig00;
        Real gam22 = gupper[2][2] - gupper[0][2]*gupper[0][2]*ig00;
        Real gam33 = gupper[3][3] - gupper[0][3]*gupper[0][3]*ig00;
        Real gam12 = gupper[1][2] - gupper[0][1]*gupper[0][2]*ig00;
        Real gam13 = gupper[1][3] - gupper[0][1]*gupper[0][3]*ig00;
        Real gam23 = gupper[2][3] - gupper[0][2]*gupper[0][3]*ig00;
        // pre-radiation fluid momentum S_i and gamma-norm magnitudes
        Real M1 = u0_(m,IM1,k,j,i), M2 = u0_(m,IM2,k,j,i), M3 = u0_(m,IM3,k,j,i);
        Real Mpre_sq = gam11*M1*M1 + gam22*M2*M2 + gam33*M3*M3
                     + 2.0*(gam12*M1*M2 + gam13*M1*M3 + gam23*M2*M3);
        Real cross   = gam11*M1*dm1 + gam22*M2*dm2 + gam33*M3*dm3
                     + gam12*(M1*dm2+M2*dm1) + gam13*(M1*dm3+M3*dm1) + gam23*(M2*dm3+M3*dm2);
        Real dS_sq   = gam11*dm1*dm1 + gam22*dm2*dm2 + gam33*dm3*dm3
                     + 2.0*(gam12*dm1*dm2 + gam13*dm1*dm3 + gam23*dm2*dm3);
        // Cold-gas Lorentz factor W = sqrt(1 + gamma^{ij} S_i S_j / D^2) (h=1; for h>1 this
        // OVER-estimates W, so bounding it is strictly conservative).  D = u0(IDN) is conserved
        // across the substep.  This bounds the RECOVERED post-coupling W directly -- no |S|->W
        // guess and no (metric-entangled) energy convention -- unlike the frame/norm limiters.
        Real d_cons = u0_(m,IDN,k,j,i);
        Real inv_d2 = 1.0/fmax(SQR(d_cons), 1.0e-300);
        Real w_pre  = sqrt(1.0 + fmax(Mpre_sq,0.0)*inv_d2);        // cold-gas pre-radiation W
        Real w_lim  = fmax(w_pre, fmin(w_hard_, fw_*w_pre));       // max[W_pre,min(W_hard,fw*W_pre)]
        Real ptgt_sq = SQR(d_cons)*(SQR(w_lim) - 1.0);            // |S|^2 that recovers W = W_limit
        Real full_sq = Mpre_sq + 2.0*cross + dS_sq;               // |S_post(lambda=1)|^2
        Real w_full = sqrt(1.0 + fmax(full_sq,0.0)*inv_d2);       // W_post at lambda=1 (diag)
        Real lam;
        if (!isfinite(full_sq) || !isfinite(dS_sq) || !isfinite(cross)
            || Mpre_sq <= 0.0 || !(d_cons > 0.0)) {
          lam = 0.0;   // FAIL CLOSED: unreliable norm/density in a flagged cell -> freeze
        } else if (full_sq > ptgt_sq) {
          // solve |S_pre + lam*dS|^2 = ptgt_sq  ->  a lam^2 + b lam + c = 0
          Real a = dS_sq, b = 2.0*cross, c = Mpre_sq - ptgt_sq;   // c < 0 => one root in (0,1)
          lam = 1.0;
          if (a > 1.0e-300) {
            Real disc = b*b - 4.0*a*c;
            lam = (disc >= 0.0) ? (-b + sqrt(disc))/(2.0*a) : 0.0;
          } else if (fabs(b) > 1.0e-300) {
            lam = -c/b;
          }
          lam = fmin(fmax(lam, 0.0), 1.0);
        } else {
          lam = 1.0;   // full kick already within the W budget
        }
        dvlimit_lambda = lam;
        dvlimit_budget_sq = ptgt_sq;   // shared with the Compton sub-step (same W budget)
        dvlimit_wpost = w_full;        // diag: recovered W at lambda=1
        dvlimit_wlim  = w_lim;         // diag: W_limit
      }

      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      for (int n=0; n<=nang1; ++n) {
        // compute coordinate normal components
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

        // compute moments before coupling (i0_old = pre-update specific intensity)
        Real i0_old = i0_(m,n,k,j,i);
        m_old[0] += (    i0_old    *solid_angles_.d_view(n));
        m_old[1] += (n_1*i0_old/n_0*solid_angles_.d_view(n));
        m_old[2] += (n_2*i0_old/n_0*solid_angles_.d_view(n));
        m_old[3] += (n_3*i0_old/n_0*solid_angles_.d_view(n));

        // update intensity (full implicit update -> i0_upd)
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_old/(n0*n_0))*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm
                      + (dtcsiga+dtcsigp)*emission
                      - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
        Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) +
                                  di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

        // compute moments after the FULL coupling (drives the gas<->radiation exchange)
        m_new[0] += (    i0_upd    *solid_angles_.d_view(n));
        m_new[1] += (n_1*i0_upd/n_0*solid_angles_.d_view(n));
        m_new[2] += (n_2*i0_upd/n_0*solid_angles_.d_view(n));
        m_new[3] += (n_3*i0_upd/n_0*solid_angles_.d_view(n));

        // Density-drop limiter: store the conservative lambda-blend of the intensity,
        // i0_final = (1-lambda)*i0_old + lambda*i0_upd.  dvlimit_lambda==1 unless this cell is
        // in the pathological regime, so this is bit-for-bit i0_upd in the normal case.
        i0_(m,n,k,j,i) = i0_old + dvlimit_lambda*(i0_upd - i0_old);

        // handle excision
        // NOTE(@pdmullen): The below zeroes all intensities within rks <= r_excision and
        // zeroes intensities within angles where n_0 is about zero. When Compton is
        // enabled, we delay the n_0_floor excision so that intensites updated via
        // absorption and scattering inform the Compton update
        if (excise) {
          bool apply_excision = (rad_mask_(m,k,j,i) ||
                                 (!(is_compton_enabled_) && fabs(n_0) < n_0_floor_));
          if (apply_excision) { i0_(m,n,k,j,i) = 0.0; }
        }
      }
      // update conserved fluid variables.  The gas gains lambda*(m_old-m_new) = m_old-m_final,
      // exactly the four-momentum the radiation lost to the stored i0_final, so gas+radiation
      // four-momentum is conserved for any lambda (lambda==1 => identical to before).
      if (affect_fluid_) {
        u0_(m,IEN,k,j,i) += dvlimit_lambda*(m_old[0] - m_new[0]);
        u0_(m,IM1,k,j,i) += dvlimit_lambda*(m_old[1] - m_new[1]);
        u0_(m,IM2,k,j,i) += dvlimit_lambda*(m_old[2] - m_new[2]);
        u0_(m,IM3,k,j,i) += dvlimit_lambda*(m_old[3] - m_new[3]);
      }
    }

    // compton scattering
    if (is_compton_enabled_) {
      // Use the partially updated gas temperature, made consistent with the density-drop
      // limiter: tgasnew is the full (lambda=1) absorption/emission solve, but only lambda of
      // that energy was actually deposited, so Compton must see tgas blended back by lambda
      // (identical to tgasnew when lambda==1, i.e. whenever the limiter did not fire).
      tgas = tgas + dvlimit_lambda*(tgasnew - tgas);

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

        // ---- W limiter for the Compton sub-step: share the SAME W budget -------------------
        // Scattering capped |S|^2 <= dvlimit_budget_sq (= D^2(W_limit^2-1)); Compton adds a
        // further exchange to the already-updated momentum, so bound the CUMULATIVE |S| against
        // the same budget (post-scattering momentum from u0(IM)).  lamc=1 if Compton pushes back
        // inside, <1 (down to 0) if it would push |S| further past the budget.
        if (dvlimit_gate) {
          Real dm1 = 0.0, dm2 = 0.0, dm3 = 0.0;
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
            Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                          u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
            Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
            Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) + di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);
            Real diff = (i0_old - i0_upd)/n_0*solid_angles_.d_view(n);
            dm1 += n_1*diff;  dm2 += n_2*diff;  dm3 += n_3*diff;
          }
          Real ig00  = 1.0/gupper[0][0];
          Real gam11 = gupper[1][1] - gupper[0][1]*gupper[0][1]*ig00;
          Real gam22 = gupper[2][2] - gupper[0][2]*gupper[0][2]*ig00;
          Real gam33 = gupper[3][3] - gupper[0][3]*gupper[0][3]*ig00;
          Real gam12 = gupper[1][2] - gupper[0][1]*gupper[0][2]*ig00;
          Real gam13 = gupper[1][3] - gupper[0][1]*gupper[0][3]*ig00;
          Real gam23 = gupper[2][3] - gupper[0][2]*gupper[0][3]*ig00;
          Real M1 = u0_(m,IM1,k,j,i), M2 = u0_(m,IM2,k,j,i), M3 = u0_(m,IM3,k,j,i);
          Real Mcur_sq = gam11*M1*M1 + gam22*M2*M2 + gam33*M3*M3
                       + 2.0*(gam12*M1*M2 + gam13*M1*M3 + gam23*M2*M3);
          Real cross   = gam11*M1*dm1 + gam22*M2*dm2 + gam33*M3*dm3
                       + gam12*(M1*dm2+M2*dm1) + gam13*(M1*dm3+M3*dm1) + gam23*(M2*dm3+M3*dm2);
          Real dS_sq   = gam11*dm1*dm1 + gam22*dm2*dm2 + gam33*dm3*dm3
                       + 2.0*(gam12*dm1*dm2 + gam13*dm1*dm3 + gam23*dm2*dm3);
          Real full_sq = Mcur_sq + 2.0*cross + dS_sq;
          Real lamc;
          if (!isfinite(full_sq) || !isfinite(dS_sq) || !isfinite(cross)) {
            lamc = 0.0;   // fail closed
          } else if (full_sq > dvlimit_budget_sq) {
            Real a = dS_sq, b = 2.0*cross, c = Mcur_sq - dvlimit_budget_sq;
            lamc = 1.0;
            if (a > 1.0e-300) {
              Real disc = b*b - 4.0*a*c;
              lamc = (disc >= 0.0) ? (-b + sqrt(disc))/(2.0*a) : 0.0;
            } else if (fabs(b) > 1.0e-300) {
              lamc = -c/b;
            }
            lamc = fmin(fmax(lamc, 0.0), 1.0);
          } else {
            lamc = 1.0;
          }
          dvlimit_lambda_c = lamc;
        }

        Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
        for (int n=0; n<=nang1; ++n) {
          // compute coordinate normal components
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

          // compute moments before coupling (i0_old = pre-update specific intensity)
          Real i0_old = i0_(m,n,k,j,i);
          m_old[0] += (    i0_old    *solid_angles_.d_view(n));
          m_old[1] += (n_1*i0_old/n_0*solid_angles_.d_view(n));
          m_old[2] += (n_2*i0_old/n_0*solid_angles_.d_view(n));
          m_old[3] += (n_3*i0_old/n_0*solid_angles_.d_view(n));

          // update intensity (full Compton update -> i0_upd)
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
          Real i0_upd = n0*n_0*fmax(i0_old/(n0*n_0) +
                                    di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

          // compute moments after the FULL coupling
          m_new[0] += (    i0_upd    *solid_angles_.d_view(n));
          m_new[1] += (n_1*i0_upd/n_0*solid_angles_.d_view(n));
          m_new[2] += (n_2*i0_upd/n_0*solid_angles_.d_view(n));
          m_new[3] += (n_3*i0_upd/n_0*solid_angles_.d_view(n));

          // density-drop limiter: conservative lambda-blend with the COMPTON scale
          // (dvlimit_lambda_c, decided against the shared budget; ==1 in the normal case)
          i0_(m,n,k,j,i) = i0_old + dvlimit_lambda_c*(i0_upd - i0_old);

          // handle excision (see notes above)
          if (excise) {
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }

        // feedback on fluid (scaled by the Compton density-drop lambda for conservation)
        if (affect_fluid_) {
          u0_(m,IEN,k,j,i) += dvlimit_lambda_c*(m_old[0] - m_new[0]);
          u0_(m,IM1,k,j,i) += dvlimit_lambda_c*(m_old[1] - m_new[1]);
          u0_(m,IM2,k,j,i) += dvlimit_lambda_c*(m_old[2] - m_new[2]);
          u0_(m,IM3,k,j,i) += dvlimit_lambda_c*(m_old[3] - m_new[3]);
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

    // Density-drop limiter diagnostics: one authoritative write per cell at the very end of the
    // update, so slots 15-16 are NEVER stale (covers the badcell / temperature-equilibrium
    // paths, where the previews above did not run).  Slot 16 = the most restrictive scale
    // actually applied this step (scattering vs Compton; both default to 1 when nothing fired).
    if (rad_dvlimit_) {
      ldiag(m,15,k,j,i) = dvlimit_lambda_s;                        // Lambda_s
      ldiag(m,16,k,j,i) = fmin(dvlimit_lambda, dvlimit_lambda_c);  // applied lambda (min of the two)
      ldiag(m,17,k,j,i) = dvlimit_wpost;                           // recovered W_post at lambda=1
      ldiag(m,18,k,j,i) = dvlimit_wlim;                            // W_limit
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
