//========================================================================================
// AthenaK astrophysical (GR)MHD code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file failure_tracker.cpp
//! \brief Implementation of the observation-only failure tracker.  See header for intent.
//!
//! Flow per cycle (only inside [start_time, stop_time]):
//!   before_timeintegrator : Snapshot()  copies u0/w0/bcc0/i0 -> snap_*  ("1 cycle before")
//!   after_stagen (each s) : Detect()     device MinLoc scan for the first non-finite cell
//!                                         (smallest SKS radius), MPI_MINLOC across ranks,
//!                                         owning rank WriteReport()s before+after stencil,
//!                                         then all ranks abort (if abort_on_nonfinite).
//! Nothing here mutates evolved state.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "eos/eos.hpp"
#include "utils/failure_tracker.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// SKS (spherical Kerr-Schild) radius from CKS coordinates; matches cartesian_ks.hpp.
KOKKOS_INLINE_FUNCTION
Real SksRadiusLocal(Real x, Real y, Real z, Real a) {
  Real rad2 = x*x + y*y + z*z;
  Real a2 = a*a;
  return sqrt((rad2 - a2 + sqrt((rad2 - a2)*(rad2 - a2) + 4.0*a2*z*z))*0.5);
}
constexpr Real kBig = 1.0e300;
}  // namespace

//----------------------------------------------------------------------------------------
FailureTracker::FailureTracker(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack) {
  enabled            = pin->GetOrAddBoolean("failure_tracker", "enabled", false);
  start_time         = pin->GetOrAddReal("failure_tracker", "start_time", -1.0e300);
  stop_time          = pin->GetOrAddReal("failure_tracker", "stop_time",   1.0e300);
  nghbr              = pin->GetOrAddInteger("failure_tracker", "neighborhood", 2);
  scan_ghost         = pin->GetOrAddBoolean("failure_tracker", "scan_ghost", true);
  abort_on_nonfinite = pin->GetOrAddBoolean("failure_tracker", "abort_on_nonfinite", true);

  has_mhd = (pmy_pack->pmhd != nullptr);
  has_rad = (pmy_pack->prad != nullptr);

  if (enabled && global_variable::my_rank == 0) {
    std::cout << "### FailureTracker ENABLED: window [" << start_time << ", " << stop_time
              << "], neighborhood=" << nghbr << ", scan_ghost=" << scan_ghost
              << ", abort_on_nonfinite=" << abort_on_nonfinite << std::endl;
  }
}

//----------------------------------------------------------------------------------------
FailureTracker::~FailureTracker() {}

//----------------------------------------------------------------------------------------
bool FailureTracker::InWindow() const {
  Real t = pmy_pack->pmesh->time;
  return (t >= start_time) && (t <= stop_time);
}

//----------------------------------------------------------------------------------------
void FailureTracker::AllocateSnapshots() {
  // (Re)allocate to match the live arrays; tolerant of a mid-window AMR remesh, since
  // deep_copy requires exactly matching extents.
  auto match = [](DvceArray5D<Real> &dst, DvceArray5D<Real> &src) {
    if (dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2) || dst.extent(3) != src.extent(3) ||
        dst.extent(4) != src.extent(4)) {
      Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2),
                      src.extent(3), src.extent(4));
    }
  };
  if (has_mhd) {
    match(snap_u, pmy_pack->pmhd->u0);
    match(snap_w, pmy_pack->pmhd->w0);
    match(snap_b, pmy_pack->pmhd->bcc0);
  } else {
    match(snap_u, pmy_pack->phydro->u0);
    match(snap_w, pmy_pack->phydro->w0);
  }
  if (has_rad) {
    match(snap_i, pmy_pack->prad->i0);
  }
  snap_ready = true;
}

//----------------------------------------------------------------------------------------
//! Snapshot state at the start of the cycle (before any stage runs).
TaskStatus FailureTracker::Snapshot(Driver *pdrive, int stage) {
  if (!enabled || reported || !InWindow()) return TaskStatus::complete;
  AllocateSnapshots();
  if (has_mhd) {
    Kokkos::deep_copy(snap_u, pmy_pack->pmhd->u0);
    Kokkos::deep_copy(snap_w, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(snap_b, pmy_pack->pmhd->bcc0);
  } else {
    Kokkos::deep_copy(snap_u, pmy_pack->phydro->u0);
    Kokkos::deep_copy(snap_w, pmy_pack->phydro->w0);
  }
  if (has_rad) {
    Kokkos::deep_copy(snap_i, pmy_pack->prad->i0);
  }
  have_snap = true;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Scan for the first non-finite cell; report + abort on the globally-first one.
TaskStatus FailureTracker::Detect(Driver *pdrive, int stage) {
  if (!enabled || reported || !InWindow()) return TaskStatus::complete;

  // fluid + radiation arrays (captured by value into the device lambda)
  DvceArray5D<Real> u0, w0, bcc0, i0;
  if (has_mhd) { u0 = pmy_pack->pmhd->u0; w0 = pmy_pack->pmhd->w0; bcc0 = pmy_pack->pmhd->bcc0; }
  else         { u0 = pmy_pack->phydro->u0; w0 = pmy_pack->phydro->w0; }
  if (has_rad) { i0 = pmy_pack->prad->i0; }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto size = pmy_pack->pmb->mb_size;
  Real spin = pmy_pack->pcoord->coord_data.bh_spin;

  int nmb = pmy_pack->nmb_thispack;
  int nc1 = u0.extent_int(4), nc2 = u0.extent_int(3), nc3 = u0.extent_int(2);
  int nu = u0.extent_int(1);
  int nw = w0.extent_int(1);
  int nb = has_mhd ? bcc0.extent_int(1) : 0;
  int na = has_rad ? i0.extent_int(1) : 0;
  bool hmhd = has_mhd, hrad = has_rad;

  int il, iu, jl, ju, kl, ku;
  if (scan_ghost) { il = 0; iu = nc1-1; jl = 0; ju = nc2-1; kl = 0; ku = nc3-1; }
  else            { il = is; iu = ie;   jl = js; ju = je;   kl = ks; ku = ke;   }
  int ni = iu-il+1, nj = ju-jl+1, nk = ku-kl+1;
  long ncells = static_cast<long>(nmb)*nk*nj*ni;

  using minloc_t = Kokkos::MinLoc<Real, long>;
  minloc_t::value_type res;
  res.val = kBig; res.loc = -1;
  Kokkos::parallel_reduce("fail_detect",
  Kokkos::RangePolicy<Kokkos::IndexType<long>>(0, ncells),
  KOKKOS_LAMBDA(const long idx, minloc_t::value_type &lmin) {
    int ii = il + static_cast<int>(idx % ni);
    long t = idx / ni;
    int jj = jl + static_cast<int>(t % nj);
    t /= nj;
    int kk = kl + static_cast<int>(t % nk);
    int m  = static_cast<int>(t / nk);

    bool bad = false;
    for (int v = 0; v < nu; ++v) { if (!isfinite(u0(m,v,kk,jj,ii))) { bad = true; } }
    for (int v = 0; v < nw; ++v) { if (!isfinite(w0(m,v,kk,jj,ii))) { bad = true; } }
    if (hmhd) { for (int v = 0; v < nb; ++v) { if (!isfinite(bcc0(m,v,kk,jj,ii))) bad = true; } }
    if (hrad) { for (int v = 0; v < na; ++v) { if (!isfinite(i0(m,v,kk,jj,ii)))   bad = true; } }
    if (!bad) return;

    Real x1v = CellCenterX(ii-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(jj-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(kk-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rks = SksRadiusLocal(x1v, x2v, x3v, spin);
    if (rks < lmin.val) { lmin.val = rks; lmin.loc = idx; }
  }, minloc_t(res));

  bool local_bad = (res.val < kBig) && (res.loc >= 0);
  Real local_min = local_bad ? res.val : kBig;

  // decode winner (m,k,j,i) on host
  int wm = -1, wk = -1, wj = -1, wi = -1;
  if (local_bad) {
    long idx = res.loc;
    wi = il + static_cast<int>(idx % ni);
    long t = idx / ni;
    wj = jl + static_cast<int>(t % nj);
    t /= nj;
    wk = kl + static_cast<int>(t % nk);
    wm = static_cast<int>(t / nk);
  }

  // select globally-first cell (smallest rks; MPI_MINLOC breaks ties by lowest rank)
  bool any = local_bad;
  int win_rank = local_bad ? global_variable::my_rank : -1;
#if MPI_PARALLEL_ENABLED
  struct { double v; int r; } in, out;
  in.v = local_min;
  in.r = local_bad ? global_variable::my_rank : (global_variable::nranks + 1);
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  any = (out.v < kBig);
  win_rank = out.r;
#endif

  if (any) {
    int cycle = pmy_pack->pmesh->ncycle;
    if (local_bad && global_variable::my_rank == win_rank) {
      WriteReport(wm, wk, wj, wi, local_min, cycle, stage);
    }
    reported = true;
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (abort_on_nonfinite) {
      if (global_variable::my_rank == win_rank) {
        std::cout << "### FailureTracker: first non-finite cell at rks=" << local_min
                  << " (cycle=" << cycle << ", stage=" << stage << ", rank=" << win_rank
                  << ") -> report written; ABORTING." << std::endl;
      }
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      std::exit(1);
#endif
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Dump the (2*nghbr+1)^3 stencil around the failing cell, "after" (live) and "before"
//! (start-of-cycle snapshot), to a human-readable text file on the owning rank.
void FailureTracker::WriteReport(int m, int k, int j, int i, Real rks,
                                 int cycle, int stage) {
  // live arrays -> host mirrors
  DvceArray5D<Real> u0, w0, bcc0, i0;
  if (has_mhd) { u0 = pmy_pack->pmhd->u0; w0 = pmy_pack->pmhd->w0; bcc0 = pmy_pack->pmhd->bcc0; }
  else         { u0 = pmy_pack->phydro->u0; w0 = pmy_pack->phydro->w0; }
  if (has_rad) { i0 = pmy_pack->prad->i0; }

  auto u_h = Kokkos::create_mirror_view(u0);  Kokkos::deep_copy(u_h, u0);
  auto w_h = Kokkos::create_mirror_view(w0);  Kokkos::deep_copy(w_h, w0);
  auto b_h = Kokkos::create_mirror_view(has_mhd ? bcc0 : u0);
  if (has_mhd) { Kokkos::deep_copy(b_h, bcc0); }
  auto ri_h = Kokkos::create_mirror_view(has_rad ? i0 : u0);
  if (has_rad) { Kokkos::deep_copy(ri_h, i0); }

  // snapshot mirrors
  auto su_h = Kokkos::create_mirror_view(snap_u);
  auto sw_h = Kokkos::create_mirror_view(snap_w);
  auto sb_h = Kokkos::create_mirror_view(has_mhd ? snap_b : snap_u);
  auto si_h = Kokkos::create_mirror_view(has_rad ? snap_i : snap_u);
  if (have_snap) {
    Kokkos::deep_copy(su_h, snap_u);
    Kokkos::deep_copy(sw_h, snap_w);
    if (has_mhd) { Kokkos::deep_copy(sb_h, snap_b); }
    if (has_rad) { Kokkos::deep_copy(si_h, snap_i); }
  }

  // excision masks -> host mirrors
  auto exfl = Kokkos::create_mirror_view(pmy_pack->pcoord->excision_floor);
  auto exfx = Kokkos::create_mirror_view(pmy_pack->pcoord->excision_flux);
  Kokkos::deep_copy(exfl, pmy_pack->pcoord->excision_floor);
  Kokkos::deep_copy(exfx, pmy_pack->pcoord->excision_flux);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nc1 = u0.extent_int(4), nc2 = u0.extent_int(3), nc3 = u0.extent_int(2);
  int na  = has_rad ? i0.extent_int(1) : 0;
  auto &sz = pmy_pack->pmb->mb_size.h_view;
  Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool flat = pmy_pack->pcoord->coord_data.is_minkowski;
  bool excise = pmy_pack->pcoord->coord_data.bh_excise;  // excision arrays sized only if true
  Real gm1 = (has_mhd ? pmy_pack->pmhd->peos->eos_data.gamma
                      : pmy_pack->phydro->peos->eos_data.gamma) - 1.0;
  int gid = pmy_pack->gids + m;

  // open output file
  char fname[256];
  std::snprintf(fname, sizeof(fname),
                "failure_tracker.rank_%05d.cycle_%08d.stage_%d.txt",
                global_variable::my_rank, cycle, stage);
  FILE *fp = std::fopen(fname, "w");
  if (fp == nullptr) {
    std::cout << "### FailureTracker: could not open " << fname << std::endl;
    return;
  }

  std::fprintf(fp, "# AthenaK FailureTracker report\n");
  std::fprintf(fp, "# time=%.10e dt=%.6e cycle=%d stage=%d\n",
               pmy_pack->pmesh->time, pmy_pack->pmesh->dt, cycle, stage);
  std::fprintf(fp, "# FIRST non-finite cell: rank=%d gid=%d m=%d (k,j,i)=(%d,%d,%d) rks=%.6e\n",
               global_variable::my_rank, gid, m, k, j, i, rks);
  std::fprintf(fp, "# neighborhood half-width=%d  have_before_snapshot=%d  nangles=%d\n",
               nghbr, static_cast<int>(have_snap), na);
  std::fprintf(fp, "# fields: rho eint pgas vx vy vz W | U(DN M1 M2 M3 EN) | "
                   "B1 B2 B3 sigma=B^2/rho | Erad_sum |I|max | ex_floor ex_flux | bad\n");

  // generic per-cell printer (host generic lambda; not a device kernel)
  auto dump_block = [&](const char *label, auto &U, auto &W, auto &B, auto &RI, bool valid) {
    std::fprintf(fp, "\n=== %s ===\n", label);
    if (!valid) { std::fprintf(fp, "(no data)\n"); return; }
    for (int dk = -nghbr; dk <= nghbr; ++dk) {
    for (int dj = -nghbr; dj <= nghbr; ++dj) {
    for (int di = -nghbr; di <= nghbr; ++di) {
      int kk = k+dk, jj = j+dj, ii = i+di;
      if (kk < 0 || kk >= nc3 || jj < 0 || jj >= nc2 || ii < 0 || ii >= nc1) continue;

      Real x1v = CellCenterX(ii-is, nx1, sz(m).x1min, sz(m).x1max);
      Real x2v = CellCenterX(jj-js, nx2, sz(m).x2min, sz(m).x2max);
      Real x3v = CellCenterX(kk-ks, nx3, sz(m).x3min, sz(m).x3max);

      Real rho = W(m,IDN,kk,jj,ii), eint = W(m,IEN,kk,jj,ii);
      Real vx = W(m,IVX,kk,jj,ii), vy = W(m,IVY,kk,jj,ii), vz = W(m,IVZ,kk,jj,ii);
      Real pgas = gm1*eint;

      // Lorentz factor from the CKS metric
      Real gl[4][4], gu[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, gl, gu);
      Real q = gl[1][1]*vx*vx + 2.0*gl[1][2]*vx*vy + 2.0*gl[1][3]*vx*vz
             + gl[2][2]*vy*vy + 2.0*gl[2][3]*vy*vz + gl[3][3]*vz*vz;
      Real W_lor = std::sqrt(std::fmax(1.0 + q, 0.0));

      Real b1 = 0.0, b2 = 0.0, b3 = 0.0, sigma = 0.0;
      if (has_mhd) {
        b1 = B(m,IBX,kk,jj,ii); b2 = B(m,IBY,kk,jj,ii); b3 = B(m,IBZ,kk,jj,ii);
        sigma = (b1*b1 + b2*b2 + b3*b3)/std::fmax(rho, 1.0e-300);
      }

      Real erad = 0.0, imax = 0.0;
      if (has_rad) {
        for (int n = 0; n < na; ++n) {
          Real iv = RI(m,n,kk,jj,ii);
          erad += iv;
          if (std::fabs(iv) > imax) imax = std::fabs(iv);
        }
      }

      // which fields are non-finite
      char bad[64]; int bl = 0; bad[0] = '\0';
      auto flag = [&](bool cond, const char *tag) {
        if (cond && bl < 50) { bl += std::snprintf(bad+bl, sizeof(bad)-bl, "%s ", tag); }
      };
      flag(!std::isfinite(rho), "rho"); flag(!std::isfinite(eint), "eint");
      flag(!std::isfinite(vx)||!std::isfinite(vy)||!std::isfinite(vz), "v");
      flag(!std::isfinite(U(m,IEN,kk,jj,ii)), "uEN");
      flag(has_mhd && (!std::isfinite(b1)||!std::isfinite(b2)||!std::isfinite(b3)), "B");
      flag(has_rad && !std::isfinite(erad), "I");

      const char *mark = (dk==0 && dj==0 && di==0) ? " *" : "  ";
      std::fprintf(fp,
        "%s (k,j,i)=(%d,%d,%d) rks=%.4e | %.5e %.5e %.5e %.4e %.4e %.4e %.5e | "
        "%.5e %.5e %.5e %.5e %.5e | %.4e %.4e %.5e | %.5e %.4e | %d %d | %s\n",
        mark, kk, jj, ii, SksRadiusLocal(x1v,x2v,x3v,spin),
        rho, eint, pgas, vx, vy, vz, W_lor,
        U(m,IDN,kk,jj,ii), U(m,IM1,kk,jj,ii), U(m,IM2,kk,jj,ii),
        U(m,IM3,kk,jj,ii), U(m,IEN,kk,jj,ii),
        b1, b2, b3, sigma, erad, imax,
        (excise ? static_cast<int>(exfl(m,kk,jj,ii)) : -1),
        (excise ? static_cast<int>(exfx(m,kk,jj,ii)) : -1),
        (bl > 0 ? bad : "-"));
    }}}
  };

  dump_block("AFTER (failure, live state)", u_h, w_h, b_h, ri_h, true);
  dump_block("BEFORE (start of failing cycle)", su_h, sw_h, sb_h, si_h, have_snap);

  std::fclose(fp);
  std::cout << "### FailureTracker: wrote " << fname << std::endl;
}
