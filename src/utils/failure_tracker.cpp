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
constexpr Real kBig   = 1.0e300;  // "no candidate" sentinel
constexpr Real kCat   = 1.0e12;   // category offset: runaway keys = kCat + rks (rks << kCat)
constexpr Real kGhost = 1.0e6;    // ghost-zone penalty: active cells always win a tie
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
  runaway_detect     = pin->GetOrAddBoolean("failure_tracker", "runaway_detect", false);
  runaway_temp       = pin->GetOrAddReal("failure_tracker", "runaway_temp", 100.0);
  runaway_rho        = pin->GetOrAddReal("failure_tracker", "runaway_rho", 0.0);
  runaway_erad       = pin->GetOrAddReal("failure_tracker", "runaway_erad", 0.0);
  // The fatal cell is a THERMAL runaway (tgas~1e45) at W~1.3, but is bordered by the
  // velocity-clamped (W=gamma_max) population -> offer W-clamp as its OWN onset signal.
  runaway_wclamp     = pin->GetOrAddBoolean("failure_tracker", "runaway_wclamp", false);
  runaway_wfrac      = pin->GetOrAddReal("failure_tracker", "runaway_wfrac", 0.99);
  runaway_rmin       = pin->GetOrAddReal("failure_tracker", "runaway_rmin", 0.0);

  has_mhd = (pmy_pack->pmhd != nullptr);
  has_hyd = (pmy_pack->phydro != nullptr);
  has_rad = (pmy_pack->prad != nullptr);
  if (enabled && !has_mhd && !has_hyd && !has_rad) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FailureTracker: no mhd/hydro/radiation present -> disabling."
                << std::endl;
    }
    enabled = false;
  }

  if (enabled && global_variable::my_rank == 0) {
    std::cout << "### FailureTracker ENABLED [v4]: window [" << start_time << ", " << stop_time
              << "], neighborhood=" << nghbr << ", scan_ghost=" << scan_ghost
              << ", abort_on_nonfinite=" << abort_on_nonfinite << std::endl;
    std::cout << "###   runaway_detect=" << runaway_detect
              << " temp=" << runaway_temp << " rho=" << runaway_rho
              << " erad=" << runaway_erad << " wclamp=" << runaway_wclamp
              << " wfrac=" << runaway_wfrac << " rmin=" << runaway_rmin << std::endl;
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
  // (Re)allocate host mirrors to match the live device arrays; tolerant of a mid-window
  // AMR remesh, since deep_copy requires exactly matching extents.
  auto match = [](DvceArray5D<Real>::HostMirror &dst, DvceArray5D<Real> &src) {
    if (dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2) || dst.extent(3) != src.extent(3) ||
        dst.extent(4) != src.extent(4)) {
      dst = Kokkos::create_mirror_view(src);
    }
  };
  if (has_mhd) {
    match(snap_u, pmy_pack->pmhd->u0);
    match(snap_w, pmy_pack->pmhd->w0);
    match(snap_b, pmy_pack->pmhd->bcc0);
  } else if (has_hyd) {
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
    Kokkos::deep_copy(snap_u, pmy_pack->pmhd->u0);      // device -> host
    Kokkos::deep_copy(snap_w, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(snap_b, pmy_pack->pmhd->bcc0);
  } else if (has_hyd) {
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
//! Snapshot primitives after the MHD update, before radiation coupling.  At this point w0
//! holds the post-MHD primitives (RadFluidCoupling's own ConsToPrim already ran, and the
//! coupling kernel does not touch w0), i.e. the state ENTERING the radiation source term.
//! Registered with a dependency on rad_coupl, so it runs before the final ConToPrim.
TaskStatus FailureTracker::SnapshotMid(Driver *pdrive, int stage) {
  if (!enabled || reported || !InWindow()) return TaskStatus::complete;
  if (!(has_mhd || has_hyd)) return TaskStatus::complete;         // need primitives for W
  DvceArray5D<Real> w0 = has_mhd ? pmy_pack->pmhd->w0 : pmy_pack->phydro->w0;
  if (snapm_w.extent(0) != w0.extent(0) || snapm_w.extent(1) != w0.extent(1) ||
      snapm_w.extent(2) != w0.extent(2) || snapm_w.extent(3) != w0.extent(3) ||
      snapm_w.extent(4) != w0.extent(4)) {
    snapm_w = Kokkos::create_mirror_view(w0);
  }
  Kokkos::deep_copy(snapm_w, w0);                                 // device -> host
  have_snapm = true;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Scan for the first non-finite cell; report + abort on the globally-first one.
TaskStatus FailureTracker::Detect(Driver *pdrive, int stage) {
  if (!enabled || reported || !InWindow()) return TaskStatus::complete;

  // fluid + radiation arrays (captured by value into the device lambda).  Empty views
  // (nu=nw=0) are simply not scanned, so radiation-only runs are handled gracefully.
  DvceArray5D<Real> u0, w0, bcc0, i0;
  if (has_mhd)      { u0 = pmy_pack->pmhd->u0;   w0 = pmy_pack->pmhd->w0; bcc0 = pmy_pack->pmhd->bcc0; }
  else if (has_hyd) { u0 = pmy_pack->phydro->u0; w0 = pmy_pack->phydro->w0; }
  if (has_rad) { i0 = pmy_pack->prad->i0; }

  // dimensions/extents come from whichever array exists (fluid preferred, else radiation)
  bool have_fluid = (has_mhd || has_hyd);
  auto &ref = have_fluid ? u0 : i0;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto size = pmy_pack->pmb->mb_size;
  Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool flat = pmy_pack->pcoord->coord_data.is_minkowski;

  // runaway-onset trigger parameters (captured into the device lambda)
  bool run_detect = runaway_detect && have_fluid;
  Real run_temp = runaway_temp, run_wfrac = runaway_wfrac, run_rmin = runaway_rmin;
  Real run_rho = runaway_rho, run_erad = runaway_erad;
  bool run_wclamp = runaway_wclamp;
  Real gm1 = 5.0/3.0 - 1.0, gamma_max = 1.0e30;
  if (has_mhd)      { gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
                      gamma_max = pmy_pack->pmhd->peos->eos_data.gamma_max; }
  else if (has_hyd) { gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
                      gamma_max = pmy_pack->phydro->peos->eos_data.gamma_max; }

  int nmb = pmy_pack->nmb_thispack;
  int nc1 = ref.extent_int(4), nc2 = ref.extent_int(3), nc3 = ref.extent_int(2);
  int nu = have_fluid ? u0.extent_int(1) : 0;
  int nw = have_fluid ? w0.extent_int(1) : 0;
  int nb = has_mhd ? bcc0.extent_int(1) : 0;
  int na = has_rad ? i0.extent_int(1) : 0;
  bool hfld = have_fluid, hmhd = has_mhd, hrad = has_rad;

  int il, iu, jl, ju, kl, ku;
  if (scan_ghost) { il = 0; iu = nc1-1; jl = 0; ju = nc2-1; kl = 0; ku = nc3-1; }
  else            { il = is; iu = ie;   jl = js; ju = je;   kl = ks; ku = ke;   }
  int ni = iu-il+1, nj = ju-jl+1, nk = ku-kl+1;
  long ncells = static_cast<long>(nmb)*nk*nj*ni;

  Real big = kBig, cat = kCat, ghost_pen = kGhost;   // captured into the device lambda
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
    if (hfld) {
      for (int v = 0; v < nu; ++v) { if (!isfinite(u0(m,v,kk,jj,ii))) { bad = true; } }
      for (int v = 0; v < nw; ++v) { if (!isfinite(w0(m,v,kk,jj,ii))) { bad = true; } }
    }
    if (hmhd) { for (int v = 0; v < nb; ++v) { if (!isfinite(bcc0(m,v,kk,jj,ii))) bad = true; } }
    if (hrad) { for (int v = 0; v < na; ++v) { if (!isfinite(i0(m,v,kk,jj,ii)))   bad = true; } }

    Real x1v = CellCenterX(ii-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(jj-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(kk-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rks = SksRadiusLocal(x1v, x2v, x3v, spin);
    // ghost-zone cells are copies of an active cell on another block; penalize them so
    // the originating ACTIVE cell always wins a same-radius tie.
    bool is_ghost = (ii < is || ii > ie || jj < js || jj > je || kk < ks || kk > ke);
    Real gpen = is_ghost ? ghost_pen : static_cast<Real>(0.0);

    // category-encoded key: non-finite (priority) sorts below runaway; within a category,
    // active before ghost, then by min rks.
    Real key = big;
    if (bad) {
      key = gpen + rks;                                 // category 0: non-finite
    } else if (run_detect && rks >= run_rmin) {
      Real wdn = w0(m,IDN,kk,jj,ii);
      Real wen = w0(m,IEN,kk,jj,ii);
      Real tgas = gm1*wen/fmax(wdn, static_cast<Real>(1.0e-300));
      // multi-signal onset: thermal OR density OR (raw) radiation runaway
      bool sig = false;
      if (run_temp > 0.0 && isfinite(tgas) && tgas > run_temp) { sig = true; }
      if (run_rho  > 0.0 && isfinite(wdn)  && wdn  > run_rho)  { sig = true; }
      if (!sig && run_erad > 0.0 && hrad) {
        Real sumI = 0.0;
        for (int n = 0; n < na; ++n) { sumI += fabs(i0(m,n,kk,jj,ii)); }
        if (isfinite(sumI) && sumI > run_erad) { sig = true; }
      }
      if (!sig && run_wclamp) {   // velocity-ceiling-saturated cell as its own signal
        Real gl[4][4], gu[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, gl, gu);
        Real vx = w0(m,IVX,kk,jj,ii), vy = w0(m,IVY,kk,jj,ii), vz = w0(m,IVZ,kk,jj,ii);
        Real q = gl[1][1]*vx*vx + 2.0*gl[1][2]*vx*vy + 2.0*gl[1][3]*vx*vz
               + gl[2][2]*vy*vy + 2.0*gl[2][3]*vy*vz + gl[3][3]*vz*vz;
        Real Wlor = sqrt(1.0 + q);
        if (isfinite(Wlor) && Wlor >= run_wfrac*gamma_max) { sig = true; }
      }
      if (sig) { key = cat + gpen + rks; }              // category 1: runaway-onset
    }
    if (key < big && key < lmin.val) { lmin.val = key; lmin.loc = idx; }
  }, minloc_t(res));

  bool local_hit = (res.val < kBig) && (res.loc >= 0);
  Real local_key = local_hit ? res.val : kBig;

  // decode winner (m,k,j,i) on host
  int wm = -1, wk = -1, wj = -1, wi = -1;
  if (local_hit) {
    long idx = res.loc;
    wi = il + static_cast<int>(idx % ni);
    long t = idx / ni;
    wj = jl + static_cast<int>(t % nj);
    t /= nj;
    wk = kl + static_cast<int>(t % nk);
    wm = static_cast<int>(t / nk);
  }

  // select globally-first cell.  Key encodes category (non-finite < runaway) then rks,
  // so MPI_MINLOC picks non-finite over runaway, and min-radius within a category.
  bool any = local_hit;
  int win_rank = local_hit ? global_variable::my_rank : -1;
  Real win_key = local_key;
#if MPI_PARALLEL_ENABLED
  struct { double v; int r; } in, out;
  in.v = local_key;
  in.r = local_hit ? global_variable::my_rank : (global_variable::nranks + 1);
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  any = (out.v < kBig);
  win_rank = out.r;
  win_key = out.v;
#endif

  if (any) {
    int cycle = pmy_pack->pmesh->ncycle;
    int trigger = (win_key >= kCat) ? 1 : 0;                // 0=non-finite, 1=runaway
    Real k2 = (trigger == 1) ? win_key - kCat : win_key;    // strip category offset
    bool win_ghost = (k2 >= kGhost);                        // ghost-zone winner?
    Real rrks = win_ghost ? k2 - kGhost : k2;               // strip ghost penalty -> rks
    const char *what = (trigger == 0) ? "non-finite" : "runaway-onset";
    if (local_hit && global_variable::my_rank == win_rank) {
      WriteReport(wm, wk, wj, wi, rrks, cycle, stage, trigger, win_ghost);
    }
    reported = true;
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // A runaway-onset hit always stops the run (its purpose is to catch the seed); a
    // non-finite hit stops only if abort_on_nonfinite.
    if (trigger == 1 || abort_on_nonfinite) {
      if (global_variable::my_rank == win_rank) {
        std::cout << "### FailureTracker: first " << what << " cell at rks=" << rrks
                  << (win_ghost ? " [GHOST zone -> no active cell matched]" : "")
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
                                 int cycle, int stage, int trigger, bool win_ghost) {
  bool have_fluid = has_mhd || has_hyd;

  // Live device arrays (fluid preferred, else radiation for shape).
  DvceArray5D<Real> u0, w0, bcc0, i0;
  if (has_mhd)      { u0 = pmy_pack->pmhd->u0;   w0 = pmy_pack->pmhd->w0; bcc0 = pmy_pack->pmhd->bcc0; }
  else if (has_hyd) { u0 = pmy_pack->phydro->u0; w0 = pmy_pack->phydro->w0; }
  if (has_rad) { i0 = pmy_pack->prad->i0; }
  DvceArray5D<Real> &ref = have_fluid ? u0 : i0;

  // Gather ONLY the winning block m to host (host RAM), not all blocks / the whole i0.
  // Each host array is 4D (comp,k,j,i); global (k,j,i) indexing is preserved.
  auto gather = [](DvceArray5D<Real> &dev, int mm) {
    auto sub = Kokkos::subview(dev, mm, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto h = Kokkos::create_mirror_view(sub);
    Kokkos::deep_copy(h, sub);
    return h;
  };
  auto u_h  = gather(have_fluid ? u0   : ref, m);
  auto w_h  = gather(have_fluid ? w0   : ref, m);
  auto b_h  = gather(has_mhd    ? bcc0 : ref, m);
  auto ri_h = gather(has_rad    ? i0   : ref, m);

  // Excision masks (bool; the full host mirror is tiny).
  auto exfl = Kokkos::create_mirror_view(pmy_pack->pcoord->excision_floor);
  auto exfx = Kokkos::create_mirror_view(pmy_pack->pcoord->excision_flux);
  Kokkos::deep_copy(exfl, pmy_pack->pcoord->excision_floor);
  Kokkos::deep_copy(exfx, pmy_pack->pcoord->excision_flux);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nc1 = ref.extent_int(4), nc2 = ref.extent_int(3), nc3 = ref.extent_int(2);
  int na  = has_rad ? i0.extent_int(1) : 0;
  auto &sz = pmy_pack->pmb->mb_size.h_view;
  Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool flat = pmy_pack->pcoord->coord_data.is_minkowski;
  bool excise = pmy_pack->pcoord->coord_data.bh_excise;  // excision arrays sized only if true
  Real gm1 = (has_mhd ? pmy_pack->pmhd->peos->eos_data.gamma
              : (has_hyd ? pmy_pack->phydro->peos->eos_data.gamma : 5.0/3.0)) - 1.0;
  int gid = pmy_pack->gids + m;

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
  if (trigger == 1) {
    std::fprintf(fp, "# TRIGGER = RUNAWAY-ONSET: first cell (rks>=%.3g) matching ANY enabled\n"
                     "# signal, all values still FINITE.  Signals (0/off = disabled):\n"
                     "#   tgas>%.3e | rho>%.3e | sum|i0|>%.3e | wclamp(W>=%.3g*gmax)=%d\n",
                 runaway_rmin, runaway_temp, runaway_rho, runaway_erad,
                 runaway_wfrac, static_cast<int>(runaway_wclamp));
  } else {
    std::fprintf(fp, "# TRIGGER = NON-FINITE value.\n");
  }
  std::fprintf(fp, "# Earliest stage this cycle; representative cell = minimum SKS radius\n"
                   "# among matching cells (not necessarily the temporally-first, since\n"
                   "# cells within a stage update concurrently).\n");
  std::fprintf(fp, "# rank=%d gid=%d m=%d (k,j,i)=(%d,%d,%d) rks=%.6e%s\n",
               global_variable::my_rank, gid, m, k, j, i, rks,
               win_ghost ? "  [WINNER IS A GHOST CELL: no active cell matched -> likely "
                           "introduced by communication/prolongation]" : "");
  std::fprintf(fp, "# neighborhood half-width=%d  have_before_snapshot=%d  nangles=%d\n",
               nghbr, static_cast<int>(have_snap), na);
  std::fprintf(fp, "# sigma = b^mu b_mu / rho (GR magnetization, metric-correct)\n");
  std::fprintf(fp, "# sumI_raw = sum_n |i0(n)| (RAW angle sum, NOT physical/comoving Erad;\n"
                   "#            diagnostic proxy only); |I|max = max_n |i0(n)|\n");
  std::fprintf(fp, "# fields: rho eint pgas vx vy vz W | U(DN M1 M2 M3 EN) | "
                   "B1 B2 B3 sigma | sumI_raw |I|max | ex_floor ex_flux | bad(component list)\n");

  // ---- W-stage attribution for the center cell: Lorentz factor at cycle-top, entering
  // radiation coupling (post-MHD), and after the full stage.  A jump cycletop->pre_rad
  // is the MHD update; pre_rad->post_stage is the radiation coupling.
  if (have_fluid) {
    Real x1c = CellCenterX(i-is, nx1, sz(m).x1min, sz(m).x1max);
    Real x2c = CellCenterX(j-js, nx2, sz(m).x2min, sz(m).x2max);
    Real x3c = CellCenterX(k-ks, nx3, sz(m).x3min, sz(m).x3max);
    Real gl[4][4], gu[4][4];
    ComputeMetricAndInverse(x1c, x2c, x3c, flat, spin, gl, gu);
    auto Wq = [&](Real vx, Real vy, Real vz) {
      Real q = gl[1][1]*vx*vx + 2.0*gl[1][2]*vx*vy + 2.0*gl[1][3]*vx*vz
             + gl[2][2]*vy*vy + 2.0*gl[2][3]*vy*vz + gl[3][3]*vz*vz;
      return std::sqrt(std::fmax(1.0 + q, 0.0));
    };
    Real Wb = have_snap  ? Wq(snap_w(m,IVX,k,j,i), snap_w(m,IVY,k,j,i), snap_w(m,IVZ,k,j,i))  : -1.0;
    Real Wm = have_snapm ? Wq(snapm_w(m,IVX,k,j,i), snapm_w(m,IVY,k,j,i), snapm_w(m,IVZ,k,j,i)) : -1.0;
    Real Wa = Wq(w_h(IVX,k,j,i), w_h(IVY,k,j,i), w_h(IVZ,k,j,i));
    std::fprintf(fp, "# W-STAGE (center): W_cycletop=%.5g  W_pre_radiation=%.5g"
                     "  W_post_stage=%.5g   (jump cycletop->pre_rad = MHD update;"
                     " pre_rad->post_stage = radiation coupling; -1 = no snapshot)\n",
                 Wb, Wm, Wa);
  }

  // ---- radiation-source diagnostics for the center cell (opacity limiter + velocity corr) ----
  if (has_rad &&
      (pmy_pack->prad->limit_opacity || pmy_pack->prad->correct_radsrc_velocity ||
       pmy_pack->prad->rad_dvlimit) &&
      (pmy_pack->prad->limiter_diag.extent_int(0) > m)) {
    auto ld_sub = Kokkos::subview(pmy_pack->prad->limiter_diag, m,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto ld_h = Kokkos::create_mirror_view(ld_sub);
    Kokkos::deep_copy(ld_h, ld_sub);
    if (pmy_pack->prad->limit_opacity) {
      std::fprintf(fp, "# LIMITER (center): on_floor=%.0f active=%.0f hit_tcap=%.0f  "
                       "Xi_raw=%.4e Xi_lim=%.4e T_eff=%.4e | sig_a=%.4e sig_p=%.4e sig_s=%.4e\n",
                   ld_h(0,k,j,i), ld_h(4,k,j,i), ld_h(5,k,j,i),
                   ld_h(1,k,j,i), ld_h(2,k,j,i), ld_h(3,k,j,i),
                   ld_h(6,k,j,i), ld_h(7,k,j,i), ld_h(8,k,j,i));
    }
    // VELCORR is recorded whenever the diag array exists (frame estimated even if the
    // correction is NOT applied); applied=correct_radsrc_velocity says whether it acted.
    std::fprintf(fp, "# VELCORR (center): applied=%d gate=%.0f E_rad_f=%.4e rho_h=%.4e | "
                     "urad_W=%.4e vrad_sq_raw=%.4e rr_tet00=%.4e  (vrad_sq_raw>~0.9975 => "
                     "radiation frame superluminal->would clamp gas to W=gamma_max)\n",
                 static_cast<int>(pmy_pack->prad->correct_radsrc_velocity),
                 ld_h(11,k,j,i), ld_h(9,k,j,i), ld_h(10,k,j,i),
                 ld_h(12,k,j,i), ld_h(13,k,j,i), ld_h(14,k,j,i));
    // WLIMIT: velocity (W) limiter.  candidate = the cheap R_rad/Lambda_mom gate opened (where the
    // preview ran); overshoot = the limiter ACTED (lambda_scat<1, i.e. the previewed |S| exceeded
    // the W budget).  A HEALTHY radiation-dominated cell shows candidate=1, overshoot=0, lambda=1;
    // a RUNAWAY shows overshoot=1, lambda<1.  W_cold_full is the SCATTERING preview at lambda=1
    // (does not include the subsequent Compton exchange).
    if (pmy_pack->prad->rad_dvlimit) {
      const int cand = (ld_h(20,k,j,i) > pmy_pack->prad->rad_dominance_lock &&
                        ld_h(15,k,j,i) > pmy_pack->prad->rad_momentum_lambda_lock) ? 1 : 0;
      const int over = (ld_h(16,k,j,i) < 1.0) ? 1 : 0;   // limiter acted (lambda_scat<1)
      std::fprintf(fp, "# WLIMIT (center): R_rad=%.4e Lambda_mom=%.4e candidate=%d | W_pre=%.4e "
                       "W_cold_full(scat)=%.4e W_limit=%.4e overshoot=%d | lambda_scat=%.4e "
                       "lambda_compton=%.4e  (gate R_rad>%.3g & Lambda_mom>%.3g; fw=%.3g W_hard=%.3g)\n",
                   ld_h(20,k,j,i), ld_h(15,k,j,i), cand, ld_h(21,k,j,i),
                   ld_h(17,k,j,i), ld_h(18,k,j,i), over, ld_h(16,k,j,i), ld_h(19,k,j,i),
                   pmy_pack->prad->rad_dominance_lock, pmy_pack->prad->rad_momentum_lambda_lock,
                   pmy_pack->prad->rad_max_w_increase, pmy_pack->prad->rad_w_hard);
      // Frame diagnostic.  Gamma_rel_pre is the exact PRE-coupling relative gas<->radiation Lorentz
      // factor (=1 comoving; physical drag reduces it).  beyond_mag is only a MAGNITUDE heuristic
      // (W_cold_full > 2 urad_W) -- it does NOT prove the gas crossed the frame (scalar magnitudes
      // carry no direction); a rigorous test needs the predicted POST-exchange relative Lorentz
      // factor, not computed here.
      const int beyond_mag = (ld_h(12,k,j,i) > 0.0 && ld_h(17,k,j,i) > 2.0*ld_h(12,k,j,i)) ? 1 : 0;
      std::fprintf(fp, "# WLIMIT-FRAME (center): Gamma_rel_pre=%.4e urad_W=%.4e W_cold_full=%.4e "
                       "beyond_mag=%d  (heuristic only; not a directional crossing test)\n",
                   ld_h(22,k,j,i), ld_h(12,k,j,i), ld_h(17,k,j,i), beyond_mag);
    }
  }

  // generic per-cell printer (host generic lambda; not a device kernel).  Host arrays are
  // 4D (comp,k,j,i) for the winning block m; index with global (kk,jj,ii).
  auto dump_block = [&](auto &U, auto &W, auto &B, auto &RI) {
    for (int dk = -nghbr; dk <= nghbr; ++dk) {
    for (int dj = -nghbr; dj <= nghbr; ++dj) {
    for (int di = -nghbr; di <= nghbr; ++di) {
      int kk = k+dk, jj = j+dj, ii = i+di;
      if (kk < 0 || kk >= nc3 || jj < 0 || jj >= nc2 || ii < 0 || ii >= nc1) continue;

      Real x1v = CellCenterX(ii-is, nx1, sz(m).x1min, sz(m).x1max);
      Real x2v = CellCenterX(jj-js, nx2, sz(m).x2min, sz(m).x2max);
      Real x3v = CellCenterX(kk-ks, nx3, sz(m).x3min, sz(m).x3max);

      Real rho = 0.0, eint = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;
      if (have_fluid) {
        rho = W(IDN,kk,jj,ii); eint = W(IEN,kk,jj,ii);
        vx = W(IVX,kk,jj,ii); vy = W(IVY,kk,jj,ii); vz = W(IVZ,kk,jj,ii);
      }
      Real pgas = gm1*eint;

      // metric, Lorentz factor
      Real gl[4][4], gu[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, gl, gu);
      Real q = gl[1][1]*vx*vx + 2.0*gl[1][2]*vx*vy + 2.0*gl[1][3]*vx*vz
             + gl[2][2]*vy*vy + 2.0*gl[2][3]*vy*vz + gl[3][3]*vz*vz;
      Real W_lor = std::sqrt(std::fmax(1.0 + q, 0.0));

      // GR magnetization sigma = b^mu b_mu / rho (matches radiation_source.cpp)
      Real b1 = 0.0, b2 = 0.0, b3 = 0.0, sigma = 0.0;
      if (has_mhd) {
        b1 = B(IBX,kk,jj,ii); b2 = B(IBY,kk,jj,ii); b3 = B(IBZ,kk,jj,ii);
        Real alpha = std::sqrt(-1.0/gu[0][0]);
        Real uu0 = W_lor/alpha;
        Real uu1 = vx - alpha*W_lor*gu[0][1];
        Real uu2 = vy - alpha*W_lor*gu[0][2];
        Real uu3 = vz - alpha*W_lor*gu[0][3];
        Real u_1 = gl[1][0]*uu0+gl[1][1]*uu1+gl[1][2]*uu2+gl[1][3]*uu3;
        Real u_2 = gl[2][0]*uu0+gl[2][1]*uu1+gl[2][2]*uu2+gl[2][3]*uu3;
        Real u_3 = gl[3][0]*uu0+gl[3][1]*uu1+gl[3][2]*uu2+gl[3][3]*uu3;
        Real bb0 = u_1*b1 + u_2*b2 + u_3*b3;
        Real bb1 = (b1 + bb0*uu1)/uu0;
        Real bb2 = (b2 + bb0*uu2)/uu0;
        Real bb3 = (b3 + bb0*uu3)/uu0;
        Real b_0 = gl[0][0]*bb0+gl[0][1]*bb1+gl[0][2]*bb2+gl[0][3]*bb3;
        Real b_1 = gl[1][0]*bb0+gl[1][1]*bb1+gl[1][2]*bb2+gl[1][3]*bb3;
        Real b_2 = gl[2][0]*bb0+gl[2][1]*bb1+gl[2][2]*bb2+gl[2][3]*bb3;
        Real b_3 = gl[3][0]*bb0+gl[3][1]*bb1+gl[3][2]*bb2+gl[3][3]*bb3;
        Real b_sq = bb0*b_0 + bb1*b_1 + bb2*b_2 + bb3*b_3;
        sigma = b_sq/std::fmax(rho, 1.0e-300);
      }

      Real erad = 0.0, imax = 0.0;
      if (has_rad) {
        for (int n = 0; n < na; ++n) {
          Real iv = RI(n,kk,jj,ii);
          erad += iv;
          if (std::fabs(iv) > imax) imax = std::fabs(iv);
        }
      }

      // EXACT non-finite components at this cell (rescan every stored variable/angle)
      char bad[512]; int bl = 0; bad[0] = '\0';
      auto add = [&](const char *fmt, int idx) {
        if (bl < 480) { bl += std::snprintf(bad+bl, sizeof(bad)-bl, fmt, idx); }
      };
      if (have_fluid) {
        for (int v = 0; v < U.extent_int(0); ++v) { if (!std::isfinite(U(v,kk,jj,ii))) add("u%d ", v); }
        for (int v = 0; v < W.extent_int(0); ++v) { if (!std::isfinite(W(v,kk,jj,ii))) add("w%d ", v); }
      }
      if (has_mhd) { for (int v = 0; v < B.extent_int(0);  ++v) { if (!std::isfinite(B(v,kk,jj,ii)))  add("b%d ", v); } }
      if (has_rad) { for (int n = 0; n < RI.extent_int(0); ++n) { if (!std::isfinite(RI(n,kk,jj,ii))) add("i%d ", n); } }

      const char *mark = (dk==0 && dj==0 && di==0) ? " *" : "  ";
      Real uDN = have_fluid ? U(IDN,kk,jj,ii) : 0.0;
      Real uM1 = have_fluid ? U(IM1,kk,jj,ii) : 0.0;
      Real uM2 = have_fluid ? U(IM2,kk,jj,ii) : 0.0;
      Real uM3 = have_fluid ? U(IM3,kk,jj,ii) : 0.0;
      Real uEN = have_fluid ? U(IEN,kk,jj,ii) : 0.0;
      std::fprintf(fp,
        "%s (k,j,i)=(%d,%d,%d) rks=%.4e | %.5e %.5e %.5e %.4e %.4e %.4e %.5e | "
        "%.5e %.5e %.5e %.5e %.5e | %.4e %.4e %.4e %.5e | %.5e %.4e | %d %d | %s\n",
        mark, kk, jj, ii, SksRadiusLocal(x1v,x2v,x3v,spin),
        rho, eint, pgas, vx, vy, vz, W_lor,
        uDN, uM1, uM2, uM3, uEN,
        b1, b2, b3, sigma, erad, imax,
        (excise ? static_cast<int>(exfl(m,kk,jj,ii)) : -1),
        (excise ? static_cast<int>(exfx(m,kk,jj,ii)) : -1),
        (bl > 0 ? bad : "-"));
    }}}
  };

  std::fprintf(fp, "\n=== AFTER (failure, live state) ===\n");
  dump_block(u_h, w_h, b_h, ri_h);
  std::fprintf(fp, "\n=== BEFORE (start of failing cycle) ===\n");
  if (have_snap) {
    // snapshot is already HOST-resident: subview block m (no copy)
    auto su_h = Kokkos::subview(have_fluid ? snap_u : snap_i, m,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto sw_h = Kokkos::subview(have_fluid ? snap_w : snap_i, m,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto sb_h = Kokkos::subview(has_mhd ? snap_b : snap_u, m,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto si_h = Kokkos::subview(has_rad ? snap_i : snap_u, m,
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    dump_block(su_h, sw_h, sb_h, si_h);
  } else {
    std::fprintf(fp, "(no snapshot: failure occurred before first in-window cycle top)\n");
  }

  std::fclose(fp);
  std::cout << "### FailureTracker: wrote " << fname << std::endl;
}
