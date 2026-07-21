#ifndef UTILS_FAILURE_TRACKER_HPP_
#define UTILS_FAILURE_TRACKER_HPP_
//========================================================================================
// AthenaK astrophysical (GR)MHD code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file failure_tracker.hpp
//! \brief Observation-only debug subsystem that locates the FIRST non-finite cell in a
//! chosen time window and dumps its neighborhood both at failure and at the start of the
//! failing cycle ("1 cycle before").  It applies NO floors and changes NO physics: it only
//! snapshots state at the top of each cycle, scans for non-finite values after every
//! stage, selects the globally-first offending cell (min radius, then rank), writes a
//! human-readable report, and (optionally) aborts so the run stops on the first failure.
//!
//! Enabled via the input block (all defaults keep it OFF / zero-overhead):
//!   <failure_tracker>
//!   enabled            = true
//!   start_time         = 23960.0
//!   stop_time          = 23970.0
//!   neighborhood       = 2       # half-width of the dumped stencil (2 -> 5x5x5)
//!   scan_ghost         = true    # include ghost zones in the non-finite scan
//!   abort_on_nonfinite = true    # stop the run at the first non-finite cell

#include <string>

#include "athena.hpp"
#include "tasklist/task_list.hpp"   // TaskStatus

// forward declarations
class MeshBlockPack;
class Driver;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class FailureTracker
//! \brief per-MeshBlockPack owner of snapshot buffers + failure-detection tasks

class FailureTracker {
 public:
  FailureTracker(MeshBlockPack *ppack, ParameterInput *pin);
  ~FailureTracker();

  // Task 1: snapshot state at the top of each cycle (list "before_timeintegrator").
  TaskStatus Snapshot(Driver *pdrive, int stage);
  // Task 2: scan for the first non-finite cell after every stage (list "after_stagen").
  TaskStatus Detect(Driver *pdrive, int stage);

  bool enabled = false;

 private:
  MeshBlockPack *pmy_pack;

  // controls (from <failure_tracker>)
  Real start_time, stop_time;
  int  nghbr;                 // stencil half-width
  bool scan_ghost;
  bool abort_on_nonfinite;

  // precursor "runaway-onset" trigger: fire on the first cell whose gas temperature
  // exceeds runaway_temp AND (optionally) whose Lorentz factor is at the ceiling.
  bool runaway_detect;
  // onset is any of these OR-signals (each 0/false disables); whichever fires first wins:
  Real runaway_temp;          // tgas = (gamma-1)*eint/rho threshold  (thermal runaway)
  Real runaway_rho;           // rho threshold                        (density runaway)
  Real runaway_erad;          // raw sum|i0| threshold                (radiation runaway)
  bool runaway_wclamp;        // W >= runaway_wfrac*gamma_max         (velocity-clamp)
  Real runaway_wfrac;         // W-fraction for the velocity-clamp signal
  Real runaway_rmin;          // ignore runaway candidates with rks < this (excise-edge shell)

  // state
  bool snap_ready = false;    // snapshot buffers allocated
  bool have_snap  = false;    // a snapshot was taken this cycle
  bool reported   = false;    // a failure has already been reported

  bool has_mhd, has_hyd, has_rad;

  // "1 cycle before" snapshot buffers.  HOST-resident (host RAM is plentiful) so we do
  // NOT double the large device radiation array i0 in GPU memory during the window.
  DvceArray5D<Real>::HostMirror snap_u, snap_w, snap_b, snap_i;

  bool InWindow() const;
  void AllocateSnapshots();
  // dump neighborhood of cell (m,k,j,i) both live ("after") and from snapshot ("before").
  // trigger: 0 = non-finite value, 1 = runaway-onset.  win_ghost: winner is a ghost cell.
  void WriteReport(int m, int k, int j, int i, Real rks, int cycle, int stage,
                   int trigger, bool win_ghost);
};

#endif // UTILS_FAILURE_TRACKER_HPP_
