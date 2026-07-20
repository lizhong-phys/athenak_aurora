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

  // state
  bool snap_ready = false;    // snapshot buffers allocated
  bool have_snap  = false;    // a snapshot was taken this cycle
  bool reported   = false;    // a failure has already been reported

  bool has_mhd, has_rad;

  // "1 cycle before" snapshot buffers (device), matching u0/w0/bcc0/i0 shapes
  DvceArray5D<Real> snap_u, snap_w, snap_b, snap_i;

  bool InWindow() const;
  void AllocateSnapshots();
  // dump neighborhood of cell (m,k,j,i) both live ("after") and from snapshot ("before")
  void WriteReport(int m, int k, int j, int i, Real rks, int cycle, int stage);
};

#endif // UTILS_FAILURE_TRACKER_HPP_
