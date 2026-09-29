#ifndef DIAGNOSTICS_OBSERVER_HPP_
#define DIAGNOSTICS_OBSERVER_HPP_
#include "athena.hpp"

namespace diagnostics {
// Host-side boundary, used only when recording. No observer kernel overlaps an
// evolution kernel, and no diagnostic output is returned to the solver.
class ObserverPhase {
 public:
  ObserverPhase() { Kokkos::fence("solver -> energy observer"); }
  ~ObserverPhase() { Kokkos::fence("energy observer -> solver"); }
  ObserverPhase(const ObserverPhase&) = delete;
  ObserverPhase& operator=(const ObserverPhase&) = delete;
};

template<class View>
typename View::const_type ReadOnly(const View &view) { return view; }

struct ReadOnlyFaceFlux {
  DvceArray5D<const Real> x1f, x2f, x3f;
  explicit ReadOnlyFaceFlux(const DvceFaceFld5D<Real> &v)
      : x1f(v.x1f), x2f(v.x2f), x3f(v.x3f) {}
};
struct ReadOnlyEdgeField {
  DvceArray4D<const Real> x1e, x2e, x3e;
  explicit ReadOnlyEdgeField(const DvceEdgeFld4D<Real> &v)
      : x1e(v.x1e), x2e(v.x2e), x3e(v.x3e) {}
};
} // namespace diagnostics
#endif
