// Host-side checks of the same inline SRMHD solver used by GRMHD.
// Compile with the serial AthenaK build's include flags and Kokkos libraries.
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

void CheckClose(Real actual, Real expected) {
  if (!std::isfinite(actual) || std::abs(actual/expected - 1.0) > 1e-10) {
    std::cerr << "actual=" << actual << " expected=" << expected << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  EOS_Data eos{};
  eos.gamma = 5.0/3.0;
  eos.sfloor = 1e-9;
  eos.sfloor1 = 3e-9;
  eos.sfloor2 = 1e-8;
  eos.rho1 = 10.0;
  eos.rho2 = 1000.0;
  // Verify compatibility of the default-disabled path across density regimes.
  for (Real d : {1e-13, 1e-6, 1e-3, 1.0, 10.0, 1000.0}) {
    const Real log_s = log10(eos.sfloor1) + (log10(d)-log10(eos.rho1)) *
        (log10(eos.sfloor2)-log10(eos.sfloor1)) /
        (log10(eos.rho2)-log10(eos.rho1));
    CheckClose(eos.EntropyFloor(d), fmax(eos.sfloor, pow(10.0, log_s)));
  }
  const Real dense_s = eos.EntropyFloor(1.0);
  eos.bhl_local_entropy = true;
  eos.bhl_rho_start = 1.0;
  eos.bhl_rho_mid = 1e-3;
  eos.bhl_rho_low = 1e-6;
  eos.bhl_s_mid = 1e-10*pow(1e-3, 1.0-eos.gamma);
  eos.bhl_s_low = 1e-9*pow(1e-6, 1.0-eos.gamma);
  CheckClose(eos.EntropyFloor(1.0), dense_s);
  CheckClose(eos.EntropyFloor(10.0), 3e-9);
  CheckClose(eos.EntropyFloor(1e-3), 1e-8);
  CheckClose(eos.EntropyFloor(1e-6), 1e-5);
  CheckClose(eos.EntropyFloor(1e-10), 1e-5);
  // Continuity at all three anchors, including both sides of the middle node.
  for (Real d : {1.0, 1e-3, 1e-6}) {
    const Real at = eos.EntropyFloor(d);
    CheckClose(eos.EntropyFloor(d*(1.0-1e-12)), at);
    CheckClose(eos.EntropyFloor(d*(1.0+1e-12)), at);
  }
  // A deliberately cold rest state must actually be heated to the new entropy
  // floor by C2P, rather than merely reporting the right diagnostic coefficient.
  for (Real d : {1.0, 1e-3, 1e-6}) {
    eos.dfloor = 1e-7*d;
    eos.pfloor = 3.3333333333333335e-13*d;
    MHDCons1D cons{};
    cons.d = d;
    cons.e = d*1e-16/(eos.gamma-1.0);
    HydPrim1D prim{};
    bool density_used = false, energy_used = false, failure = false;
    int iterations = 0;
    SingleC2P_IdealSRMHD(cons, eos, 0.0, 0.0, 0.0, prim,
                        density_used, energy_used, failure, iterations);
    if (failure || !energy_used) return EXIT_FAILURE;
    CheckClose(prim.d, d);
    CheckClose((eos.gamma-1.0)*prim.e,
               fmax(eos.pfloor, eos.EntropyFloor(d)*pow(d, eos.gamma)));
  }
  std::cout << "PASS: original entropy law unchanged when disabled; reference "
            << "conversion; dense-gas preservation; continuity; cold SRMHD C2P "
            << "uses the local entropy floor." << std::endl;
  return EXIT_SUCCESS;
}
