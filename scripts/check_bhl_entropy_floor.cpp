// Regression checks using the original inline SRMHD solver and fixed final floors.
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

Real OriginalEntropy(const EOS_Data &eos, const Real d) {
  const Real log_s = log10(eos.sfloor1) + (log10(d)-log10(eos.rho1)) *
      (log10(eos.sfloor2)-log10(eos.sfloor1))/(log10(eos.rho2)-log10(eos.rho1));
  return fmax(eos.sfloor, pow(10.0, log_s));
}

int main() {
  EOS_Data eos{};
  eos.gamma = 5.0/3.0;
  eos.dfloor = 1e-13;
  eos.pfloor = 3.3333333333333335e-19;
  eos.sfloor = 1e-9; eos.sfloor1 = 3e-9; eos.sfloor2 = 1e-8;
  eos.rho1 = 10.0; eos.rho2 = 1000.0;
  // Existing cold gas must remain admissible under the original entropy law,
  // including the densities that triggered the unnecessary radiation trials.
  for (Real d : {0.003, 0.01, 0.1, 1.0, 10.0}) {
    const Real p = 1.1*fmax(eos.pfloor, OriginalEntropy(eos,d)*pow(d,eos.gamma));
    MHDPrim1D w{};
    w.d = d; w.e = p/(eos.gamma-1.0);
    w.vx = 0.1/sqrt(1.0-0.1*0.1);
    w.by = sqrt(2*1.53e-8/(10*(1-0.1*0.1)));
    HydCons1D h{};
    SingleP2C_IdealSRMHD(w,eos.gamma,h);
    MHDCons1D u{};
    u.d=h.d; u.e=h.e; u.mx=h.mx; u.my=h.my; u.mz=h.mz; u.by=w.by;
    HydPrim1D recovered{};
    bool df=false,ef=false,failed=false; int iterations=0;
    SingleC2P_IdealSRMHD(u,eos,SQR(u.mx)+SQR(u.my)+SQR(u.mz),SQR(u.by),
                        0.0,recovered,df,ef,failed,iterations);
    if (df || ef || failed ||
        fabs((eos.gamma-1.0)*recovered.e/p - 1.0) > 1e-5) return EXIT_FAILURE;
  }
  // Verify the final pressure floor in the actual solver, including rare gas.
  for (Real d : {1.0,1e-3,1e-6,1e-10}) {
    MHDCons1D u{};
    u.d=d; u.e=1e-30;
    HydPrim1D w{};
    bool df=false,ef=false,failed=false; int iterations=0;
    SingleC2P_IdealSRMHD(u,eos,0.0,0.0,0.0,w,df,ef,failed,iterations);
    const Real expected=fmax(eos.pfloor,OriginalEntropy(eos,d)*pow(d,eos.gamma));
    if (failed || !ef || !std::isfinite(w.e) ||
        fabs((eos.gamma-1.0)*w.e/expected - 1.0) > 1e-10) return EXIT_FAILURE;
  }
  std::cout << "PASS: original entropy law and cold-state admissibility; fixed "
            << "1e-13 density / 3.3333333333333335e-19 pressure floors in SRMHD C2P."
            << std::endl;
  return EXIT_SUCCESS;
}
