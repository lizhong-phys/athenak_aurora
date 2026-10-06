#!/usr/bin/env python3
"""Serial, double-precision restart integration check (Python standard library).

Usage: python3 scripts/check_bhl_density_ramp_restart.py OLD_BINARY RAMP_BINARY
OLD_BINARY must use xin_bhl_xy_rad_corrected; RAMP_BINARY uses xin_bhl_density_ramp.
Runs a tiny 3D GR-MHD+radiation wind and a compressed-time density ramp.
"""

import array
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


def read_restart(path):
    """Read this fork's single-block, double-precision checkpoint layout."""
    raw = path.read_bytes()
    offset = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
    header = raw[:offset].decode()
    params, block = {}, None
    for line in header.splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("<"):
            block = line.strip("<>")
        elif "=" in line:
            name, value = line.split("=", 1)
            params[f"{block}/{name.strip()}"] = value.strip()
    nmb, _ = struct.unpack_from("=ii", raw, offset)
    assert nmb == 1, "Check requires a single MeshBlock"
    offset += 8 + 9*8 + 19*4  # two ints, RegionSize, mesh RegionIndcs
    indcs = struct.unpack_from("=19i", raw, offset)
    offset += 19*4
    time, dt, cycle = struct.unpack_from("=ddi", raw, offset)
    offset += 20 + 16 + 4 + 8  # time/dt/cycle, logical location, cost, data size
    state = array.array("d")
    state.frombytes(raw[offset:])
    assert all(math.isfinite(x) for x in state), "Nonfinite checkpoint state"
    return params, time, dt, cycle, indcs, state


def run(binary, directory, args, succeeds=True):
    directory.mkdir()
    result = subprocess.run([str(binary), *args, "-d", str(directory)],
                            text=True, capture_output=True, timeout=60)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    if succeeds:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "Expected incompatible-floor rejection"
        assert "fixed gas/excision floors" in result.stdout, (
            result.stdout + result.stderr)
        return None
    restart = sorted((directory / "rst").glob("*.rst"))[-1]
    return read_restart(restart), restart


def history(directory):
    lines = next(directory.glob("*.user.hst")).read_text().splitlines()
    labels = re.findall(r"\[\d+\]=(\S+)", lines[1])
    return [dict(zip(labels, map(float, line.split()))) for line in lines
            if line and not line.startswith("#")]


def close(actual, expected, tolerance=2e-6):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=1e-30), (
        actual, expected)


def main():
    old, ramp = (Path(arg).resolve() for arg in sys.argv[1:])
    repo = Path(__file__).resolve().parents[1]
    override = repo / "inputs/grmhd/bhl_density_ramp_restart.athinput"
    with tempfile.TemporaryDirectory(prefix="bhl-ramp-check-") as tmp:
        root = Path(tmp)
        seed_input = root / "seed.athinput"
        seed_input.write_text("""<job>
basename = seed
<mesh>
nghost = 4
nx1 = 8
nx2 = 8
nx3 = 8
x1min = -32
x2min = -32
x3min = -32
x1max = 32
x2max = 32
x3max = 32
ix1_bc = user
ox1_bc = user
ix2_bc = user
ox2_bc = user
ix3_bc = user
ox3_bc = user
<meshblock>
nx1 = 8
nx2 = 8
nx3 = 8
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.001
tlim = 0.048123456789
nlim = -1
ndiag = 100
<coord>
general_rel = true
a = 0.9
excise = true
dexcise = 1e-7
pexcise = 3.3333333333333335e-13
<mhd>
eos = ideal
gamma = 1.6666666666666667
reconstruct = plm
rsolver = hlle
dfloor = 1e-7
pfloor = 3.3333333333333335e-13
sfloor = 1e-9
sfloor1 = 3e-9
sfloor2 = 1e-8
rho1 = 10
rho2 = 1000
fofc = true
gamma_max = 20
<radiation>
reconstruct = plm
nlevel = 2
rotate_geo = false
kappa_s = 0.34
kappa_a = 0
kappa_p = 0
compton = true
rad_wlimit = true
<units>
bhmass_msun = 100
density_cgs = 1e-6
mu = 0.6
<problem>
rho0 = 1
t0 = 1.53e-8
v_wind = 0.1
beta_target = 10
user_hist = true
<output1>
file_type = hst
user_hist_only = true
dt = 0
dcycle = 1
data_format = %24.16e
<output2>
file_type = rst
dt = 0
dcycle = 1
single_file_per_rank = false
""")
        seed, seed_path = run(old, root / "seed", ["-i", str(seed_input)])
        _, start, _, seed_cycle, indcs, seed_state = seed
        run(ramp, root / "bad_floors", ["-r", str(seed_path), "-i", str(override),
            "mhd/dfloor=1e-6"], succeeds=False)

        base = ["-r", str(seed_path), "-i", str(override)]
        default, _ = run(ramp, root / "defaults", base + [f"time/nlim={seed_cycle}"])
        params, time, _, _, _, state = default
        close(float(params["problem/density_ramp_start"]), start, tolerance=1e-13)
        close(float(params["time/tlim"]), start + 50000, tolerance=1e-13)
        close(time, start)
        close(float(params["units/density_cgs"]), 1e-6)
        close(float(params["mhd/dfloor"]), 1e-13)
        close(float(params["mhd/pfloor"]), 3.3333333333333335e-19)
        # Use the same physical constants as units.hpp to derive the ORIGINAL
        # total beta. beta_target=10 in the old pgen denotes gas beta.
        c = 2.99792458e10
        t0 = 1.53e-8
        temperature = t0*0.6*1.660538921e-24*c*c/1.3806488e-16
        prad = 7.56573325e-15*temperature**4/(3*1e-6*c*c)
        beta_total = 10*(1+prad/t0)
        close(float(params["problem/density_ramp_beta_total0"]), beta_total,
              tolerance=1e-13)

        # Compare every active conserved gas/radiation cell and active magnetic face.
        # Only ghost zones should change during this zero-step continuation.
        ng, nx, ny, nz, is_, ie, js, je, ks, ke = indcs[:10]
        n1, n2, n3 = nx + 2*ng, ny + 2*ng, nz + 2*ng
        nc = n1*n2*n3
        nbx, nby, nbz = (n1+1)*n2*n3, n1*(n2+1)*n3, n1*n2*(n3+1)
        arrays = [(0, 5, n1, n2, is_, ie, js, je, ks, ke),
                  (5*nc, 1, n1+1, n2, is_, ie+1, js, je, ks, ke),
                  (5*nc+nbx, 1, n1, n2+1, is_, ie, js, je+1, ks, ke),
                  (5*nc+nbx+nby, 1, n1, n2, is_, ie, js, je, ks, ke+1),
                  (5*nc+nbx+nby+nbz, 42, n1, n2, is_, ie, js, je, ks, ke)]
        for offset, nv, ni, nj, i0, i1, j0, j1, k0, k1 in arrays:
            nk = n3+1 if offset == 5*nc+nbx+nby else n3
            for v in range(nv):
                for k in range(k0, k1+1):
                    for j in range(j0, j1+1):
                        for i in range(i0, i1+1):
                            index = offset + ((v*nk+k)*nj+j)*ni+i
                            close(state[index], seed_state[index], tolerance=1e-11)

        fast = ["problem/density_ramp_duration=0.048", "problem/density_ramp_hold=0.032"]
        middle, middle_path = run(ramp, root / "middle", base + fast +
                                  [f"time/nlim={seed_cycle+3}"])
        params, middle_time, _, middle_cycle, _, _ = middle
        assert start < middle_time < start + 0.048
        probe, _ = run(ramp, root / "probe", ["-r", str(middle_path),
                       f"time/nlim={middle_cycle}"])
        _, _, _, _, _, probe_state = probe
        rho = 10**(-6*(middle_time-start)/0.048)
        ghost = (ks*n2+js)*n1 + is_-1
        # Identical metric and normal-frame velocity make D linear in rho.
        close(probe_state[ghost], rho*seed_state[ghost], tolerance=1e-11)
        by_ghost = 5*nc + nbx + (ks*(n2+1)+js)*n1 + is_-1
        by0 = math.sqrt(2*1.53e-8/(10*(1-0.1**2)))
        close(probe_state[by_ghost],
              by0*math.sqrt((rho*t0+prad)/(t0+prad)), tolerance=1e-11)
        fixed, _ = run(ramp, root / "fixed_b", ["-r", str(middle_path),
                       f"time/nlim={middle_cycle}", "problem/density_ramp_scale_b=false"])
        close(fixed[-1][by_ghost], by0, tolerance=1e-11)
        close(history(root / "fixed_b")[-1]["beta_tot"],
              10*(rho+prad/t0), tolerance=1e-11)

        finish, _ = run(ramp, root / "finish", ["-r", str(middle_path), "time/nlim=-1"])
        params, final_time, _, _, _, _ = finish
        close(float(params["problem/density_ramp_start"]), start, tolerance=1e-13)
        close(final_time, start+0.080)
        close(float(params["time/tlim"]), start+0.080)
        close(float(params["mhd/dfloor"]), 1e-13)
        close(float(params["mhd/pfloor"]), 3.3333333333333335e-19)
        close(float(params["coord/dexcise"]), 1e-13)
        close(float(params["coord/pexcise"]), 3.3333333333333335e-19)
        assert params["problem/density_ramp_track_floors"] in ("false", "0")
        assert params["problem/density_ramp_local_entropy"] in ("false", "0")
        rows = history(root / "middle") + history(root / "finish")
        for row in rows:
            elapsed = row["time"] - start
            expected = max(1e-6, 10**(-6*elapsed/0.048))
            close(row["rho_inf"], expected)
            close(row["rho_cgs"], expected*1e-6)
            close(row["By_inf"], by0*math.sqrt((expected*t0+prad)/(t0+prad)))
            close(row["pgas_inf"], expected*t0)
            close(row["prad_inf"], prad)
            close(row["beta_tot"], beta_total, tolerance=1e-11)
            close(row["t_ramp"], elapsed)
            close(row["dfloor"], 1e-13)
            close(row["pfloor"], 3.3333333333333335e-19)
            close(row["dexcise"], 1e-13)
        close(rows[-1]["sf_at_inf"], 1e-9)
        assert sum(row["time"] >= start+0.048 for row in rows) >= 2
        # A repeated -i must preserve the fixed floors and saved absolute clock.
        repeated, _ = run(ramp, root / "repeated_input", ["-r", str(middle_path),
            "-i", str(override), f"time/nlim={middle_cycle}"] + fast)
        repeated_params = repeated[0]
        close(float(repeated_params["mhd/dfloor"]), 1e-13)
        close(float(repeated_params["mhd/pfloor"]), 3.3333333333333335e-19)
        # Probe exactly the 1e-9 reference point without advancing the flow.
        mid_start = middle_time - 0.024
        exact_mid, _ = run(ramp, root / "exact_mid", ["-r", str(middle_path),
            f"time/nlim={middle_cycle}", f"problem/density_ramp_start={mid_start:.17g}"])
        mid_rows = history(root / "exact_mid")
        close(mid_rows[-1]["rho_inf"], 1e-3)
        close(mid_rows[-1]["dfloor"], 1e-13)
        close(mid_rows[-1]["pfloor"], 3.3333333333333335e-19)
        close(mid_rows[-1]["sf_at_inf"], 1e-9)
        # Retired adaptive settings from an older header cannot enable a new
        # entropy prescription or a floor-update task in the fixed-floor pgen.
        legacy, _ = run(ramp, root / "legacy_options", ["-r", str(middle_path),
            f"time/nlim={middle_cycle}", "problem/density_ramp_track_floors=true",
            "problem/density_ramp_local_entropy=true"])
        for name, expected in (("sfloor", 1e-9), ("sfloor1", 3e-9),
                               ("sfloor2", 1e-8), ("rho1", 10.0), ("rho2", 1000.0)):
            close(float(legacy[0]["mhd/"+name]), expected)
        assert legacy[0]["problem/density_ramp_track_floors"] in ("false", "0")
        assert legacy[0]["problem/density_ramp_local_entropy"] in ("false", "0")
        print("PASS: old-checkpoint continuation preserves active state; default 30000+20000 M "
              "schedule; original total-pressure beta and actual inflow ghosts in both B modes; "
              "restart persistence; "
              "logarithmic history/hold; fixed density/pressure/excision floors; "
              "original entropy law; repeated-input restart; floor rejection; "
              "finite MHD+radiation state.")


if __name__ == "__main__":
    main()
