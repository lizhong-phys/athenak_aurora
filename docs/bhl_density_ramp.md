# BHL density decline from an existing checkpoint

Build `xin_bhl_density_ramp` on branch `bh_transient`. It inherits the uniform
`xin_bhl_xy_rad_corrected` wind, radiation boundary conditions and accretion
histories. It changes the upstream boundary reservoir, preserving the evolved
interior state when loading the checkpoint.

If the original checkpoint has `units/density_cgs = 1e-6` and `problem/rho0 = 1`,
the default physical upstream density is

```
elapsed <= 0 M:       rho_inf = 1e-6 g/cm^3
0 < elapsed < 30000:  rho_inf = 1e-6 * 10^(-elapsed/5000) g/cm^3
elapsed >= 30000 M:   rho_inf = 1e-12 g/cm^3
stop:                elapsed = 50000 M
```

`elapsed = mesh_time - density_ramp_start`. On the first continuation,
`density_ramp_start` defaults to the checkpoint time. That value is saved into
new restart files, so a later restart continues the same ramp and retains the
same absolute stopping time. The pgen sets `time/tlim` before Driver construction.
Fresh runs start the ramp at time zero. For example, a checkpoint at `60000 M`
ramps through `90000 M` and stops at `110000 M`.

The decrease is logarithmic (one decade per `5000 M`). Wind speed and temperature
remain fixed, with `p_inf = rho_inf * t0`. The incoming thermal radiation remains
at the original temperature. By default the injected magnetic field scales as
`B_y = B_y_initial * sqrt(rho_inf/rho0)` to preserve the original **gas** plasma
beta. Set `problem/density_ramp_scale_b=false` to keep the initial field strength.

## Build on Aurora

From the repository root, using the same Intel/MPI environment as the old run:

```bash
bash scripts/build_bhl_density_ramp_aurora.sh /path/to/build_bhl_density_ramp
```

The script uses the existing Aurora SYCL/PVC build settings. `BUILD_JOBS` controls
the build parallelism (default 104). The resulting binary is
`/path/to/build_bhl_density_ramp/src/athena`.

## First continuation in a new run directory

Copy the selected checkpoint and the partial input
`inputs/grmhd/bhl_density_ramp_restart.athinput` into the new run folder. For the
per-rank checkpoint format, copy **all** corresponding `rank_00000000`,
`rank_00000001`, ... files and keep their directory structure; use the same MPI
rank count as the original run.

In the existing PBS launcher, point the build/executable and input at these new
files. The executable arguments are:

```bash
/path/to/build_bhl_density_ramp/src/athena \
  -r /path/to/new_run/rst/rank_00000000/old_run.XXXXX.rst \
  -i /path/to/new_run/bhl_density_ramp_restart.athinput \
  -d /path/to/new_run
```

Run this command through the original MPI launcher and GPU binding configuration.
For a single shared restart file, pass that file directly to `-r` instead.
The `-i` file is a **partial override**: the rest of the original parameters and
mesh come from the checkpoint. Do not supply a full density-1e-12 input file.
The startup banner prints the start, decrease duration, hold, endpoint and field
prescription; check that it matches the chosen checkpoint.

Keep `units/density_cgs = 1e-6` and `problem/rho0 = 1`. The final density is
`1e-6` in these fixed code units. Changing the unit normalization during a
restart changes the meaning of stored gas, magnetic and radiation variables.

## Floors in physical units

`density_ramp_track_floors=true` updates the active EOS and coordinate floors;
it does not just edit the input file. The initial density/pressure/excision
floors come from the checkpoint plus the partial input. The pgen saves their
original values as `problem/density_ramp_dfloor0`, `pfloor0`, `dexcise0`, and
`pexcise0` (all names have the `density_ramp_` prefix). With
`f = rho_inf(t)/rho0`, their runtime values are the original values times `f`.
They therefore keep the same physical ratios to the prescribed inflow.

For the standard `rho0=1`, fixed `units/density_cgs=1e-6` setup:

| Ambient density, g/cm^3 | dfloor / dexcise | pfloor / pexcise | Entropy coefficient at ambient density |
|---|---:|---:|---:|
| 1e-6 | 1e-7 | 3.3333333333333335e-13 | original effective value |
| 1e-9 | 1e-10 | 3.3333333333333335e-16 | 1e-8 |
| 1e-12 | 1e-13 | 3.3333333333333335e-19 | 1e-5 |

The original entropy profile is saved separately as `density_ramp_sfloor0`,
`sfloor10`, `sfloor20`, `rho10`, and `rho20`. The standalone 1e-9 run uses
`sfloor=sfloor1=sfloor2=1e-10`; the 1e-12 run uses `1e-9`. Because the entropy
floor imposes `p >= s * rho^gamma`, converting a reference coefficient into the
unchanged checkpoint density unit requires

```
rho_anchor = rho_anchor_cgs / fixed_density_unit
s_anchor = s_reference * rho_anchor^(1-gamma)
```

The default **local entropy floor** (`density_ramp_local_entropy=true`) is
applied inside every MHD C2P calculation, including the relativistic root
iteration, final primitive recovery, FOFC and radiation C2P checks. Hydro's C2P
uses the same local law when this pgen enables it. At or above the original
ambient density, it retains the original density-dependent entropy floor. At
1e-9 and 1e-12 g/cm^3 it matches the converted standalone-run coefficients.
Between these anchors it interpolates logarithmically in local rest-mass density
and coefficient; below the lowest anchor it holds the final coefficient. The
higher-density gas left in the box is not assigned the much larger coefficient
appropriate to rarefied gas. Already rarefied material can be affected when this
local prescription is first enabled; this is an explicit change to the floor law.

Setting `density_ramp_local_entropy=false` instead applies a spatially uniform,
time-dependent entropy profile, interpolated through the same reference runs;
the density anchors `rho1/2` then also scale by `f`. This can artificially heat
dense gas that remains in the box. Local entropy is the default for the transient.

The pgen restores the current floors before Driver initialization, updates them
in `before_stagen` before reconstruction/FOFC/radiation coupling, and updates
them to the timestep endpoint in `after_timeintegrator` before output and AMR.
As for the boundary forcing, stage floors use the current mesh time (a one-step
timing uncertainty). Checkpoint headers record current active values plus the
original anchors; resuming never treats current floors as new initial anchors.
The EOS additions are disabled by default for all other problem generators.

The partial input now supplies the **initial** density/pressure/excision floors,
not their final low values. Repeated `-i` overrides are safe: saved original
anchors plus absolute ramp time restore the correct current floor values. If
PBS command-line floor arguments conflict with these settings, update them;
for this pgen, saved `problem/density_ramp_*0` anchors govern the subsequent
schedule. Other temperature/magnetization and radiation opacity limiter settings
are inherited, independently of this density/entropy/excision prescription.
To use fixed floors instead, set `density_ramp_track_floors=false` and provide
sufficiently low `mhd/dfloor` and `mhd/pfloor` yourself.

## Later restarts and diagnostics

For subsequent resumes, use a **new ramp checkpoint** with `-r`. Its saved
parameters already include the schedule and floors. The same partial input may
also be supplied: it omits the saved ramp start. Do not set `density_ramp_start`
to the new checkpoint time. Use a separate output folder or retain the existing
output counters if continuing to append history; the provided input resets the
history header for the initial new folder.

The existing `mdot`, `edot`, `ldot`, and magnetic-flux histories are retained.
Eight columns are added: `rho_inf` (code density), `rho_cgs`, `By_inf` (code
field), and `t_ramp` (M). They describe the supplied reservoir, not the
instantaneous density near the hole. The additional columns `dfloor`, `pfloor`,
`sf_at_inf` and `dexcise` report the active EOS/coordinate density, pressure,
effective entropy coefficient at the prescribed ambient density, and excision
density floor (all in fixed code units). Actual local entropy coefficients
elsewhere depend on the local gas density.

The final `20000 M` is a **boundary** hold. The rarefied flow takes time to reach
the hole and relax. With the old upstream face at `x=-1024 M` and `v_wind=0.1`,
the nominal advection delay is `10240 M`; the low-density boundary therefore
corresponds to roughly `9760 M` at the hole before the default stop, subject to
the actual flow response. To deliberately extend the hold, override
`problem/density_ramp_hold`; to choose a custom absolute stopping time, set
`problem/density_ramp_set_tlim=false` and supply `time/tlim`.

Boundary forcing uses the current mesh time because AthenaK's user boundary
callback has no RK-stage time argument. Its timing uncertainty is of order one
step; resolve the density e-folding time (`30000 / ln(1e6) = 2171.47 M`) with
the timestep. No global density or energy rescaling is applied.

## Local integration check

Build serial, double-precision binaries for `xin_bhl_xy_rad_corrected` and
`xin_bhl_density_ramp`, then run:

```bash
python3 scripts/check_bhl_density_ramp_restart.py /path/to/old_binary /path/to/ramp_binary
```

The check creates a tiny Kerr GR-MHD+radiation checkpoint, verifies a zero-step
continuation preserves all active gas/radiation cells and magnetic faces, tests
the actual inflow ghost density and both magnetic prescriptions, and restarts
midway through a compressed-time ramp to verify the saved start and final hold.
It also checks the default `30000+20000 M` end time, fractional timestamp
precision, finite evolved state, active floor updates through restart/hold,
repeated partial-input restarts, both entropy reference points, and rejection
of incompatible pressure floors in fixed-floor mode. The companion
`scripts/check_bhl_entropy_floor.cpp` checks the original disabled behavior,
continuity and dense-gas preservation, and sends cold states through the real
SRMHD C2P solver to verify it actually applies the local entropy floor.
This checks restart/boundary behavior on a CPU; the Aurora SYCL build and the
full production run still need to be performed on Aurora.
