# Bondi runtime energy diagnostics

Development branch: **`bh_transient_energy_check`**. The `bh_transient` branch is
unchanged. This extends the BHL diagnostic machinery already committed on the energy
branch: new Bondi pgens/output alias, runtime sampling metadata, repair flags,
radiation bookkeeping correction and supported-configuration guards.

These pgens continue the existing Bondi physics while recording the quantities needed
for a BHL-style energy-dissipation check:

| Field | Diagnostic pgen | Original physics |
| --- | --- | --- |
| Vertical Bz | `xin_bondi_rad_corrected_energy_check` | `xin_bondi_rad_corrected` |
| Horizontal By | `xin_bondi_By_energy_check` | `xin_bondi_By` |

The original pgen files are unchanged. The new pgens retain their initialization,
reservoir boundaries, history functions and restart initialization; their only new
behavior is an identifying startup message. The actual measurements require the
shared solver hooks described below. Commit those hooks together with the pgens.

## Activation and output

Select the appropriate pgen with CMake's `-D PROBLEM=...`, using a new build directory.
The diagnostic switch defaults to **false**, including when a diagnostic pgen is
selected. The continuation helper enables it in a newly generated input:

```text
<problem>
energy_diagnostics = true
energy_diagnostics_start = CHECKPOINT_TIME
energy_diagnostics_dt = 1.0

<outputN>
file_type = bin
id = ediag
variable = bondi_energy_diag
dt = 1.0
last_time = CHECKPOINT_TIME
file_number = 0
```

Replace `CHECKPOINT_TIME` with the actual checkpoint time and `N` with a new output
number. Keep the diagnostic and output cadences aligned. `analysis_4` provides
`prepare_bondi_diagnostics.py` to extract runtime parameters from an explicitly chosen
checkpoint, disable inherited outputs, and configure a new output tree. Its defaults
are a 10 M continuation and 1 M cadence. Postprocessing separately defaults to the
last 10 M of available samples. No past solver diagnostics can be recovered if they
were never recorded.

## What changed in the solver

| Files | Purpose |
| --- | --- |
| `src/diagnostics/energy_diagnostics.{hpp,cpp}` | Own detached arrays, schedule sampled timesteps, accumulate RK-weighted energy contributions, and calculate derived heating and morphology fields. |
| `src/mesh/meshblock_pack.{hpp,cpp}` | Own the optional tracker and register its before-step, before-stage and after-step tasks. No diagnostic arrays or tasks are created when disabled. |
| `src/mhd/mhd_tasks.cpp`, `mhd_update.cpp`, `mhd.hpp` | Select instrumented GR-HLLE/FOFC only on sampled steps; measure flux, coordinate-source and other-source contributions. |
| `src/mhd/mhd_fluxes_diag.cpp`, `mhd_fofc_diag.cpp`, `rsolvers/hlle_grmhd_diag.hpp` | Preserve production flux/electric-field calculations while additionally recording detached entropy, internal-energy, velocity, HLLE and FOFC face quantities. |
| `src/eos/ideal_grmhd.cpp`, `eos.hpp` | Call the original primitive recovery, record its actual conserved-energy change and copy its floor, ceiling, failure and excision decisions to diagnostic flags. |
| `src/radiation/radiation_source.cpp`, `radiation_update.cpp`, `radiation.hpp` | Record accepted gas/radiation coupling and radiation spatial/angular transport and update corrections. |
| `src/outputs/outputs.hpp`, `basetype_output.cpp` | Register `bondi_energy_diag`; keep the upstream `bhl_energy_diag` alias for compatibility. |
| `src/CMakeLists.txt` | Compile the new tracker and instrumented flux routines. |

The tracker runs for the solver timestep that crosses each requested sample time.
At its start it saves gas, radiation and physical-state data. At each RK stage it
weights previous increments with the solver's RK coefficient and records the new
contributions. After the timestep it converts increments to rates and writes the
thermal equations, ledgers, sensor fields and sample metadata to detached output
arrays. These arrays never feed back into the evolved state or timestep selection.

Radiation coupling is bracketed **after** the source routine's preliminary primitive
recovery. This prevents a recovery repair from being counted again as radiation
exchange. The original accepted source update and opacity/limiter formulas are used.

## Thermal budget and portions

The primary residual is

```text
q_diss_energy = q_storage + q_advection - q_compression + qent_rad
```

- Storage is the change in `e u^t` over the sampled solver timestep.
- Advection uses the diagnostic internal-energy face flux.
- Compression is reversible pressure work, `-p div_4(u)`.
- `qent_rad` is positive for energy lost by gas to radiation; it is obtained by
  contracting the actual accepted gas four-momentum change with the saved fluid
  four-velocity. AthenaK stores gas energy as `T^t_t + D`, so the signs follow that
  convention.

A separate entropy calculation uses `s = log(p/rho^gamma)/(gamma-1)` and the solver
face transport to obtain `qent_num`. Agreement between the energy and entropy routes
is required before interpreting the residual as heating. Separate gas and radiation
operator ledgers expose bookkeeping closure, repair contributions and source terms.
HLLE and FOFC entries are parts of the flux ledger, not extra terms to add to it.

The Python analyzer in `analysis_4` partitions positive heating in this order:

1. **Magnetic/current sheet:** positive magnetic-loss residual with sufficiently
   strong current and an antiparallel field reversal; capped at available heating.
2. **Shock:** remaining heating in converging gas with a pressure jump.
3. **Turbulent:** remaining nonshock heating where Favre transfer is positive at both
   three- and five-cell filter widths, with sufficient shear or vorticity.
4. **Other:** the unassigned positive remainder.

These are ordered associations with numerical dissipation in ideal GRMHD. They are
not independent measurements of physical resistivity, viscosity or reconnection.
Compression and radiation exchange have already been separated and are not additional
members of this four-part heating partition. The analyzer publishes null fractions
when finite-field/entropy-transport checks fail or integrated energy–entropy
disagreement exceeds 30%.

## Sampling and spatial support

`sample_time`, `sample_dt` and `sample_id` identify the completed solver step. A dump
is **one sampled timestep**, not an integral over the output interval. Initialization,
stale end-of-run outputs and duplicate samples are rejected by the analyzer. The
last-10-M selection operates on the recorded sample time. Sparse means weight samples
equally and do not imply continuous coverage.

Diagnostic face arrays are not refluxed across static-refinement interfaces. To avoid
using those interfaces quantitatively, `stencil_edge` marks two active-cell layers
along every block boundary. The analyzer excludes these layers and repair-contaminated
neighbor stencils, requires complete filter support for turbulence, and reports the
excluded volume, mass and heating. Its fractions describe the retained support, not an
unqualified whole-domain energy budget. Physical excision uses Kerr–Schild radius.

AthenaK's binary fields are float32. The analyzer preserves residuals calculated in
solver precision before serialization, instead of subtracting separately rounded large
terms. It also records a serialized budget-identity check.

## Supported configurations and limits

Diagnostics require radiation plus ideal MHD, fixed GR, HLLE, RK2 and a nonadaptive
mesh. Runtime guards reject unsupported extra diffusion, external source/driving,
shearing/orbital terms, KO dissipation and velocity damping. Nonradiative Bondi is not
supported by this tracker.

Total accepted radiation exchange is measured. The compatibility fields
`dE_gas_abs`, `dE_gas_compt` and `dE_rad_reject` are reserved and unpopulated; their zeros
must not be interpreted as measured zero absorption, Compton exchange or rejected
energy. Radiation-update repair flags are inferred using a residual tolerance and can
be sensitive to roundoff. Magnetic-loss derivatives are centered diagnostics, so their
association with physical reconnection requires additional resolution checks.

Both pgens passed local serial CPU compilation and full checkpoint-payload comparisons
with diagnostics on/off, including fresh runs, restarts and static refinement. The
comparisons were bitwise identical. The continuation and analysis scripts passed their
30 focused tests. Generated test inputs, outputs and validation logs are intentionally
not part of this source change. Aurora MPI/SYCL compilation, runtime performance and
scientific convergence remain to be checked on the target platform.
