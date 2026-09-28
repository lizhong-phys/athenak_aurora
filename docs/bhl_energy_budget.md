# Passive BHL energy budget — implementation v2 (isolated observers)

This is a new, self-contained workflow. It does not depend on `frontier_server`.
The executable marker is `bhl-passive-budget-v2-isolated`. The unchanged binary
field layout remains budget schema 1; implementation version and data schema are
different identifiers.

## What changes, and what does not

The BHL energy-check pgen uses the same initial conditions, wind boundaries,
radiation coupling, primitive recovery, floors, Riemann solver, and CT evolution.
New arrays never feed the evolved state, timestep, reconstruction, or refinement.
In v1 the diagnostic HLLE/FOFC kernels also wrote production fluxes and EMFs.
Although their evolution expressions matched the originals, that was not an
isolated observer: diagnostic code could change GPU compiler optimization and
subsequent nonlinear recovery/limiter decisions. Frontier's two-step comparison
reported a 0.7048 relative difference in a rank's maximum internal energy. The
summary does not identify the first diverging operation or establish its cause.

Version 2 always calls the original CalculateFluxes and FOFC kernels. A separate
reconstruction computes diagnostic currents, without any production-flux or EMF
output arguments. The original FOFC's accepted fluxes are read before/after its
correction. An observer reads the actual flags before they are cleared, updates
each diagnostic face once, and uses the accepted mass flux for upwind entropy
and internal energy. It performs no additional floor-test/C2P and does not alter
production flags. Original HLLE, FOFC, reconstruction, source, recovery, and CT
arithmetic is unchanged. This removes the replacement-kernel failure path;
Frontier passivity must still be confirmed rather than inferred from CPU tests.

Two switches are required for the continuous ledger:

```
<problem>
energy_diagnostics = true
energy_budget = true
energy_diagnostics_dt = 1
```

With both absent/false there is no diagnostic object. The optional
`energy_fingerprint=true` switch enrolls a read-only end-of-run check independently
of those switches. It hashes active conserved/scalar fields, all active magnetic
faces, and every active radiation intensity. It also records means, absolute
means, RMS, extrema, and finite/NaN/Inf counts for each component of those fields
and the gas primitives. Ghost zones and scratch arrays are excluded. Each rank
writes a compact JSON file; this read-only host summary does not alter evolution.

One prerequisite repair is **not** switch-gated: the split-restart reader now
passes `single_file_per_rank` to `GetPosition`. Otherwise it calls MPI on a FILE*
and can abort before reaching the solver. This is an I/O argument correction,
not a change to the evolved equations.

The pre-existing diagnostic sidecar flux exchange is now connected to the
radiation-MHD task graph. Receives are posted only on recorded steps, and sends
and receives are completed before reusing the buffers. Primitive-repair flags
are recorded only for the actual evolved array, not coarse boundary scratch arrays.

## Physics and numerical meaning

Use the same comoving heat-rate convention throughout:

```
q_total = d_t(e u^t) + div(e u^i) - q_compression + Lambda_rad
q_compression = -p div_4(u)
```

Positive Lambda means gas loses heat to radiation. Compression is reversible;
storage and advection redistribute internal energy. They are NOT mechanisms of
irreversible dissipation.

The existing internal-energy and entropy routes are both retained. Transport uses
the reconstructed face states, HLLE fan, FOFC replacements, actual directional
cell widths, RK2 weights, and fine-to-coarse flux correction. Entropy transport
is a passive diagnostic flux, not a separately evolved production entropy law.
FOFC entropy/internal-energy sidecars retain the previous mass-flux-upwind
prescription. Pressure work remains a discrete face-velocity p-div-u estimate;
we do not claim that energy/entropy equality is exact at shocks.

The new EM route forms

```
T_EM^mu_nu = b^2 u^mu u_nu + (b^2/2) delta^mu_nu - b^mu b_nu
R_nu = [Delta T_EM^t_nu - I_nu(flux+geometry) - Delta_EM(source)] / dt
q_mag_budget = u_mid^nu R_nu
q_mech_budget = q_total - q_mag_budget
```

Here I_nu is RK-integrated -div(F_EM)+G_EM. EM face fluxes are the linear
EM-stress part of the *same* HLLE fan; first-order corrected faces use LLF
with the same first-order states and speed prescription. Geometric sources are
(1/2) T_EM^{ab} partial_nu(g_ab); they vanish for nu=t in the stationary metric.
Cartesian Kerr-Schild has sqrt(-g)=1. The midpoint four-velocity is normalized
with the local metric.

The source correction measures EM-state changes from the existing pre-radiation
primitive recovery to the existing post-radiation/end-stage recovery, without
performing another recovery. Post-source repairs are included in that correction
and flagged; repaired cell-windows are reported separately. This avoids mistaking
radiation-induced changes of the ideal electric field for magnetic dissipation.

The spatial and temporal components together separate comoving heat from
coordinate energy exchange. With C_nu=-R_nu:

```
R_0 = mag_heat_inf + em_bulk_work_inf
mag_heat_inf = (-u_mid,0) q_mag_budget
em_bulk_work_inf = R_0 + u_mid,0 q_mag_budget
```

These last two are stationary-coordinate/Killing-energy rates. Do not add them
to the comoving thermal equation. The bulk-work remainder can be reversible;
a decrease in magnetic energy alone is NOT proof of heating.

### What the split can and cannot establish

This is an **effective, scheme-dependent numerical energy-budget estimate**,
not eta J^2 and not a shock/current morphology classifier. A cell can have both
signed magnetic and mechanical contributions. No positive clipping or capping
is used to force a percentage. Mechanical heating is a remainder, not an
independent measurement.

The HLL EM-stress split is **not a proven CT-compatible discrete Poynting theorem**.
Its residual can include HLL/CT inconsistency, nonlinear RK mixing, and finite
time-discretization errors. The corrected-edge CT check tests the induction
update bookkeeping; it does NOT by itself validate the magnetic heating split.
The endpoint projection spread is a sensitivity indicator, not an error bound.
Smooth-flow, resolution/timestep, and physically dissipative controlled tests
are required before quoting a physical magnetic fraction. Passing the analysis
gates is necessary, not sufficient. Do not interpret q_mag+q_mech=q_total as an
independent closure test.

## Sampling and output

Every timestep is measured; each output is an actual-duration-weighted interval
average. Products, projections, positive/negative parts, and absolute closure
errors are computed BEFORE time averaging and BEFORE float32 binary conversion.
Repair flags are ORed over the window; CT error is the worst step in the window.
A roundoff-sized final timestep is accumulated into the final window rather than
overwriting it. The analysis verifies interval coverage, not just file counts.

Default: 10 rg/c, ten interval diagnostic volumes, one ordinary end-state volume.
No extra 3-D volumes are required by the default passivity preflight: two short
runs produce per-rank main-field summaries instead. The gate compares these at a
default relative tolerance of 1e-10, with zero absolute tolerance. For each
statistic its scale is the larger absolute value or either state's absolute mean
(with a 1e-300 numerical floor). Time, cycle, dt, mesh, and cell/non-finite counts
must agree. Hash mismatches are informational only. Matching non-finite counts
are reported, not interpreted as proof that a run is healthy. Summary agreement
cannot bound cellwise differences or prove diagnostic accuracy. Old hash-only
records cannot be compared numerically; rebuild with `major-field-summary-v1`.
Local CPU regression additionally compares complete restart payload bytes.

The compact diagnostic output contains 47 float32 fields. Size is approximately
47 * 4 * active_cell_count bytes per volume, plus headers; there are ten volumes.
The ordinary end-state and headers are additional. Use the actual model's leaf
cell count; neither a common mesh nor a fixed terabyte estimate is assumed.

## Analysis and rejection

The default region is 3 <= r_KS < 100 rg, with simultaneous inner-cut checks at
2, 3, and 5 rg. Adjust RMIN/RMAX when submitting analysis. All leaf cells are
considered; block-edge strips and ambient-gas classifications are NOT used to
select the budget.

Exclude non-finite/nonphysical data and cell-windows with floors, ceilings,
excision, failed recovery, or radiation repairs. FOFC alone is not excluded.
Never discard a cell just because energy/entropy closure is poor.
Report retained volume, finite baryon mass (using D=rho u^t), and positive
heating separately. A flag at any step rejects that cell's entire output window;
this deliberately conservative loss of support is visible in the report.

All percentages are integrals of energy rates weighted by dV*dt, not volume or
cell-count fractions. Signed net fractions are distinct from separately reported
gross positive and negative heating; gross positive magnetic and mechanical
rates need not add to gross positive total heating.

Analysis gates (explicit QA choices, not physical laws): energy/entropy L1
mismatch <=20% of gross dissipation; conservative gas relative error <=1e-6;
CT bookkeeping error <=1e-10; projection spread <=10% of gross dissipation.
Large magnetic cancellation or fractions outside [0,1] suppress the candidate
partition. No failed gate is repaired by reclassifying cells. The 1e12 run is
always an adiabatic/control budget, with no mechanism fraction claim.

NaN/Inf flags check the evolved conserved state, recovered primitives, magnetic
field, and every radiation intensity at each recorded step end. This does not
inspect every transient nonlinear-iteration state. Non-finite diagnostics are
also counted independently. Overflow cells do not silently enter physical sums.
