# SA-PQ paper reference implementation

This is a reference implementation of the critical-front allocation and
monitoring procedures described in Sections 4.3 and 4.4 and Algorithms 1-2.
The package contains Python source, the published algorithm settings and
interface documentation; it contains **no empirical or example datasets,
business identifiers, private paths, business costs, result files, saved
populations, checkpoints or logs**.

## Scope

The package accepts an already ranked critical nondominated front and an
already updated historical monitoring archive. It implements integer quota
allocation, recursive local-epsilon selection, quota protection, state
monitoring and adaptive ratios. F1, F2 and F3 share one implementation.
FIXED is the same local selection with dynamic allocation disabled.

Model evaluation, task/ship decoding, initialization, genetic operators,
population ranking, construction of the historical nondominated archive and
TOPSIS evaluation are outside this delivery scope. Reproducing the full
reported results requires a complete local application and the private inputs
described in the data-availability statement of the manuscript; that part is
addressed separately in that statement.

## Installation

The manuscript reports Python 3.12.6 and NumPy 2.4.4, and the dependency file
locks that NumPy version. Install with
`python -m pip install -r requirements.txt` and import `sa_pq` from this
directory. This component does not use pandas or read input files.

## Interface

```
from sa_pq import Candidate, PaperSettings, ProtectionEvidence, Selector
selector = Selector('F1')
```

- `Selector(preference, adaptive=True)` creates an isolated controller.
  `preference` is `F1`, `F2` or `F3`; `adaptive=False` gives FIXED. The
  published selection settings are already supplied by `PaperSettings`. The CV
  admission tolerance defaults to the author's confirmed uniform value, so the
  default setup requires **no individual numerical settings**; an explicit
  `cv_tolerance` override remains available.
- `Candidate(chromosome, objectives, violation)` describes an already evaluated
  candidate. The chromosome is an integer encoding. Objectives use the internal
  minimization convention `[F1,-F2,-F3]`. Original-unit values must be in t·h,
  Chinese yuan (CNY) and score points, consistent with the appendix thresholds;
  do not use the scaled units of the result tables.
- `selector.quotas(count)` returns the largest-remainder quotas for the base
  ratio. It does not advance monitoring or activate dynamic allocation.
- Call `selector.advance(critical_front, keep, generation, monitor_archive, ...)`
  **once per consecutive generation**, starting at zero or one. Pass an empty
  front and `keep=0` when there is no truncation event. Update the monitoring
  archive before this call. The archive must already be historical,
  constraint-admissible and objective-nondominated; this component **never
  inserts it back into the population**.
- Selection uses the previously determined target states. The current
  generation's quota opportunities are then counted and the monitoring block
  is closed; updated target states affect subsequent selection.
- Returned objects belong to the input front; crowding fields are updated in
  memory. Inputs, the archive, controller state and diagnostic dictionaries
  remain in memory; nothing is exported or sent over a network.

## Reference values (fixed conventions)

- **Quality baseline.** The improvement percentages of the quality conditions
  are measured against the monitoring-archive detection value at the
  preference activation generation (default Gen300). No input is required:
  `advance` derives it from the supplied archive. If the D-generation
  detection value is unavailable (fewer than the minimum representatives), the
  target never enters saturation and the baseline is not moved to a later
  generation. An explicit `quality_baseline=(B1_ref,B2_ref,B3_ref)` (original
  objective directions) must be supplied no later than the activation
  boundary; once the baseline is established, only an identical value is
  accepted and any other value raises an error.
- **Protection admission.** When a saturated target first qualifies for dynamic
  allocation, the reference values are the current monitoring-archive
  detection values and the measured values are the best values of the current
  critical-front candidate set. No input is required; an explicit
  `protection=ProtectionEvidence(measured, reference)` overrides this default.

For saturated targets, the guard comparisons are

```
F1_measured <= F1_reference + max(1000, 0.02 × F1_reference)
F2_measured >= F2_reference - max(10000, 0.01 × F2_reference)
F3_measured >= F3_reference - 0.20
```

The remaining-budget requirement is checked at the moment dynamic allocation
actually first activates, together with the protection check (Section 4.4.2).
A failed protection check retains base ratios and **does not lock the budget
decision**; a later eligible event re-checks protection and the remaining
generations. Once activated, the decision stays allowed and dynamic ratios
follow the current target states, including the minimum saturation holding
period.

## Reading the code

- `sa_pq/config.py`: published numeric settings and the confirmed CV admission tolerance
- `sa_pq/api.py`: input validation and protection inequalities
- `sa_pq/core.py`: quota allocation, local selection and state monitoring
