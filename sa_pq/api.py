"""In-memory interface to the paper's critical-front selection component."""
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

from .config import PaperSettings, PREFERENCE_RATIOS


def _triple(values):
    values = tuple(float(x) for x in values)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError('Three finite values are required.')
    return values


class Candidate:
    """An already evaluated candidate. Internal objectives are [F1,-F2,-F3]."""
    def __init__(self, chromosome, objectives, violation):
        self.chromosome = tuple(chromosome)
        if not self.chromosome or not all(isinstance(x, int) for x in self.chromosome):
            raise ValueError('A nonempty integer encoding is required.')
        self.objs = _triple(objectives)
        self.cv = float(violation)
        if not math.isfinite(self.cv) or self.cv < 0:
            raise ValueError('Constraint violation must be finite and nonnegative.')
        self.distance = self.cd = 0.0


@dataclass(frozen=True)
class ProtectionEvidence:
    """Optional explicit measured and reference values for Section 4.4.2.

    Both triples are [F1,F2,F3], not [F1,-F2,-F3]. When this evidence is not
    supplied, the package applies its fixed default: the reference is the
    current monitoring-archive detection value and the measured value is the
    best value of the critical-front candidate set, using the Table B-1 guard
    widths. Supplying this class overrides that default.
    """
    measured: tuple
    reference: tuple

    def __post_init__(self):
        object.__setattr__(self, 'measured', _triple(self.measured))
        object.__setattr__(self, 'reference', _triple(self.reference))
        if self.reference[0] < 0 or self.reference[1] < 0:
            raise ValueError('Relative guard references must be nonnegative.')

    def passes(self, saturated, settings):
        """Appendix B-1 guard inequalities, checked for saturated targets."""
        r1, r2, r3 = self.reference
        f1, f2, f3 = self.measured
        eps = settings.numerical_epsilon
        checks = (
            f1 <= r1 + max(settings.f1_guard_absolute, settings.f1_guard_relative*r1) + eps,
            f2 >= r2 - max(settings.f2_guard_absolute, settings.f2_guard_relative*r2) - eps,
            f3 >= r3 - settings.f3_guard_absolute - eps,
        )
        return all(checks[i] for i in range(3) if saturated[i])


class _Discard:
    def append(self, value):
        pass
    def extend(self, values):
        pass


class Selector:
    """One controller per run; the caller owns the NSGA-II population and archive.

    Call advance once every generation, even when no front needs truncation.
    Call with an empty front and keep=0 for such generations. Ranking,
    evaluation, initial populations and genetic operators are outside this
    critical-front package. No file or network output is performed.
    """
    def __init__(self, preference, *, cv_tolerance=None, adaptive=True, settings=None):
        if preference not in PREFERENCE_RATIOS:
            raise ValueError('Preference must be F1, F2 or F3.')
        self.settings = settings or PaperSettings()
        cv_tolerance = float(self.settings.cv_tolerance if cv_tolerance is None else cv_tolerance)
        if not math.isfinite(cv_tolerance) or cv_tolerance < 0:
            raise ValueError('CV tolerance must be finite and nonnegative.')
        if not isinstance(adaptive, bool):
            raise ValueError('Adaptive must be Boolean.')
        settings_map = self.settings.core_settings(preference, cv_tolerance, adaptive)
        for name, value in settings_map.items():
            if isinstance(value, (int,float)) and not math.isfinite(value):
                raise ValueError('Algorithm settings must be finite.')
        for value in (self.settings.observation_window, self.settings.stagnant_blocks,
                      self.settings.representative_count, self.settings.minimum_representatives,
                      self.settings.candidate_pool_multiplier):
            if not isinstance(value,int) or value <= 0:
                raise ValueError('Window sizes and counts must be positive integers.')
        if self.settings.minimum_representatives > self.settings.representative_count:
            raise ValueError('Minimum representative count exceeds requested count.')
        if not 0 <= self.settings.preference_start <= self.settings.maximum_generations:
            raise ValueError('Preference activation must fall within the generation budget.')
        if len(self.settings.grid_steps) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.settings.grid_steps):
            raise ValueError('Grid steps must be positive and finite.')
        spec = importlib.util.spec_from_file_location('sa_pq._instance',Path(__file__).with_name('core.py'))
        self.core = importlib.util.module_from_spec(spec)
        self.core.__dict__.update(settings_map)
        spec.loader.exec_module(self.core)
        self.state = self.core.init_sa_state()
        self.generation = None
        self.adaptive = adaptive
        self._sink = _Discard()

    def quotas(self, count):
        if not isinstance(count,int) or count < 0:
            raise ValueError('Quota count must be a nonnegative integer.')
        return self.core.allocate_quotas(count)

    def advance(self, critical_front, keep, generation, monitor_archive, *,
                quality_baseline=None, protection=None):
        """Algorithm 1 selection/update ordering and Algorithm 2 selection.

        The supplied archive must already be historical, constraint-admissible
        and objective-nondominated. Its chromosomes are deduplicated in the
        monitoring code. It is never inserted into the population.

        Quality baseline: defaults to the monitoring-archive detection value
        at the preference activation generation (Section 4.4.1). An explicit
        quality_baseline=[B1_ref,B2_ref,B3_ref] overrides that default and,
        if supplied, must be provided no later than the activation boundary.

        Protection admission (Section 4.4.2): defaults to comparing the
        critical-front candidate best values against the current archive
        detection values with the Table B-1 guard widths. An explicit
        ProtectionEvidence overrides that default. A failed check retains
        base ratios.
        """
        front, archive = list(critical_front), list(monitor_archive)
        if not isinstance(keep,int) or not 0 <= keep <= len(front):
            raise ValueError('Invalid critical-front selection size.')
        if not isinstance(generation,int) or not 0 <= generation <= self.settings.maximum_generations:
            raise ValueError('Generation is outside the configured budget.')
        if self.generation is None:
            if generation not in (0,1):
                raise ValueError('Start the controller at generation zero or one.')
        elif generation != self.generation + 1:
            raise ValueError('Advance once per consecutive generation.')
        if len({id(c) for c in front}) != len(front):
            raise ValueError('The front must contain distinct candidate objects.')
        if not all(isinstance(c,Candidate) for c in front+archive):
            raise ValueError('All inputs must be Candidate objects.')
        if quality_baseline is not None:
            baseline = _triple(quality_baseline)
            if baseline[0] <= 0 or baseline[1] <= 0:
                raise ValueError('Relative quality baselines must be positive.')
            st = self.state.get('rsat_v2') or {}
            if st.get('initialized'):
                # The baseline was already established (automatically or
                # explicitly) at the activation generation. Accept only an
                # identical value; any other value is rejected instead of
                # silently taking effect only in part.
                established = st.get('baseline_metrics')
                if established is None or established != list(baseline):
                    raise ValueError('Quality baseline was already established at the activation generation; it cannot be changed afterwards.')
            else:
                if self.core.QUALITY_BASELINE is not None and baseline != self.core.QUALITY_BASELINE:
                    raise ValueError('Quality baselines cannot change within a run.')
                self.core.QUALITY_BASELINE = baseline
        if protection is not None and not isinstance(protection,ProtectionEvidence):
            raise ValueError('Protection input must be ProtectionEvidence.')
        self.core.PROTECTION_CHECK = None if protection is None else lambda flags: protection.passes(flags,self.settings)
        self.core.update_sa_state(self.state,generation,archive)
        if keep == 0:
            selected = []
        elif keep == len(front):
            selected = list(front)
        else:
            selected,_ = self.core.environment_select_critical_front(
                front,keep,0,generation,self._sink,self._sink,self._sink,self.state,archive)
        # Monitoring closes after selection. Updated states are used next time.
        if self.adaptive:
            self.core.finalize_rsat_block(self.state,generation,archive,0,self._sink)
        if len(selected) != keep or len({id(c) for c in selected}) != keep:
            raise RuntimeError('Selection did not fill the requested distinct candidates.')
        self.generation = generation
        return selected
