"""Paper settings and the confirmed CV admission tolerance; no data records."""
from dataclasses import dataclass

PREFERENCE_RATIOS = {'F1': (2, 1, 1), 'F2': (1, 2, 1), 'F3': (1, 1, 2)}


@dataclass(frozen=True)
class PaperSettings:
    """Sections 4.3-4.4, Table 5-1 and Appendix Table B-1.

    Objectives use t*h, currency units and score points, not the scaled
    display units used in the result tables. Numerical epsilon is an
    implementation tolerance rather than a reported model parameter.
    """
    preference_start: int = 300
    maximum_generations: int = 1000
    # Confirmed uniform admission setting, separate from the appendix thresholds.
    cv_tolerance: float = 50000.0
    local_f1_relative: float = 0.02
    local_f2_relative: float = 0.02
    local_f3_absolute: float = 0.05
    candidate_pool_multiplier: int = 3
    grid_steps: tuple = (500.0, 10000.0, 0.03)
    representative_count: int = 10
    minimum_representatives: int = 5
    near_band_steps: int = 2
    observation_window: int = 80
    stagnant_blocks: int = 2
    minimum_quota_slots: int = 10
    minimum_quota_events: int = 3
    f1_quality_relative: float = 0.20
    f2_quality_relative: float = 0.04
    f3_quality_level: float = 91.80
    f1_guard_absolute: float = 1000.0
    f1_guard_relative: float = 0.02
    f2_guard_absolute: float = 10000.0
    f2_guard_relative: float = 0.01
    f3_guard_absolute: float = 0.20
    companion_expansion_cells: int = 2
    strong_change_steps: int = 2
    saturation_hold: int = 160
    minimum_remaining_generations: int = 80
    numerical_epsilon: float = 1e-12

    def core_settings(self, preference, cv_tolerance, adaptive):
        return {
            'PREFERENCE_RATIO': PREFERENCE_RATIOS[preference],
            'PREFERENCE_START_GEN': self.preference_start,
            'MAX_GEN': self.maximum_generations,
            'EPS': self.numerical_epsilon,
            'CV_TOL': cv_tolerance,
            'EPS_F1_REL': self.local_f1_relative,
            'EPS_F2_REL': self.local_f2_relative,
            'EPS_F3_ABS': self.local_f3_absolute,
            'CANDIDATE_POOL_MULTIPLIER': self.candidate_pool_multiplier,
            'RSAT_APPLY_DYNAMIC_QUOTA': adaptive,
            'RSAT_GRID_STEPS': self.grid_steps,
            'RSAT_REPRESENTATIVE_COUNT': self.representative_count,
            'RSAT_MIN_REPRESENTATIVES': self.minimum_representatives,
            'RSAT_NEAR_BAND_STEPS': self.near_band_steps,
            'RSAT_BLOCK_SIZE': self.observation_window,
            'RSAT_CONFIRM_STAGNANT_BLOCKS': self.stagnant_blocks,
            'RSAT_MIN_QUOTA_SLOTS': self.minimum_quota_slots,
            'RSAT_MIN_QUOTA_EVENTS': self.minimum_quota_events,
            'RSAT_F1_PROGRESS_REL': self.f1_quality_relative,
            'RSAT_F2_PROGRESS_REL': self.f2_quality_relative,
            'RSAT_F3_GOOD_LEVEL': self.f3_quality_level,
            'SA_F1_GUARD_ABS': self.f1_guard_absolute,
            'SA_F1_GUARD_REL': self.f1_guard_relative,
            'SA_F2_GUARD_ABS': self.f2_guard_absolute,
            'SA_F2_GUARD_REL': self.f2_guard_relative,
            'SA_F3_GUARD_ABS': self.f3_guard_absolute,
            'RSAT_MIN_COMPANION_EXPANSION_CELLS': self.companion_expansion_cells,
            'RSAT_STRONG_COMPANION_CELLS': self.companion_expansion_cells,
            'RSAT_STRONG_REACTIVATE_STEPS': self.strong_change_steps,
            'RSAT_MIN_SAT_HOLD_GENERATIONS': self.saturation_hold,
            'RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS': self.minimum_remaining_generations,
            'METHOD_NAME': 'SA-PQ' if adaptive else 'FIXED',
            'SAVE_QUOTA_SELECTED_DETAIL': False,
            'RSAT_SCHEMA_VERSION': 'paper_reference',
            # None means the fixed default: baseline = archive detection value at
            # the preference activation generation (Section 4.4.1).
            'QUALITY_BASELINE': None,
            # None means the fixed default: Section 4.4.2 protection admission
            # using archive detection values and critical-front best values.
            'PROTECTION_CHECK': None,
        }
