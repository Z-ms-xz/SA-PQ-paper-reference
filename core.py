"""Reference implementation of manuscript Sections 4.3 and 4.4.

Configured by the public interface. No empirical data or archived experiment
configuration is embedded. This is not a historical experiment executable.
"""
import hashlib
import json
import math
import numpy as np

def chromosome_hash(chromosome):
    s = ','.join(map(str, chromosome))
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:12]

def calc_crowding_only(front):
    if not front:
        return
    for p in front:
        p.cd = 0.0
        p.distance = 0.0
    if len(front) <= 2:
        for p in front:
            p.cd = float('inf')
            p.distance = float('inf')
        return
    for m in range(3):
        front.sort(key=lambda x: x.objs[m])
        front[0].cd = float('inf')
        front[-1].cd = float('inf')
        obj_min = front[0].objs[m]
        obj_max = front[-1].objs[m]
        if abs(obj_max - obj_min) > EPS:
            for i in range(1, len(front) - 1):
                if front[i].cd != float('inf'):
                    front[i].cd += (front[i + 1].objs[m] - front[i - 1].objs[m]) / (obj_max - obj_min + EPS)
    for p in front:
        p.distance = p.cd

def allocate_quotas(remain_k, ratio=PREFERENCE_RATIO):
    ratio = np.array(ratio, dtype=float)
    raw = remain_k * ratio / ratio.sum()
    base = np.floor(raw).astype(int)
    left = int(remain_k - base.sum())
    order = sorted(range(len(ratio)), key=lambda i: (raw[i] - base[i], ratio[i]), reverse=True)
    for i in order[:left]:
        base[i] += 1
    return tuple((int(x) for x in base))

def _auto_protection_check(flags, front, monitor_archive):
    """Section 4.4.2 default protection admission (Table B-1 guard widths).

    Reference values are the current monitoring-archive detection values
    (top-10 grid-representative medians, Section 4.4.1). Measured values are
    the best values of the current critical-front candidate set in the
    original objective directions. A missing detection value fails the check
    for that saturated target.
    """
    refs = []
    for obj_idx in range(3):
        snap = _snapshot_for_target(monitor_archive or [], obj_idx)
        refs.append(snap['robust_metric'])
    measured = [None, None, None]
    if front:
        measured[0] = min(original_value(p, 0) for p in front)
        measured[1] = max(original_value(p, 1) for p in front)
        measured[2] = max(original_value(p, 2) for p in front)
    for obj_idx in range(3):
        if not flags[obj_idx]:
            continue
        ref = refs[obj_idx]
        val = measured[obj_idx]
        if ref is None or val is None:
            return False
        if obj_idx == 0:
            if val > ref + max(SA_F1_GUARD_ABS, SA_F1_GUARD_REL * ref) + EPS:
                return False
        elif obj_idx == 1:
            if val < ref - max(SA_F2_GUARD_ABS, SA_F2_GUARD_REL * ref) - EPS:
                return False
        else:
            if val < ref - SA_F3_GUARD_ABS - EPS:
                return False
    return True

def get_effective_sa_ratio(sa_state, gen=None, front=None, monitor_archive=None):
    """Section 4.4.2: resolve the policy before critical-front allocation.

    The remaining-budget requirement is checked at the moment dynamic
    allocation actually first activates (protection passed), not at the first
    quota event. A failed protection check therefore does not lock the budget
    decision; a later eligible event re-checks protection and re-checks the
    remaining generations. Once activated, the decision stays ALLOWED.
    """
    st = _ensure_rsat_state(sa_state)
    flags = [target['status'] == 'SATURATED' for target in st['targets']]
    if not RSAT_APPLY_DYNAMIC_QUOTA or not any(flags):
        return (tuple(PREFERENCE_RATIO), [False, False, False], 'base')
    gate = st.setdefault('budget_gate', {})
    if gate.get('decision') == 'BLOCKED_LATE':
        return (tuple(PREFERENCE_RATIO), [False, False, False], 'budget_blocked')
    if not st.get('protection_admitted', False):
        if PROTECTION_CHECK is None:
            passed = _auto_protection_check(flags, front, monitor_archive)
        else:
            passed = PROTECTION_CHECK(tuple(flags))
        if not isinstance(passed, bool):
            raise ValueError('The protection check must return a Boolean decision.')
        if not passed:
            return (tuple(PREFERENCE_RATIO), [False, False, False], 'protection_blocked')
        # Actual first activation: both Section 4.4.2 conditions must hold now.
        if gen is None:
            raise ValueError('Generation is required when dynamic allocation first activates.')
        remaining = MAX_GEN - gen
        if remaining < RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS:
            gate.update(decision='BLOCKED_LATE', decision_gen=gen, remaining_generations=remaining, minimum_followup_generations=RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS)
            return (tuple(PREFERENCE_RATIO), [False, False, False], 'budget_blocked')
        gate.update(decision='ALLOWED', decision_gen=gen, remaining_generations=remaining, minimum_followup_generations=RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS)
        st['protection_admitted'] = True
    return (tuple((1 if flag else 2 for flag in flags)), flags, 'dynamic')

def allocate_objective_quotas(total_k, objectives, effective_ratio):
    if total_k <= 0 or not objectives:
        return {}
    sub_ratio = np.array([effective_ratio[i] for i in objectives], dtype=float)
    raw = total_k * sub_ratio / sub_ratio.sum()
    base = np.floor(raw).astype(int)
    left = int(total_k - base.sum())
    order = sorted(range(len(objectives)), key=lambda i: (raw[i] - base[i], sub_ratio[i]), reverse=True)
    for i in order[:left]:
        base[i] += 1
    return {objectives[i]: int(base[i]) for i in range(len(objectives))}

def original_value(ind, obj_idx):
    if obj_idx == 0:
        return float(ind.objs[0])
    if obj_idx == 1:
        return float(-ind.objs[1])
    if obj_idx == 2:
        return float(-ind.objs[2])
    raise ValueError(obj_idx)

def _distance_desc_key(ind):
    d = getattr(ind, 'distance', getattr(ind, 'cd', 0.0))
    return float('-inf') if d == float('inf') else -float(d)

def stable_crowding_key(ind):
    return (_distance_desc_key(ind), ind.objs[0], ind.objs[1], ind.objs[2], chromosome_hash(ind.chromosome))

def stable_objective_key(ind, obj_idx):
    return (ind.objs[obj_idx], _distance_desc_key(ind), ind.objs[0], ind.objs[1], ind.objs[2], chromosome_hash(ind.chromosome))

def sort_by_objective(individuals, obj_idx):
    return sorted(individuals, key=lambda x: stable_objective_key(x, obj_idx))

def epsilon_good_count_and_set(pool, obj_idx):
    if not pool:
        return ([], None, None)
    if obj_idx == 0:
        best = min((original_value(p, 0) for p in pool))
        threshold = best * (1.0 + EPS_F1_REL)
        good = [p for p in pool if original_value(p, 0) <= threshold + EPS]
    elif obj_idx == 1:
        best = max((original_value(p, 1) for p in pool))
        threshold = best * (1.0 - EPS_F2_REL)
        good = [p for p in pool if original_value(p, 1) >= threshold - EPS]
    elif obj_idx == 2:
        best = max((original_value(p, 2) for p in pool))
        threshold = best - EPS_F3_ABS
        good = [p for p in pool if original_value(p, 2) >= threshold - EPS]
    else:
        raise ValueError(obj_idx)
    return (good, best, threshold)

def _empty_target_state(obj_idx):
    return {'objective': int(obj_idx), 'status': 'ACTIVE', 'stagnant_streak': 0, 'block_start_gen': PREFERENCE_START_GEN + 1, 'block_start_metric': None, 'baseline_source_gen': None, 'block_start_band': None, 'block_start_companion_cells': [], 'quota_slots': 0, 'quota_events': 0, 'first_watch_gen': None, 'first_saturated_gen': None, 'saturated_since_gen': None, 'last_transition_gen': None, 'reactivation_count': 0, 'completed_blocks': 0, 'valid_evidence_blocks': 0, 'invalid_block_count': 0, 'last_block': None}

def _new_rsat_state():
    return {'schema_version': RSAT_SCHEMA_VERSION, 'initialized': False, 'baseline_metrics': [None, None, None], 'targets': [_empty_target_state(i) for i in range(3)], 'block_index': 0, 'last_finalized_gen': None, 'budget_gate': {'decision': 'UNDECIDED', 'decision_gen': None, 'remaining_generations': None, 'minimum_followup_generations': int(RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS)}}

def init_sa_state():
    return {'gen': [], 'F1': [], 'F2': [], 'F3': [], 'rsat_v2': _new_rsat_state()}

def _ensure_rsat_state(sa_state):
    if sa_state is None:
        raise ValueError('sa_state must not be None for RSAT-V2')
    if not isinstance(sa_state.get('rsat_v2'), dict):
        sa_state['rsat_v2'] = _new_rsat_state()
    st = sa_state['rsat_v2']
    defaults = _new_rsat_state()
    for key, value in defaults.items():
        if key not in st:
            st[key] = value
    if not isinstance(st.get('targets'), list) or len(st['targets']) != 3:
        st['targets'] = [_empty_target_state(i) for i in range(3)]
    for i in range(3):
        default_t = _empty_target_state(i)
        if not isinstance(st['targets'][i], dict):
            st['targets'][i] = default_t
        else:
            for key, value in default_t.items():
                if key not in st['targets'][i]:
                    st['targets'][i][key] = value
    return st

def archive_best_values(archive):
    if not archive:
        return None
    return {'F1': min((original_value(p, 0) for p in archive)), 'F2': max((original_value(p, 1) for p in archive)), 'F3': max((original_value(p, 2) for p in archive))}

def _cell_index(value, step):
    return int(math.floor(float(value) / float(step) + 1e-12))

def _business_values(ind):
    return (original_value(ind, 0), original_value(ind, 1), original_value(ind, 2))

def _full_grid_representatives(archive):
    by_chrom = {}
    for ind in archive or []:
        if float(ind.cv) > float(CV_TOL) + EPS:
            continue
        h = chromosome_hash(ind.chromosome)
        vals = _business_values(ind)
        key = (vals[0], -vals[1], -vals[2], float(ind.cv), h)
        old = by_chrom.get(h)
        if old is None or key < old[0]:
            by_chrom[h] = (key, ind, vals)
    by_cell = {}
    for key, ind, vals in by_chrom.values():
        cell = tuple((_cell_index(vals[i], RSAT_GRID_STEPS[i]) for i in range(3)))
        old = by_cell.get(cell)
        if old is None or key < old[0]:
            by_cell[cell] = (key, ind, vals, cell)
    return [(item[1], item[2], item[3]) for item in by_cell.values()]

def _target_sort_key(item, obj_idx):
    ind, vals, cell = item
    h = chromosome_hash(ind.chromosome)
    if obj_idx == 0:
        return (vals[0], -vals[1], -vals[2], h)
    if obj_idx == 1:
        return (-vals[1], vals[0], -vals[2], h)
    return (-vals[2], vals[0], -vals[1], h)

def _robust_metric(reps, obj_idx):
    if len(reps) < RSAT_MIN_REPRESENTATIVES:
        return (None, 0)
    ordered = sorted(reps, key=lambda item: _target_sort_key(item, obj_idx))
    k = min(RSAT_REPRESENTATIVE_COUNT, len(ordered))
    values = [item[1][obj_idx] for item in ordered[:k]]
    return (float(np.median(np.asarray(values, dtype=float))), int(k))

def _band_threshold(metric, obj_idx):
    if metric is None:
        return None
    width = RSAT_NEAR_BAND_STEPS * RSAT_GRID_STEPS[obj_idx]
    return float(metric + width) if obj_idx == 0 else float(metric - width)

def _inside_target_band(vals, obj_idx, threshold):
    if threshold is None:
        return False
    if obj_idx == 0:
        return vals[0] <= threshold + EPS
    return vals[obj_idx] >= threshold - EPS

def _cell_dominates(a, b, objectives):
    no_worse = True
    strict = False
    for pos, obj_idx in enumerate(objectives):
        av, bv = (a[pos], b[pos])
        if obj_idx == 0:
            if av > bv:
                no_worse = False
                break
            if av < bv:
                strict = True
        else:
            if av < bv:
                no_worse = False
                break
            if av > bv:
                strict = True
    return bool(no_worse and strict)

def _companion_nondominated_cells(reps, target_idx, band_threshold):
    companions = [i for i in range(3) if i != target_idx]
    cells = set()
    for _ind, vals, full_cell in reps:
        if _inside_target_band(vals, target_idx, band_threshold):
            cells.add((full_cell[companions[0]], full_cell[companions[1]]))
    ordered = sorted(cells)
    nd = []
    for cell in ordered:
        if any((_cell_dominates(other, cell, companions) for other in ordered if other != cell)):
            continue
        nd.append(cell)
    return set(nd)

def _meaningful_new_companion_cells(start_cells, end_cells, target_idx):
    companions = [i for i in range(3) if i != target_idx]
    start = set((tuple(x) for x in start_cells))
    end = set((tuple(x) for x in end_cells))
    meaningful = set()
    for cell in end.difference(start):
        if any((_cell_dominates(old, cell, companions) or old == cell for old in start)):
            continue
        meaningful.add(cell)
    return meaningful

def _snapshot_for_target(archive, obj_idx, fixed_band=None):
    reps = _full_grid_representatives(archive)
    metric, k = _robust_metric(reps, obj_idx)
    band = _band_threshold(metric, obj_idx) if fixed_band is None else fixed_band
    companion = _companion_nondominated_cells(reps, obj_idx, band)
    return {'feasible_grid_count': len(reps), 'representative_count': k, 'robust_metric': metric, 'band_threshold': band, 'companion_cells': companion}

def initialize_region_state(sa_state, monitor_archive, gen=PREFERENCE_START_GEN):
    st = _ensure_rsat_state(sa_state)
    if st.get('initialized'):
        return st
    baselines = []
    for obj_idx in range(3):
        snap = _snapshot_for_target(monitor_archive, obj_idx)
        baselines.append(snap['robust_metric'])
        target = st['targets'][obj_idx]
        target['block_start_gen'] = int(gen) + 1
        target['block_start_metric'] = snap['robust_metric']
        target['baseline_source_gen'] = int(gen) if snap['robust_metric'] is not None else None
        target['block_start_band'] = snap['band_threshold']
        target['block_start_companion_cells'] = sorted(snap['companion_cells'])
        target['quota_slots'] = 0
        target['quota_events'] = 0
    # Quality baseline: the monitoring-archive detection value at the
    # preference activation generation (Section 4.4.1). An explicit
    # quality_baseline overrides this fixed default.
    st['baseline_metrics'] = list(QUALITY_BASELINE) if QUALITY_BASELINE is not None else baselines
    st['initialized'] = True
    st['block_index'] = 0
    return st

def update_sa_state(sa_state, gen, monitor_archive):
    vals = archive_best_values(monitor_archive)
    if vals is not None:
        if sa_state.get('gen') and sa_state['gen'][-1] == gen:
            sa_state['F1'][-1] = vals['F1']
            sa_state['F2'][-1] = vals['F2']
            sa_state['F3'][-1] = vals['F3']
        else:
            sa_state.setdefault('gen', []).append(gen)
            sa_state.setdefault('F1', []).append(vals['F1'])
            sa_state.setdefault('F2', []).append(vals['F2'])
            sa_state.setdefault('F3', []).append(vals['F3'])
    st = _ensure_rsat_state(sa_state)
    if int(gen) == int(PREFERENCE_START_GEN) and (not st.get('initialized')):
        initialize_region_state(sa_state, monitor_archive, gen)

def _quality_gate(obj_idx, baseline, current):
    if baseline is None or current is None:
        return False
    if obj_idx == 0:
        return current <= baseline * (1.0 - RSAT_F1_PROGRESS_REL) + EPS
    if obj_idx == 1:
        return current >= baseline * (1.0 + RSAT_F2_PROGRESS_REL) - EPS
    return current >= RSAT_F3_GOOD_LEVEL - EPS

def _objective_improvement(obj_idx, start, end):
    if start is None or end is None:
        return None
    return float(start - end) if obj_idx == 0 else float(end - start)

def record_rsat_quotas(sa_state, allocations):
    st = _ensure_rsat_state(sa_state)
    for obj_idx in range(3):
        q = int(allocations[obj_idx])
        target = st['targets'][obj_idx]
        target['quota_slots'] = int(target.get('quota_slots', 0)) + q
        if q > 0:
            target['quota_events'] = int(target.get('quota_events', 0)) + 1

def _start_next_block(target, monitor_archive, obj_idx, next_start_gen):
    snap = _snapshot_for_target(monitor_archive, obj_idx)
    target['block_start_gen'] = int(next_start_gen)
    target['block_start_metric'] = snap['robust_metric']
    target['block_start_band'] = snap['band_threshold']
    target['block_start_companion_cells'] = sorted(snap['companion_cells'])
    target['quota_slots'] = 0
    target['quota_events'] = 0

def finalize_rsat_block(sa_state, gen, monitor_archive, run_id, saturation_log_rows):
    if int(gen) <= int(PREFERENCE_START_GEN):
        return
    if (int(gen) - int(PREFERENCE_START_GEN)) % int(RSAT_BLOCK_SIZE) != 0:
        return
    st = _ensure_rsat_state(sa_state)
    if st.get('last_finalized_gen') == int(gen):
        return
    if not st.get('initialized'):
        initialize_region_state(sa_state, monitor_archive, PREFERENCE_START_GEN)
    st['block_index'] = int(st.get('block_index', 0)) + 1
    rows_to_append = []
    for obj_idx in range(3):
        target = st['targets'][obj_idx]
        status_before = str(target['status'])
        start_gen = int(target.get('block_start_gen', int(gen) - RSAT_BLOCK_SIZE + 1))
        start_metric = target.get('block_start_metric')
        fixed_band = target.get('block_start_band')
        start_cells = set((tuple(x) for x in target.get('block_start_companion_cells', [])))
        end_snap = _snapshot_for_target(monitor_archive, obj_idx, fixed_band=fixed_band)
        end_metric = end_snap['robust_metric']
        end_cells = set(end_snap['companion_cells'])
        new_cells = _meaningful_new_companion_cells(start_cells, end_cells, obj_idx)
        improvement = _objective_improvement(obj_idx, start_metric, end_metric)
        step = float(RSAT_GRID_STEPS[obj_idx])
        own_stagnant = improvement is not None and -step - EPS <= improvement < step - EPS
        material_worsening = improvement is not None and improvement < -step - EPS
        strong_improvement = improvement is not None and improvement >= RSAT_STRONG_REACTIVATE_STEPS * step - EPS
        strong_worsening = improvement is not None and improvement <= -RSAT_STRONG_REACTIVATE_STEPS * step + EPS
        companion_expanded = len(new_cells) >= RSAT_MIN_COMPANION_EXPANSION_CELLS
        strong_companion = len(new_cells) >= RSAT_STRONG_COMPANION_CELLS
        opportunity_ok = int(target.get('quota_slots', 0)) >= RSAT_MIN_QUOTA_SLOTS and int(target.get('quota_events', 0)) >= RSAT_MIN_QUOTA_EVENTS
        baseline = st['baseline_metrics'][obj_idx]
        # The quality baseline is fixed at the preference activation generation
        # D (Section 4.4.1). If the D-generation detection value is unavailable
        # (fewer than the minimum representatives), the target cannot satisfy
        # the quality condition and never enters saturation; the baseline is
        # not moved to a later generation.
        quality_ok = bool(baseline is not None and _quality_gate(obj_idx, baseline, end_metric))
        representatives_ok = start_metric is not None and end_metric is not None and (int(end_snap['representative_count']) >= RSAT_MIN_REPRESENTATIVES)
        valid_evidence_block = bool(representatives_ok and opportunity_ok)
        stagnant_candidate = bool(valid_evidence_block and own_stagnant and (not companion_expanded) and (not material_worsening))
        quality_lost_strong = False
        if baseline is not None and end_metric is not None:
            if obj_idx == 0:
                gate = baseline * (1.0 - RSAT_F1_PROGRESS_REL)
                quality_lost_strong = end_metric > gate + step + EPS
            elif obj_idx == 1:
                gate = baseline * (1.0 + RSAT_F2_PROGRESS_REL)
                quality_lost_strong = end_metric < gate - step - EPS
            else:
                quality_lost_strong = end_metric < RSAT_F3_GOOD_LEVEL - step - EPS
        recovery_signal = bool(strong_improvement or strong_worsening or strong_companion or not quality_ok)
        transition = ''
        decision = ''
        if not valid_evidence_block:
            decision = 'INSUFFICIENT_EVIDENCE_STATE_UNCHANGED'
            target['invalid_block_count'] = int(target.get('invalid_block_count', 0)) + 1
        elif status_before == 'SATURATED':
            decision = 'SATURATED_MONITOR'
            sat_since = int(target.get('saturated_since_gen') or gen)
            hold_ok = int(gen) - sat_since >= int(RSAT_MIN_SAT_HOLD_GENERATIONS)
            if hold_ok and recovery_signal:
                target['status'] = 'ACTIVE'
                target['stagnant_streak'] = 0
                target['saturated_since_gen'] = None
                target['reactivation_count'] = int(target.get('reactivation_count', 0)) + 1
                target['last_transition_gen'] = int(gen)
                transition = 'SATURATED_TO_ACTIVE_STRONG_RECOVERY'
        elif baseline is None:
            decision = 'INSUFFICIENT_BASELINE_AT_D'
            target['invalid_block_count'] = int(target.get('invalid_block_count', 0)) + 1
        elif not representatives_ok:
            decision = 'INVALID_INSUFFICIENT_REPRESENTATIVES'
            target['invalid_block_count'] = int(target.get('invalid_block_count', 0)) + 1
        elif not opportunity_ok:
            decision = 'INVALID_INSUFFICIENT_OPPORTUNITY'
            target['invalid_block_count'] = int(target.get('invalid_block_count', 0)) + 1
        elif stagnant_candidate:
            target['valid_evidence_blocks'] = int(target.get('valid_evidence_blocks', 0)) + 1
            target['stagnant_streak'] = int(target.get('stagnant_streak', 0)) + 1
            if status_before == 'ACTIVE':
                target['status'] = 'WATCH'
                if target.get('first_watch_gen') is None:
                    target['first_watch_gen'] = int(gen)
                target['last_transition_gen'] = int(gen)
                transition = 'ACTIVE_TO_WATCH'
            if int(target['stagnant_streak']) >= int(RSAT_CONFIRM_STAGNANT_BLOCKS):
                if quality_ok:
                    decision = 'VALID_STAGNATION_CONFIRMED_QUALITY_OK'
                    target['status'] = 'SATURATED'
                    target['saturated_since_gen'] = int(gen)
                    if target.get('first_saturated_gen') is None:
                        target['first_saturated_gen'] = int(gen)
                    target['last_transition_gen'] = int(gen)
                    transition = 'WATCH_TO_SATURATED'
                else:
                    decision = 'VALID_STAGNATION_LOW_QUALITY_NO_TRANSFER'
            else:
                decision = 'VALID_STAGNATION_FIRST'
        else:
            decision = 'VALID_PROGRESS_OR_REGION_EXPANSION'
            target['valid_evidence_blocks'] = int(target.get('valid_evidence_blocks', 0)) + 1
            if status_before == 'WATCH':
                target['status'] = 'ACTIVE'
                target['last_transition_gen'] = int(gen)
                transition = 'WATCH_TO_ACTIVE_PROGRESS'
            target['stagnant_streak'] = 0
        target['completed_blocks'] = int(target.get('completed_blocks', 0)) + 1
        block_row = {'method': METHOD_NAME, 'run_id': int(run_id), 'generation': int(gen), 'rule': 'RSAT_V2_block_evaluation', 'scheduled_boundary_index': int(st['block_index']), 'block_start_gen': start_gen, 'block_end_gen': int(gen), 'window_length_generations': int(gen) - start_gen + 1, 'window_closed': 1, 'decision': decision, 'objective': f'F{obj_idx + 1}', 'status_before': status_before, 'status_after': target['status'], 'transition': transition, 'baseline_metric': baseline, 'baseline_source_gen': target.get('baseline_source_gen'), 'start_robust_metric': start_metric, 'end_robust_metric': end_metric, 'objective_improvement': improvement, 'business_grid_step': step, 'quality_gate_ok': int(quality_ok), 'quality_lost_strong': int(quality_lost_strong), 'representatives_ok': int(representatives_ok), 'end_representative_count': int(end_snap['representative_count']), 'monitor_archive_size': len(monitor_archive or []), 'feasible_grid_count': int(end_snap['feasible_grid_count']), 'near_band_threshold': fixed_band, 'companion_nd_cells_start': len(start_cells), 'companion_nd_cells_end': len(end_cells), 'new_companion_nd_cells': len(new_cells), 'companion_expansion_threshold': int(RSAT_MIN_COMPANION_EXPANSION_CELLS), 'companion_cells_start_json': json.dumps(sorted(start_cells), ensure_ascii=False, separators=(',', ':')), 'companion_cells_end_json': json.dumps(sorted(end_cells), ensure_ascii=False, separators=(',', ':')), 'new_companion_cells_json': json.dumps(sorted(new_cells), ensure_ascii=False, separators=(',', ':')), 'companion_expanded': int(companion_expanded), 'quota_slots': int(target.get('quota_slots', 0)), 'quota_events': int(target.get('quota_events', 0)), 'opportunity_ok': int(opportunity_ok), 'valid_evidence_block': int(valid_evidence_block), 'own_stagnant': int(own_stagnant), 'material_worsening': int(material_worsening), 'stagnant_block': int(stagnant_candidate), 'stagnant_streak_after': int(target.get('stagnant_streak', 0)), 'strong_improvement': int(strong_improvement), 'strong_worsening': int(strong_worsening), 'strong_companion_expansion': int(strong_companion), 'recovery_signal': int(recovery_signal), 'reactivation_count': int(target.get('reactivation_count', 0)), 'saturated_since_gen': target.get('saturated_since_gen'), 'dynamic_quota_enabled': int(bool(RSAT_APPLY_DYNAMIC_QUOTA))}
        target['last_block'] = dict(block_row)
        rows_to_append.append(block_row)
        _start_next_block(target, monitor_archive, obj_idx, int(gen) + 1)
    st['last_finalized_gen'] = int(gen)
    saturation_log_rows.extend(rows_to_append)

def rsat_state_columns(sa_state):
    st = _ensure_rsat_state(sa_state)
    out = {'rsat_block_index': int(st.get('block_index', 0))}
    for i, name in enumerate(('F1', 'F2', 'F3')):
        target = st['targets'][i]
        out[f'state_{name}'] = target['status']
        out[f'sat_{name}'] = int(target['status'] == 'SATURATED')
        out[f'stagnant_streak_{name}'] = int(target.get('stagnant_streak', 0))
        out[f'first_watch_gen_{name}'] = target.get('first_watch_gen')
        out[f'first_saturated_gen_{name}'] = target.get('first_saturated_gen')
        out[f'saturated_since_gen_{name}'] = target.get('saturated_since_gen')
        out[f'reactivation_count_{name}'] = int(target.get('reactivation_count', 0))
    return out

def crowding_fallback(candidates, quota, used_ids):
    rest = [p for p in candidates if id(p) not in used_ids]
    rest.sort(key=stable_crowding_key)
    picked = []
    for cand in rest:
        if len(picked) >= quota:
            break
        used_ids.add(id(cand))
        picked.append((cand, 'crowding_fallback'))
    return picked

def select_recursive_epsilon(candidates, quota, target_obj, active_objs, used_ids, run_id, gen, top_target, saturation_log_rows, effective_ratio, global_sat_flags, path_prefix='', depth=0):
    picked = []
    if quota <= 0:
        return picked
    available = [p for p in candidates if id(p) not in used_ids]
    if not available:
        return picked
    quota = min(quota, len(available))
    sorted_target = sort_by_objective(available, target_obj)
    pool_size = min(len(sorted_target), max(quota, quota * CANDIDATE_POOL_MULTIPLIER))
    pool = sorted_target[:pool_size]
    good, best_val, threshold = epsilon_good_count_and_set(pool, target_obj)
    is_saturated = len(good) >= quota
    remaining_objs = [o for o in active_objs if o != target_obj and (not bool(global_sat_flags[o]))]
    saturation_log_rows.append({'method': METHOD_NAME, 'run_id': run_id, 'generation': gen, 'top_quota_block': f'F{top_target + 1}', 'current_target': f'F{target_obj + 1}', 'depth': depth, 'quota': quota, 'candidate_pool_size': pool_size, 'epsilon_good_count': len(good), 'is_saturated': int(is_saturated), 'best_value_in_pool': best_val, 'epsilon_threshold': threshold, 'active_objectives': ','.join((f'F{o + 1}' for o in active_objs)), 'remaining_objectives': ','.join((f'F{o + 1}' for o in remaining_objs)), 'effective_ratio': ':'.join(map(str, effective_ratio)), 'global_sat_flags': ':'.join(('1' if x else '0' for x in global_sat_flags)), 'rule': 'epsilon_saturation' if is_saturated else 'primary_objective_sort'})
    if not is_saturated:
        for cand in sorted_target:
            if len(picked) >= quota:
                break
            if id(cand) in used_ids:
                continue
            used_ids.add(id(cand))
            picked.append((cand, path_prefix + f'F{target_obj + 1}_primary'))
        return picked
    if not remaining_objs:
        tmp = crowding_fallback(good, quota, used_ids)
        return [(cand, path_prefix + 'all_active_targets_exhausted_crowding') for cand, _ in tmp]
    sub_quotas = allocate_objective_quotas(quota, remaining_objs, effective_ratio)
    for sub_obj, sub_q in sub_quotas.items():
        if sub_q <= 0:
            continue
        sub_picked = select_recursive_epsilon(good, sub_q, sub_obj, remaining_objs, used_ids, run_id, gen, top_target, saturation_log_rows, effective_ratio, global_sat_flags, path_prefix + f'F{target_obj + 1}_sat_to_', depth + 1)
        picked.extend(sub_picked)
    if len(picked) < quota:
        need = quota - len(picked)
        tmp = crowding_fallback(good, need, used_ids)
        picked.extend([(cand, path_prefix + f'F{target_obj + 1}_sat_good_crowding') for cand, _ in tmp])
    if len(picked) < quota:
        for cand in sorted_target:
            if len(picked) >= quota:
                break
            if id(cand) in used_ids:
                continue
            used_ids.add(id(cand))
            picked.append((cand, path_prefix + f'F{target_obj + 1}_sat_target_fallback'))
    return picked[:quota]

def baseline_crowding_select(front, remain_k):
    calc_crowding_only(front)
    return sorted(front, key=stable_crowding_key)[:remain_k]

def environment_select_critical_front(front, remain_k, run_id, gen, quota_log_rows, quota_detail_rows, saturation_log_rows, sa_state=None, monitor_archive=None):
    if gen <= PREFERENCE_START_GEN:
        return (baseline_crowding_select(front, remain_k), 'B')
    if not _ensure_rsat_state(sa_state).get('initialized'):
        initialize_region_state(sa_state, monitor_archive, PREFERENCE_START_GEN)
    selected = preference_quota_select(front, remain_k, run_id, gen, quota_log_rows, quota_detail_rows, saturation_log_rows, sa_state, monitor_archive)
    return (selected, 'SA')

def preference_quota_select(front, remain_k, run_id, gen, quota_log_rows, quota_detail_rows, saturation_log_rows, sa_state=None, monitor_archive=None):
    calc_crowding_only(front)
    effective_ratio, sat_flags, quota_mode = get_effective_sa_ratio(sa_state, gen, front, monitor_archive)
    quotas = allocate_quotas(remain_k, effective_ratio)
    sorted_lists = []
    rank_maps = []
    for m in range(3):
        lst = sort_by_objective(front, m)
        sorted_lists.append(lst)
        rank_maps.append({id(ind): idx + 1 for idx, ind in enumerate(lst)})
    selected = []
    selected_ids = set()
    selected_from = [0, 0, 0]
    fallback_count = 0
    active_objs = [o for o in range(3) if not sat_flags[o]]
    for m, quota in enumerate(quotas):
        if quota <= 0:
            continue
        if RSAT_APPLY_DYNAMIC_QUOTA and sat_flags[m]:
            picked = []
            for cand in sorted_lists[m]:
                if len(picked) >= quota:
                    break
                if id(cand) in selected_ids:
                    continue
                selected_ids.add(id(cand))
                picked.append((cand, f'F{m + 1}_global_sat_protection'))
        else:
            recursion_objs = active_objs if active_objs else [0, 1, 2]
            picked = select_recursive_epsilon(front, quota, m, recursion_objs, selected_ids, run_id, gen, m, saturation_log_rows, effective_ratio, sat_flags, path_prefix=f'F{m + 1}_quota_', depth=0)
        for cand, label in picked:
            selected.append(cand)
            selected_from[m] += 1
            if SAVE_QUOTA_SELECTED_DETAIL:
                quota_detail_rows.append({'method': METHOD_NAME, 'run_id': run_id, 'generation': gen, 'selected_order': len(selected), 'selected_by': label, 'top_quota_block': f'F{m + 1}', 'rank_in_F1': rank_maps[0].get(id(cand), None), 'rank_in_F2': rank_maps[1].get(id(cand), None), 'rank_in_F3': rank_maps[2].get(id(cand), None), 'F1': cand.objs[0], 'F2': -cand.objs[1], 'F3': -cand.objs[2], 'CV': cand.cv, 'individual_hash': chromosome_hash(cand.chromosome)})
    if len(selected) < remain_k:
        rest = [ind for ind in front if id(ind) not in selected_ids]
        rest.sort(key=lambda x: getattr(x, 'distance', 0.0), reverse=True)
        for cand in rest:
            if len(selected) >= remain_k:
                break
            selected.append(cand)
            selected_ids.add(id(cand))
            fallback_count += 1
    record_rsat_quotas(sa_state, selected_from)
    state_cols = rsat_state_columns(sa_state)
    gate = _ensure_rsat_state(sa_state).get('budget_gate', {})
    quota_row = {'method': METHOD_NAME, 'run_id': run_id, 'generation': gen, 'preference_ratio': ':'.join(map(str, PREFERENCE_RATIO)), 'effective_ratio': ':'.join(map(str, effective_ratio)), 'quota_mode': quota_mode, 'dynamic_quota_enabled': int(bool(RSAT_APPLY_DYNAMIC_QUOTA)), 'dynamic_ratio_active': int(tuple(effective_ratio) != tuple(PREFERENCE_RATIO)), 'budget_gate_decision': gate.get('decision', 'UNDECIDED'), 'budget_gate_decision_gen': gate.get('decision_gen'), 'budget_gate_remaining_generations': gate.get('remaining_generations'), 'budget_gate_minimum_followup_generations': gate.get('minimum_followup_generations', RSAT_MIN_DYNAMIC_FOLLOWUP_GENERATIONS), 'remain_k': remain_k, 'critical_front_size': len(front), 'quota_F1': quotas[0], 'quota_F2': quotas[1], 'quota_F3': quotas[2], 'selected_from_F1_block': selected_from[0], 'selected_from_F2_block': selected_from[1], 'selected_from_F3_block': selected_from[2], 'fallback_count': fallback_count, 'selected_total': len(selected)}
    quota_row.update(state_cols)
    quota_log_rows.append(quota_row)
    return selected[:remain_k]

def feasible_only(front):
    return [p for p in front if float(p.cv) <= float(CV_TOL) + EPS]
