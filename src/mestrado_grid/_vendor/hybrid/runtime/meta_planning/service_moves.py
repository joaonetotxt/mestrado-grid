"""Common, explicit directed proposals for service restoration.

These are search operators, NOT feasibility repairs or an economic evaluator.
One engine displacement determines the amplitude of one complete candidate.
No prices, objective calls, power-flow trials or historical dispatches are used.
Ordinary offspring remain available in runner.py, retaining the full domain.
"""
import copy
import numpy as np
from .resource_construction import resources, allocate, sub_limits
from .temporal import storage_dispatch, demand_bounds


def _balance(st, candidate, reference, affected):
    """One P/Q headroom allocation including SUB as a *target*, never a gene."""
    lo, hi, qlo, qhi, _, fixed = resources(st, candidate)
    _, _, _, _, _, rf = resources(st, reference)
    qr = np.asarray(st.electrical['q_ratio'])
    residual_p = residual_q = 0.
    for k, s in affected:
        op = candidate.operation; ref = reference.operation
        served = op.conventional[k, :, s] - op.cut[k, :, s]
        rs = ref.conventional[k, :, s] - ref.cut[k, :, s]
        pl, ph, ql, qh, sm = sub_limits(st, k, s)
        # Small loss allowance is a proposal rule, not a relaxed physical limit.
        ql_eff = max(ql, -.98*sm); qh_eff = min(qh, .98*sm)
        if qh_eff >= 0: qh_eff *= .98
        target_q = float(np.dot(served, qr[:, s]))
        ref_q = float(np.dot(rs, qr[:, s]) - ref.existing_q[k, s].sum())
        q = allocate(np.r_[op.existing_q[k, s], ref_q],
                     np.r_[qlo[k, s], ql_eff], np.r_[qhi[k, s], qh_eff], target_q)
        sub_upper = max(pl, min(ph, np.sqrt(max(0., (.98*sm)**2-q[-1]**2))))
        ref_p = float((rs+rf[k, :, s]).sum() - ref.existing_p[k, s].sum() - ref.new_p[k, s].sum())
        target_p = float((served+fixed[k, :, s]).sum())
        p = allocate(np.r_[op.existing_p[k, s], op.new_p[k, s], ref_p],
                     np.r_[lo[k, s], pl], np.r_[hi[k, s], sub_upper], target_p)
        op.existing_p[k, s] = p[:st.G]; op.new_p[k, s] = p[st.G:-1]
        op.existing_q[k, s] = q[:-1]
        residual_p = max(residual_p, abs(float(p.sum()-target_p)))
        residual_q = max(residual_q, abs(float(q.sum()-target_q)))
    return residual_p, residual_q


def _donor(st, candidate, k, s):
    """Other block of the same day with greatest aggregate supply headroom."""
    cy = next(cy for cy in st.cycles if s in cy)
    _, hi, _, _, _, fixed = resources(st, candidate)
    # Demand BEFORE shedding avoids choosing a heavily shed block as a free
    # recharge source. Other-block AC capacity is not certified by this score.
    def spare(t):
        _, ph, _, _, sm = sub_limits(st, k, t)
        return float(hi[k, t].sum()+min(ph, sm)-
                     (candidate.operation.conventional[k, :, t]+fixed[k, :, t]).sum())
    others = [t for t in cy if t != s]
    return cy, max(others, key=lambda t: (spare(t), -t)) if others else None


def _storage_pulse(st, candidate, k, target, donor, cy, amplitude):
    """Recharge at donor, hold added SOC, discharge at target, close the day."""
    op = candidate.operation
    before_ch, before_di = storage_dispatch(st, candidate)
    td = cy.index(donor); ts = cy.index(target)
    arc = []; j = (td+1) % len(cy)
    while True:
        arc.append(cy[j])
        if j == ts: break
        j = (j+1) % len(cy)
    cap = np.array([u['energy_MWh'] for u in st.units])
    power = np.array([u['power_MW'] for u in st.units])
    ec = np.array([u['eta_charge'] for u in st.units])
    ed = np.array([u['eta_discharge'] for u in st.units])
    # Internal-energy bounds on the two modified transitions include any
    # charging/discharging already present in the reference cycle.
    donor_delta = op.soc[k, :, cy[(td+1) % len(cy)]] - op.soc[k, :, donor]
    target_delta = op.soc[k, :, cy[(ts+1) % len(cy)]] - op.soc[k, :, target]
    room = cap - np.max(op.soc[k][:, arc], axis=1)
    amount = np.minimum.reduce([amplitude*cap, room,
        power*ec*st.hours[donor]-donor_delta,
        target_delta+power*st.hours[target]/ed])
    amount = np.maximum(0., amount)*st.storage_active(candidate.plan)[k]
    for t in arc: op.soc[k, :, t] += amount
    after_ch, after_di = storage_dispatch(st, candidate)
    delivered = ((after_di-after_ch)-(before_di-before_ch))[k, :, target]
    buses = np.array([st.bi[u['bus']] for u in st.units], int)
    extra = np.bincount(buses, weights=np.maximum(0., delivered), minlength=st.B)
    op.cut[k, :, target] -= np.minimum(op.cut[k, :, target], extra)
    return float(amount.sum())


def service_proposal(codec, genome, anchor, ordinal, step, group=None):
    """Apply one declared move, with target priority from the current anchor.

    In joint modes odd offspring cycle restoration, storage, flexibility and
    capacity. In block mode the matching move follows the active block.
    """
    if type(ordinal) != int or ordinal < 0 or group not in (None, 0, 1, 2, 3):
        raise ValueError('Invalid service proposal schedule')
    if isinstance(step, bool) or not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError('Invalid service proposal step')
    st = codec.study; ref, repairs = codec.decode(anchor)
    raw = codec.unpack(genome); base = codec.unpack(anchor)
    candidate = copy.deepcopy(ref); op = candidate.operation
    kind = ('restore', 'storage', 'flex', 'capacity')[ordinal % 4] if group is None else ('capacity', 'restore', 'storage', 'flex')[group]
    round_id = ordinal // 4 if group is None else ordinal
    # Rank annual impact across stages. No monetary objective or hardcoded
    # state IDs enter the targeting rule. Zero-cut states are never preferred.
    scores = np.sum(ref.operation.cut, axis=(0, 1))*st.probabilities
    order = sorted(range(st.S), key=lambda s: (-scores[s], s))[:min(4, st.S)]
    s = order[round_id % len(order)]; k = round_id % st.K
    names = dict(restore=('cut',), storage=('soc',), flex=('conventional', 'ev'), capacity=('new_capacity',))[kind]
    amplitude = min(step, max((float(np.max(abs(raw[n]-base[n]), initial=0.)) for n in names), default=0.))
    affected = [(k, s)]; donor = None; added_energy = added_capacity = 0.
    if kind == 'restore':
        op.cut[k, :, s] -= np.minimum(op.cut[k, :, s], amplitude*op.conventional[k, :, s])
    elif kind in ('storage', 'flex'):
        cy, donor = _donor(st, candidate, k, s)
        if donor is not None:
            affected.append((k, donor))
            if kind == 'storage':
                added_energy = _storage_pulse(st, candidate, k, s, donor, cy, amplitude)
            else:
                lo, hi, elo, ehi = demand_bounds(st)
                for name, lower, upper in [('conventional', lo, hi), ('ev', elo, ehi)]:
                    a = getattr(op, name)
                    shifted = amplitude*np.maximum(0., np.minimum(a[k, :, s]-lower[k, :, s],
                        (upper[k, :, donor]-a[k, :, donor])*st.hours[donor]/st.hours[s]))
                    a[k, :, s] -= shifted
                    a[k, :, donor] += shifted*st.hours[s]/st.hours[donor]
                    if name == 'conventional': op.cut[k, :, s] -= np.minimum(op.cut[k, :, s], shifted)
    else:
        # A directed capacity proposal at sites whose technology is available
        # in the target state. Installed technology is not treated as firm when
        # its scenario availability is zero. Ordinary offspring can remove or
        # replace resources and modify trees; this move only proposes additions.
        buy_stage = 0 if st.data['investment_mode'] == 'single' else k
        cap = np.array([g['max_MW'] for g in st.new])
        av = np.array([g['availability'][str(s)] for g in st.new])
        addition = amplitude*np.maximum(0., cap-candidate.new_capacity.sum(axis=0))*av
        candidate.new_capacity[buy_stage] += addition
        added_capacity = float(addition.sum())
        op.cut[k, :, s] -= np.minimum(op.cut[k, :, s], amplitude*op.conventional[k, :, s])
    # An engine's exact zero displacement stays a no-op, rather than silently
    # replacing it with a fresh deterministic redispatch.
    changed = (np.max(abs(candidate.new_capacity-ref.new_capacity), initial=0.) > 1e-12 or
               any(np.max(abs(getattr(op, name)-getattr(ref.operation, name)), initial=0.) > 1e-12
                   for name in ('cut', 'soc', 'conventional', 'ev')))
    if amplitude > 0 and changed:
        rp, rq = _balance(st, candidate, ref, affected)
    else: rp = rq = 0.
    encoded = codec.encode(candidate) if amplitude > 0 and changed else np.asarray(anchor, float).copy()
    return encoded, dict(rule='service_directed_proposal_v1', move=kind,
        ordinal=ordinal, stage=k, year=st.years[k], state=s+1,
        donor_state=None if donor is None else donor+1, amplitude=amplitude, effective_move=bool(changed),
        anchor_id=ref.candidate_id, added_SOC_pulse_MWh=added_energy,
        added_capacity_MW=added_capacity,
        restored_cut_MW=float(ref.operation.cut.sum()-op.cut.sum()),
        maximum_unmet_P_MW=rp, maximum_unmet_Q_MVAr=rq,
        changed_coordinates=int(np.count_nonzero(abs(encoded-genome)>1e-12)),
        AC_evaluations=0, objective_evaluations=0, decode_repairs=repairs,
        role='explicit_search_operator_not_feasibility_repair')
