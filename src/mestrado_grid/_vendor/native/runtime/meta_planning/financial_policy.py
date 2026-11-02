"""Explicit annual financial selection over the unchanged complete AC ledger."""
import numpy as np

POLICY = 'CASTRO_ALIGNED_ANNUAL_WEIGHTED_v1'
VP30 = 'CASTRO_ADAPTADO_VP_T4_v1'
OFFICIAL = VP30  # Historical identifier, retained for imports.
DEFAULT = POLICY
POLICIES = (POLICY, VP30)
ALIASES = {'castro-aligned': POLICY, 'vp30': VP30}


def resolve(config):
    """Resolve explicitly configured policy, otherwise the current AC default."""
    name = config.get('selection_financial_policy', DEFAULT)
    if not isinstance(name, str) or name not in (*POLICIES, *ALIASES):
        raise ValueError('Unsupported explicit AC financial selection policy')
    return ALIASES.get(name, name)


def configure(study, name=None):
    """Return a new study with its effective policy recorded; never edit input files."""
    import copy
    from .schema import Study
    data = copy.deepcopy(study.data)
    if name is not None:
        data['config']['selection_financial_policy'] = name
    data['config']['selection_financial_policy'] = resolve(data['config'])
    return Study(data, study.electrical)


def compact_metrics(metrics, keys=('objective_VP',)):
    """Keep selected fitness when compacting; preserve the named VP30 field."""
    if metrics is None:
        return None
    out = {key:metrics[key] for key in keys}
    for key in ('selection_objective', 'financial_policy'):
        if key in metrics:
            out[key] = metrics[key]
    return out


def apply(st, candidate, cases, vp):
    from .economics import weights, tail, deterministic
    policy = resolve(st.c)
    w = weights(st)
    q = w / w.sum()
    p = st.probabilities
    H = st.c['annual_hours']
    rows = {(r['stage'], r['state_id']): r for r in cases}
    if len(rows) != st.K * st.S or not all(r['converged'] for r in cases):
        raise ValueError('Complete converged AC portfolio required for pricing')
    original_fixed = deterministic(st, candidate)
    fixed_parts = {n:v for n,v in original_fixed.items() if not n.startswith('switching_')}
    fixed = sum(fixed_parts.values())
    costs = np.zeros(st.S)
    annual = np.zeros(st.K)
    operating_parts = {}
    for k in range(st.K):
        for s in range(st.S):
            for name, hourly in rows[k, s]['hourly_costs'].items():
                value = H * q[k] * hourly
                costs[s] += value
                annual[k] += H * p[s] * hourly
                operating_parts[name] = operating_parts.get(name, 0.) + p[s] * value
    # Switching is operation, although the older physical ledger stores its VP
    # beside investment costs. Preserve that benchmark; annualize this explicit
    # deterministic representative-day schedule in the new policy only.
    switch_vp = sum(v for n,v in original_fixed.items() if n.startswith('switching_'))
    if 'physical' in st.electrical:
        from .physical import installed
        ph=st.electrical['physical']
        for choice,control,kind in [('cap_choice','cap_state','capacitors'),('reg_choice','tap','regulators')]:
            per_stage=np.zeros(st.K)
            for k in range(st.K):
                for j,o in enumerate(installed(st,candidate,choice,k)):
                    if o is None: continue
                    z=np.asarray(candidate.physical[control])[k,:,j]
                    for cy in st.cycles:
                        per_stage[k]+=H*p[cy[0]]/st.hours[cy[0]]*sum(abs(z[cy]-np.roll(z[cy],1)))*ph[kind][o]['switch_cost']
            assert np.isclose(w@per_stage,original_fixed.get('switching_'+kind,0.),atol=1e-5)
            value=float(q@per_stage)
            operating_parts['switching_'+kind]=value
            costs+=value
            annual+=per_stage
    cv, tail_rows = tail(costs, p, st.c['alpha'])
    # The generic tail helper retains historical VP30 field names. These values
    # are annual in this view: give them their actual basis, without repricing.
    annual_tail = [dict(state_id=row['state_id'], probability=row['probability'],
                        tail_mass=row['tail_mass'], cost_annual=row['cost_VP'],
                        contribution_annual=row['contribution_VP']) for row in tail_rows]
    expected = float(p @ costs)
    beta = st.c['beta']
    objective = fixed + (1-beta)*expected + beta*cv
    residual = sum(operating_parts.values()) - expected
    if not np.isclose(expected, (vp['expected_operation_VP']+switch_vp)/sum(w), rtol=1e-12, atol=1e-5):
        raise ValueError('Independent AC annual/VP30 reconciliation failed')
    if abs(residual) > 1e-5:
        raise ValueError('Annual cost components fail reconciliation')
    result = dict(vp)
    result.update(selection_objective=objective, financial_policy=POLICY,
        finance=dict(policy=POLICY, temporal_basis='PC1 VP + normalized annual operation; explicit reconstruction convention',
            PC1=fixed, fixed_parts_VP=fixed_parts, operating_parts_annual=operating_parts,
            annual_operation_by_stage=annual.tolist(), stage_weights=q.tolist(),
            expected_operation=expected, expected_total=fixed+expected,
            CVaR_operation=cv, CVaR_total=fixed+cv, objective=objective,
            conditional_costs=costs.tolist(), tail=annual_tail, tail_cost_basis='annual', accounting_residual=residual,
            beta=beta, alpha=st.c['alpha'], official_VP30_unchanged=True,
            switching_policy='annual expected cost of fixed representative-day switching schedule; deterministic across cost scenarios',
            switching_reclassified_from_legacy_fixed_VP=switch_vp))
    from .economic_policy import equivalent_annual_cost
    benchmark = dict(policy=VP30, temporal_basis='Complete discounted 30-year horizon',
        PC1=vp['deterministic_VP'], expected_operation=vp['expected_operation_VP'],
        expected_total=vp['expected_VP'], CVaR_operation=vp['CVaR_operation_VP'],
        CVaR_total=vp['CVaR_total_VP'], objective=vp['objective_VP'],
        parts_VP=dict(vp['parts_VP']), conditional_costs=list(vp['conditional_costs_VP']),
        tail=vp['tail'], tail_cost_basis='VP', accounting_residual=vp['accounting_residual_VP'],
        beta=beta, alpha=st.c['alpha'])
    result['financial_views'] = {'castro_aligned': result['finance'], 'vp30': benchmark}
    result['financial_views_schema_version'] = 2
    result['EAC'] = equivalent_annual_cost(vp['deterministic_VP'], vp['expected_operation_VP'],
        vp['CVaR_operation_VP'], beta, st.c['discount_rate'], st.c['horizon_years'])
    result['financial_policy'] = policy
    if policy == VP30:
        result['selection_objective'] = vp['objective_VP']
        result['finance'] = benchmark
    return result
