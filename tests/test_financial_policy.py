"""Regression checks of the packaged scientific reference."""
import copy
import json
import subprocess
import sys
import numpy as np
import pytest
from meta_planning.catalogue import analytic_study
from meta_planning.codec import Codec
from meta_planning.initialization import initialize
from meta_planning.evaluator import Evaluator, rank
from meta_planning.financial_policy import POLICY, VP30, configure, resolve, compact_metrics
from meta_planning.schema import Study
from meta_planning import economics
from meta_planning.engines import Engine, METHODS, SearchConfig
from meta_planning.runner import run_search, unseal

def fixture(policy=None, beta=0.5):
    st = analytic_study()
    d = copy.deepcopy(st.data)
    d['config']['beta'] = beta
    st = configure(Study(d, st.electrical), policy)
    c, _, _ = initialize(Codec(st), 11, 'constructive')
    result = Evaluator(st).evaluate(c)
    assert result['status'] == 'feasible'
    return (st, c, result)

@pytest.mark.parametrize('beta', [0.0, 0.5, 1.0])
def test_independent_annual_cost_and_fractional_cvar_with_preserved_vp30(beta):
    st, c, result = fixture(beta=beta)
    m = result['metrics']
    vp = economics.portfolio_vp30(st, c, result['cases'])
    assert {k: m[k] for k in vp} == vp
    w = np.array([sum(((1 + st.c['discount_rate']) ** (-t) for t in range(y + 1, (st.years[k + 1] if k + 1 < st.K else 30) + 1))) for k, y in enumerate(st.years)])
    annual = np.zeros(st.S)
    for row in result['cases']:
        annual[row['state_id']] += 8760 * w[row['stage']] / sum(w) * sum(row['hourly_costs'].values())
    expected = sum((p * x for p, x in zip(st.probabilities, annual)))
    remaining = 1 - st.c['alpha']
    tail_cost = 0.0
    for s in sorted(range(st.S), key=lambda s: annual[s], reverse=True):
        mass = min(remaining, st.probabilities[s])
        tail_cost += mass * annual[s]
        remaining -= mass
    cvar = tail_cost / (1 - st.c['alpha'])
    fixed = vp['deterministic_VP']
    assert m['financial_policy'] == POLICY
    assert m['selection_objective'] == pytest.approx(fixed + (1 - beta) * expected + beta * cvar, rel=1e-12)
    assert m['finance']['expected_operation'] == pytest.approx(expected, rel=1e-12)
    assert m['finance']['CVaR_operation'] == pytest.approx(cvar, rel=1e-12)
    assert sum(m['finance']['fixed_parts_VP'].values()) == pytest.approx(m['finance']['PC1'])
    assert sum(m['finance']['operating_parts_annual'].values()) == pytest.approx(expected)
    assert m['financial_views']['vp30']['objective'] == vp['objective_VP']
    assert m['EAC']['EAC_objective'] == pytest.approx(m['EAC']['CRF'] * vp['objective_VP'])
    other = Evaluator(configure(st, 'vp30')).evaluate(c)
    assert other['cases'] == result['cases']
    assert other['candidate_id'] == result['candidate_id']
    assert other['metrics']['financial_policy'] == VP30
    assert other['metrics']['selection_objective'] == vp['objective_VP']
    assert other['metrics']['financial_views'] == m['financial_views']
    assert other['metrics']['finance'] == m['financial_views']['vp30']

@pytest.mark.parametrize('method', METHODS)
def test_every_engine_selects_aligned_even_when_vp30_orders_the_opposite(method):
    engine = Engine(SearchConfig(method, population=4, generations=0), np.zeros((4, 2)))
    for i in range(4):
        record = dict(status='feasible', normalized_violation=0.0, candidate_id=str(i), metrics=dict(objective_VP=100 + i, selection_objective=10 - i, financial_policy=POLICY))
        engine.tell(engine.ask(), record)
    assert engine.best['record']['candidate_id'] == '3'
    restored = Engine.from_dict(json.loads(json.dumps(engine.to_dict())))
    assert restored.best['record'] == engine.best['record']
    infeasible = dict(status='violated', normalized_violation=1.0, metrics={'selection_objective': -100.0, 'objective_VP': -100.0})
    assert rank(engine.best['record']) < rank(infeasible)
