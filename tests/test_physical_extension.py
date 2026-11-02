"""Regression checks of the packaged scientific reference."""
import copy
import math
import numpy as np
import pytest
from meta_planning.catalogue import analytic_study, load_leiria
from meta_planning.physical_catalogue import extend
from meta_planning import power_flow as pf, physical
from meta_planning.schema import Candidate, Study
from meta_planning.codec import Codec
from meta_planning.initialization import initialize
from meta_planning.evaluator import Evaluator, audit_saved, save_evaluation
from meta_planning.topology import PlanCodec, line_schedule
from meta_planning.engines import SearchConfig
from meta_planning.runner import run_search

@pytest.mark.parametrize('tap', [0.9, 1.0, 1.1])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('power', [-0.4, 0.4])
def test_tap_shunt_two_independent_solvers_and_terminal_balance(tap, reverse, power):
    e = dict(i=1 if reverse else 0, j=0 if reverse else 1, r=0.004, x=0.008, b=0.0002, tap=tap)
    sh = np.array([0.0, 0.0002 + 0.1j])
    S = np.array([0.0, -power - 0.15j])
    n = pf.newton(2, [e], S, shunts=sh)
    w = pf.sweep(2, [e], S, shunts=sh)
    assert n['converged'] and w['converged']
    assert max(abs(n['V'] - w['V'])) < 1e-08
    V = n['V']
    f = pf.branch_flows(V, [e])[0]
    assert abs(f['Sf'] - V[e['i']] * np.conj(f['If'])) < 1e-12
    assert abs(f['St'] - V[e['j']] * np.conj(f['It'])) < 1e-12
    loss = (f['Sf'] + f['St']).real
    series = (tap * V[e['i']] - V[e['j']]) / complex(e['r'], e['x'])
    assert loss == pytest.approx(e['r'] * abs(series) ** 2, abs=1e-12)
    calc = V * np.conj(pf.admittance(2, [e], sh) @ V)
    assert calc[0].real == pytest.approx(power + loss + sum(sh.real * abs(V) ** 2), abs=1e-08)

def setup(**kw):
    st = extend(analytic_study(), **kw)
    co = Codec(st)
    c, _, _ = initialize(co, 11, 'resource')
    return (st, co, c)

def choose(st, c, name, site, option=0, stage=0):
    c.physical[name][site] = physical.choice_options(st, name).index((option, stage))

@pytest.mark.parametrize('voltage,expected', [(0.95, 1.083), (1.0, 1.2), (1.05, 1.323)])
def test_capacitor_actual_voltage_squared(voltage, expected):
    st, co, c = setup()
    p = st.electrical['physical']
    o = next((i for i, r in enumerate(p['capacitors']) if r['q_MVAr'] == 1.2 and r['mode'] == 'switched'))
    choose(st, c, 'cap_choice', 0, o)
    c.physical['cap_state'][0][0][0] = 1
    from meta_planning.temporal import storage_dispatch
    ch, di = storage_dispatch(st, c)
    ix = {b: i for i, b in enumerate(st.buses)}
    dev = physical.case_devices(st, c, 0, 0, ix, [], ch, di, np.full(st.B, voltage, complex))
    assert dev['capacitors'][0]['Q_MVAr'] == pytest.approx(expected)
    assert dev['capacitors'][0]['loss_MW'] == pytest.approx(0.0005 * expected)
    c.physical['cap_state'][0][0][0] = 0
    assert physical.case_devices(st, c, 0, 0, ix, [], ch, di)['capacitors'][0]['Q_MVAr'] == 0

def test_no_free_capacitor_or_regulator_before_purchase():
    st, co, c = setup()
    choose(st, c, 'cap_choice', 0, 1, 1)
    choose(st, c, 'reg_choice', 0, 0, 1)
    c.physical['cap_state'][0][0][0] = 1
    c.physical['tap'][0][0][0] = 1
    assert physical.structural_violations(st, c)['physical_installation'] == 1
    assert Evaluator(st).evaluate(c)['status'] == 'violated'
    with pytest.raises(ValueError, match='physical controls'):
        co.encode(c)
