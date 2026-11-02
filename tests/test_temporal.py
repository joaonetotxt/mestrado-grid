"""Regression checks of the packaged scientific reference."""
import copy
import numpy as np
import pytest
from meta_planning.catalogue import analytic_study, load_leiria
from meta_planning.codec import Codec
from meta_planning.initialization import initialize
from meta_planning.temporal import balance_profile, normalize, storage_dispatch, violations
from meta_planning.schema import Candidate, Study

@pytest.mark.parametrize('seed', [0, 1, 2, 19])
def test_complete_round_trip_and_idempotent_repairs(seed):
    st = analytic_study()
    co = Codec(st)
    a, x, _ = initialize(co, seed, 'random')
    assert max(violations(st, a).values()) < 1e-10
    b, _ = co.decode(x)
    for name in vars(a.operation):
        np.testing.assert_allclose(getattr(a.operation, name), getattr(b.operation, name), rtol=0, atol=1e-12)
    before = copy.deepcopy(b)
    normalize(st, b)
    for name in vars(a.operation):
        np.testing.assert_allclose(getattr(before.operation, name), getattr(b.operation, name), rtol=0, atol=1e-12)
    ch, dis = storage_dispatch(st, a)
    assert np.all(ch * dis == 0)
    assert np.sum(ch * st.hours) >= np.sum(dis * st.hours) - 1e-10

def test_soc_power_repair_and_inactive_canonicalization():
    st = analytic_study()
    co = Codec(st)
    a, x, _ = initialize(co)
    a.plan.storage_buys = [('ess1', 1)]
    a.operation.soc[:] = 0
    a.operation.soc[1:, 0, :] = [0, 0, 1, 1, 0, 0, 1, 1]
    assert violations(st, a)['BESS_power'] > 0
    normalize(st, a)
    assert max(violations(st, a).values()) < 1e-10
    blocks = co.unpack(co.encode(a))
    blocks['soc'][:, 1, :] = 0.9
    blocks['soc'][0, 0, :] = 0.8
    b, log = co.decode(co.pack(blocks))
    assert np.all(b.operation.soc[:, 1, :] == 0) and np.all(b.operation.soc[0] == 0)
    assert log['inactive_SOC_entries_zeroed'] > 0

def test_bounded_energy_identity_and_inconsistent_window():
    h = np.array([12.0, 5.0, 3.0, 4.0])
    base = np.array([0.1, 0.2, 0.5, 0.3])
    np.testing.assert_array_equal(balance_profile(base, h, 0.0, 1.0, base @ h), base)
    z = balance_profile(np.ones(4), h, 0.0, 1.0, base @ h)
    assert z @ h == pytest.approx(base @ h)
    with pytest.raises(ValueError):
        balance_profile(base, h, 0.0, 0.1, 20.0)
