# Pure contract logic isolated from optimization/contratos.py on 2026-09-21.
"""Versioned PV/WT PPAs; operator coefficients and a separate producer ledger.

Only new PV/WT are affected. Existing generators, BIO, BESS and flexibility
retain their original inputs. Money is in constant study units, not a quote.
"""
import copy
import hashlib
import json
import math

DEFAULT = dict(version=1, energy_per_MWh=45., curtailment_per_MWh=45.,
               duration_years=30, renewal='none', currency='constant_real_model_units',
               horizon_policy='payments_and_service_inside_horizon_only',
               participation='assumed_available_offer_not_profit_certificate')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name}: finite number >= {minimum} required')
    return float(value)


def policy(overrides=None):
    result = dict(DEFAULT)
    if overrides:
        if set(overrides) - set(result):
            raise ValueError('Unknown contract policy fields')
        result.update(overrides)
    for key in ('energy_per_MWh', 'curtailment_per_MWh'):
        number(result[key], key)
    years = number(result['duration_years'], 'duration_years', 1)
    if years != int(years):
        raise ValueError('Duration must be an integer number of years')
    result['duration_years'] = int(years)
    for key in ('version', 'renewal', 'currency', 'horizon_policy', 'participation'):
        if result[key] != DEFAULT[key]:
            raise ValueError(f'Unsupported {key}: {result[key]}')
    return result


def prepare(data, overrides=None):
    """Return a deep copy with executable operator prices and retained asset costs."""
    d = copy.deepcopy(data)
    rules = policy(overrides)
    spec = d['extension']
    for g in spec.get('generators', []):
        if g['technology'] not in ('PV', 'WT'):
            continue
        costs = copy.deepcopy(g.get('producer_costs', {key: g[key] for key in
                    ('capex_per_MW', 'fixed_per_MW_year', 'variable_per_MWh')}))
        g.update(owner='third_party', producer_id=f'new_{g["technology"]}_{g["id"]}',
                 producer_costs=costs, contract=dict(rules), capex_per_MW=0., fixed_per_MW_year=0.,
                 variable_per_MWh=rules['energy_per_MWh'], curtail_per_MWh=rules['curtailment_per_MWh'])
    spec['contracts_policy'] = rules
    spec.setdefault('original_asset_assumptions', spec.get('other_assumptions', ''))
    spec['ownership'] = 'New PV/WT third-party PPAs; BIO/operator and all existing resources unchanged'
    spec['other_assumptions'] = ('PV/WT operator payments follow contracts; producer CAPEX/O&M retained separately. '
        'Service availability and replacement responsibility are assumptions, not manufacturer or profit guarantees.')
    spec['policy_sha256'] = fingerprint(rules)
    validate(d)
    return d


def validate(data):
    c = data['config']; ys = c['investment_years']; end = c['horizon_years']
    if not ys or ys[0] != 0 or ys != sorted(set(ys)) or any(type(y) is not int or y < 0 for y in ys):
        raise ValueError('Investment years must be distinct increasing integers starting at zero')
    if type(end) is not int or end <= ys[-1]:
        raise ValueError('Invalid horizon')
    number(c['discount_rate'], 'discount_rate')
    number(c['annual_hours'], 'annual_hours', 1e-12)
    probabilities = [number(s['probability'], 'probability') for s in data['scenarios']]
    if abs(sum(probabilities)-1.) > 1e-10:
        raise ValueError('Scenario probabilities must sum to one')
    if not 0 <= number(c['beta'], 'beta') <= 1 or not 0 < number(c['alpha'], 'alpha') < 1:
        raise ValueError('Invalid beta or alpha')
    gens = data.get('extension', {}).get('generators', [])
    if len({g['id'] for g in gens}) != len(gens):
        raise ValueError('Duplicate new generator ID')
    scenario_ids = {str(s['id']) for s in data['scenarios']}
    if len(scenario_ids) != len(data['scenarios']):
        raise ValueError('Duplicate scenario ID')
    producer_ids = [g.get('producer_id') for g in gens if 'contract' in g]
    if len(set(producer_ids)) != len(producer_ids) or any(not p for p in producer_ids):
        raise ValueError('Missing or duplicate producer ID')
    for g in gens:
        if g['bus'] not in data['buses']:
            raise ValueError('Unknown generator bus')
        for key in ('max_MW', 'capex_per_MW', 'fixed_per_MW_year', 'variable_per_MWh', 'curtail_per_MWh', 'emissions_ton_per_MWh'):
            number(g[key], key)
        if set(g['availability']) != scenario_ids:
            raise ValueError('Incomplete availability')
        for value in g['availability'].values():
            if number(value, 'availability') > 1:
                raise ValueError('Availability fraction must be between zero and one')
        if 'contract' not in g:
            if g.get('owner') == 'third_party':
                raise ValueError('Third-party generator without contract')
            continue
        terms = policy(g['contract'])
        if g['technology'] not in ('PV', 'WT') or g.get('owner') != 'third_party':
            raise ValueError('Only new PV/WT PPAs supported')
        expected = (0., 0., terms['energy_per_MWh'], terms['curtailment_per_MWh'])
        actual = tuple(g[k] for k in ('capex_per_MW', 'fixed_per_MW_year', 'variable_per_MWh', 'curtail_per_MWh'))
        if actual != expected:
            raise ValueError('Contract prices and operator coefficients disagree')
        for key in ('capex_per_MW', 'fixed_per_MW_year', 'variable_per_MWh'):
            number(g['producer_costs'][key], 'producer_' + key)
        # A single dispatch represents each stage: do not average a mid-stage expiry.
        for start in ys:
            expiry = start + terms['duration_years']
            if expiry < end and expiry not in ys:
                raise ValueError('Expiry inside a stage requires a finer time grid')


def active(g, start, stage):
    return start <= stage and ('contract' not in g or stage < start + g['contract']['duration_years'])


def capacity(g, years, values, stage):
    return sum(values[f'dg_buy[{g["id"]},{j}]'] for j, start in enumerate(years)
               if active(g, start, years[stage]))


def generator_prices(data, g, new=True, emission=None):
    """Same dispatch coefficients for the MILP, candidate evaluator and AC."""
    energy = g['variable_per_MWh'] if new else g['price']
    curt = g['curtail_per_MWh'] if new else g['curtail_price']
    carbon = data['config']['carbon_price_per_ton'] * (g['emissions_ton_per_MWh'] if new else emission)
    return dict(energy=energy, carbon=carbon, dispatch=energy + carbon, curtailment=curt)


def payments(data, values, dispatch=None):
    """Per-state transfers and producer partial cash ledger, independent of Pyomo.

    dispatch[(g,k,s)] optionally supplies (used_MW,curtailed_MW), e.g. AC.
    Cohort attribution is proportional to active MW, with identical terms per g.
    """
    validate(data)
    c = data['config']; ys = c['investment_years']; end = c['horizon_years']; rate = c['discount_rate']
    weights = [sum((1+rate)**-y for y in range(start+1, (ys[k+1] if k+1 < len(ys) else end)+1))
               for k, start in enumerate(ys)]
    probs = {s['id']: s['probability'] for s in data['scenarios']}
    by_state = {s: 0. for s in probs}; detail = []; producers = []; cohorts = []
    for g in data['extension'].get('generators', []):
        if 'contract' not in g:
            continue
        costs = g['producer_costs']; received = capex = fixed = variable = 0.
        for k, year in enumerate(ys):
            buy = values[f'dg_buy[{g["id"]},{k}]']
            if not math.isfinite(buy) or buy < -1e-7:
                raise ValueError('Invalid contracted capacity')
            buy = max(0., buy)
            capex += buy * costs['capex_per_MW'] * (1+rate)**-year
            expiry = year + g['contract']['duration_years']
            if buy > 1e-7:
                cohorts.append(dict(producer=g['producer_id'], generator=g['id'], start=year, expiry=expiry,
                    MW=buy, years_inside=min(expiry,end)-year, years_beyond=max(0,expiry-end),
                    amount_beyond_horizon=None, allocation='proportional_active_capacity', renewal='none'))
            on = capacity(g, ys, values, k)
            fixed += on * costs['fixed_per_MW_year'] * weights[k]
            for s, prob in probs.items():
                used, curtailed = (dispatch[g['id'],k,s] if dispatch is not None else
                    (values[f'dg_gen[{g["id"]},{k},{s},0]'], values[f'dg_curt[{g["id"]},{k},{s},0]']))
                if not all(math.isfinite(v) and v >= -1e-6 for v in (used,curtailed)):
                    raise ValueError('Invalid metered energy')
                available = on*g['availability'][str(s)]
                if abs(used + curtailed - available) > 1e-5:
                    raise ValueError('Delivered plus curtailed differs from eligible availability')
                used, curtailed = max(0.,used), max(0.,curtailed)
                hours = c['annual_hours'] * prob
                paid_energy = used * g['contract']['energy_per_MWh'] * hours
                paid_curt = curtailed * g['contract']['curtailment_per_MWh'] * hours
                conditional = c['annual_hours'] * (used*g['contract']['energy_per_MWh'] + curtailed*g['contract']['curtailment_per_MWh'])
                by_state[s] += conditional * weights[k]
                received += (paid_energy + paid_curt) * weights[k]
                variable += used * costs['variable_per_MWh'] * hours * weights[k]
                detail.append(dict(producer=g['producer_id'], generator=g['id'], technology=g['technology'], bus=g['bus'],
                    year=year, state=s, active_MW=on, delivered_MWh_year=used*hours, curtailed_MWh_year=curtailed*hours,
                    energy_payment_year=paid_energy, curtailment_payment_year=paid_curt,
                    operator_payment_VP=(paid_energy+paid_curt)*weights[k],
                    producer_receipt_VP=(paid_energy+paid_curt)*weights[k]))
        producers.append(dict(producer=g['producer_id'], receipt_VP=received, capex_VP=capex,
            fixed_cost_VP=fixed, variable_cost_VP=variable,
            partial_modelled_balance_VP=received-capex-fixed-variable,
            scope='horizon_only; no financing/tax/replacement schedule/terminal value; not full-project profit'))
    total = sum(probs[s]*by_state[s] for s in probs)
    if abs(total - sum(p['receipt_VP'] for p in producers)) > 1e-4:
        raise ValueError('Transfer reconciliation failed')
    return dict(expected_payments_VP=total, conditional_payments_VP=by_state,
                producers=producers, details=detail, cohorts=cohorts)
