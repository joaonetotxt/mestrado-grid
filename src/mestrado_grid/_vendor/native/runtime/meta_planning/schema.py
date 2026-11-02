"""Versioned physical decisions. Array axes are explicit and JSON round-trippable."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from . import contracts

VERSION = 'direct-ac-v1'


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def fingerprint(x):
    return hashlib.sha256(json.dumps(plain(x), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


@dataclass
class Plan:
    trees: list
    line_buys: list
    storage_buys: list

    def canonical(self):
        return Plan([sorted(t) for t in self.trees], sorted(map(tuple, self.line_buys)), sorted(map(tuple, self.storage_buys)))


@dataclass
class Operation:
    # K = stage, S = state, G = existing non-slack generator, N = new generator,
    # U = storage option, B = bus. SOC is energy at the START of each state.
    existing_p: np.ndarray  # K,S,G; SUB is calculated, never a gene
    existing_q: np.ndarray  # K,S,G
    new_p: np.ndarray       # K,S,N; optional new Q lives in Candidate.physical
    soc: np.ndarray         # K,U,S; last state ends at first start in its cycle
    conventional: np.ndarray  # K,B,S, after DR and before shedding
    ev: np.ndarray            # K,B,S; delivered EV load, cannot be shed
    cut: np.ndarray           # K,B,S; conventional only


@dataclass
class Candidate:
    plan: Plan
    new_capacity: np.ndarray  # K,N: purchases, NOT cumulative installed MW
    operation: Operation
    physical: dict = None  # Optional versioned extension; omitted in legacy IDs.

    def to_dict(self):
        value=dict(version=VERSION, plan=asdict(self.plan.canonical()),
                          new_capacity=self.new_capacity, operation=asdict(self.operation),
                          shapes={name:list(a.shape) for name,a in [('new_capacity',self.new_capacity),*vars(self.operation).items()]})
        if self.physical is not None:value['physical']=self.physical
        return plain(value)

    @classmethod
    def from_dict(cls, value):
        if value.get('version') != VERSION:
            raise ValueError('Unsupported candidate version')
        if set(value)-{'physical'} != {'version', 'plan', 'new_capacity', 'operation','shapes'}:
            raise ValueError('Unexpected candidate fields')
        if set(value['shapes'])!={'new_capacity',*value['operation']}:
            raise ValueError('Missing shape metadata')
        def array(name,data):
            shape=value['shapes'][name]
            if len(shape) not in (2,3) or any(type(n)!=int or n<0 for n in shape):
                raise ValueError('Invalid shape metadata')
            return np.asarray(data,dtype=float).reshape(shape)
        return cls(Plan(**value['plan']).canonical(), array('new_capacity',value['new_capacity']),
                   Operation(**{k: array(k,v) for k, v in value['operation'].items()}),copy.deepcopy(value.get('physical')))

    @property
    def candidate_id(self):
        return fingerprint(self.to_dict())

    @property
    def plan_id(self):
        value=dict(plan=asdict(self.plan.canonical()), new_capacity=self.new_capacity)
        if self.physical is not None:value['physical_installations']={k:v for k,v in self.physical.items() if k.endswith('_choice')}
        return fingerprint(value)


class Study:
    """Copied input catalogue plus an explicit electrical catalogue, on a 1 MVA base.

    Source dictionaries remain available for transparent pricing. No Pyomo model
    or optimized vector is needed to construct this object.
    """
    def __init__(self, data, electrical):
        self.data = copy.deepcopy(data)
        self.electrical = copy.deepcopy(electrical)
        d = self.data
        contracts.validate(d)
        self.c = d['config']
        from .financial_policy import resolve
        resolve(self.c)  # Validate without mutating the source catalogue.
        if self.c.get('storage_formulation','representative_day_SOC') != 'representative_day_SOC' or self.c.get('storage_energy_balance','none') != 'none':
            raise ValueError('Only the declared cyclic-day formulation without additional annual constraints is supported')
        for field in ('carbon_price_per_ton','unserved_energy_per_MWh','loss_cost_per_MWh','EENS_cost_per_MWh'):
            contracts.number(self.c[field],field)
        for field in ('substation_rated_MVA','substation_active_limit_MW'):
            contracts.number(self.c[field],field,1e-12)
        contracts.number(self.c['prices_per_MWh']['ESS'],'ESS price')
        contracts.number(self.c['emissions_ton_per_MWh']['ESS'],'ESS emissions')
        flex=d['extension'].get('flex',{})
        for field in ('shift_compensation_per_MWh','ev_compensation_per_MWh'):
            contracts.number(flex.get(field,0.),field)
        self.years = self.c['investment_years']
        self.K, self.S = len(self.years), len(d['scenarios'])
        if [s['id'] for s in d['scenarios']] != list(range(self.S)):
            raise ValueError('State IDs must be contiguous, ordered and zero-based')
        if [t['id'] for t in d['time_steps']] != [0]:
            raise ValueError('Only explicit representative-day states with one time row supported')
        if d['investment_mode'] not in ('single', 'multi'):
            raise ValueError('Unknown investment mode')
        self.buses = list(d['buses'])
        if len(set(self.buses)) != len(self.buses) or d['root'] not in self.buses:
            raise ValueError('Invalid buses/root')
        self.B = len(self.buses)
        self.bi = {b: i for i, b in enumerate(self.buses)}
        d['consumers'] = {int(k): v for k, v in d['consumers'].items()}
        if set(d['consumers']) != set(self.buses) or any(not np.isfinite(v) or v < 0 for v in d['consumers'].values()) or sum(d['consumers'].values()) <= 0:
            raise ValueError('Customer counts are required for reliability')
        self.lines = sorted(d['lines'], key=lambda e: e['id'])
        self.units = sorted(d['storage'], key=lambda u: u['id'])
        self.old = sorted([g for g in d['generators'] if g['technology'] != 'SUB'], key=lambda g: g['id'])
        subs = [g for g in d['generators'] if g['technology'] == 'SUB']
        if len(subs) != 1 or subs[0]['bus'] != d['root']:
            raise ValueError('Exactly one SUB at the root is required')
        self.sub = subs[0]
        self.new = sorted(d['extension']['generators'], key=lambda g: g['id'])
        self.G, self.N, self.U = len(self.old), len(self.new), len(self.units)
        for rows in (self.lines, self.units, d['generators']):
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate resource IDs')
        self.cycles = self.c['storage_scenario_cycles']
        self.hours = np.array(self.c['scenario_block_hours'], float)
        if sorted(s for cy in self.cycles for s in cy) != list(range(self.S)) or any(not cy for cy in self.cycles):
            raise ValueError('Daily cycles must partition the states exactly once')
        if self.hours.shape != (self.S,) or not np.all(np.isfinite(self.hours)) or min(self.hours) <= 0:
            raise ValueError('Invalid physical durations')
        if any(abs(sum(self.hours[cy])-24) > 1e-9 for cy in self.cycles):
            raise ValueError('This version models cyclic 24-hour representative days')
        for field in ('load_growth_by_stage', 'generation_growth_by_stage', 'SAIFI_limits', 'SAIDI_limits'):
            vals = np.asarray(self.c[field], float)
            if vals.shape != (self.K,) or not np.all(np.isfinite(vals)) or min(vals) < 0:
                raise ValueError('Invalid stage field: ' + field)
        self.av = {(a['generator'], a['scenario']): a for a in d['availability']}
        if any(a['time']!=0 for a in d['availability']) or any(r['time']!=0 for r in d['operations']):
            raise ValueError('Unexpected time index in state data')
        if len(self.av) != len(d['generators'])*self.S or len(self.av) != len(d['availability']):
            raise ValueError('Incomplete or duplicate availability')
        ops = {(r['bus'], r['scenario']): r for r in d['operations']}
        if len(ops) != self.B*self.S or len(ops) != len(d['operations']):
            raise ValueError('Incomplete or duplicate demand')
        self.base_load = np.array([[[ops[b,s]['load_MW']*self.c['load_growth_by_stage'][k] for s in range(self.S)] for b in self.buses] for k in range(self.K)])
        self.base_ev = np.array([[[ops[b,s]['ev_MW'] for s in range(self.S)] for b in self.buses] for k in range(self.K)])
        for a in (self.base_load, self.base_ev):
            if not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError('Invalid demand values')
        self.probabilities = np.array([s['probability'] for s in d['scenarios']])
        # Each cycle is one representative day: its blocks must occur equally
        # often per year. Otherwise a cyclic SOC or energy-neutral DR/EV shift
        # can create/destroy annual energy when weighted by 8760*probability.
        # Different days may still have different occurrence counts.
        for cy in self.cycles:
            occurrences=self.c['annual_hours']*self.probabilities[cy]/self.hours[cy]
            if not np.allclose(occurrences,occurrences[0],rtol=1e-10,atol=1e-12):
                raise ValueError('Inconsistent occurrence weights inside a representative-day cycle')
        el = self.electrical
        if el.get('Sbase_MVA') != 1 or not np.isfinite(el['Ibase_A']) or el['Ibase_A'] <= 0:
            raise ValueError('Explicit 1 MVA electrical base and positive current base required')
        for key, shape in [('q_ratio',(self.B,self.S)), ('vmin',(self.B,)), ('vmax',(self.B,))]:
            a = np.asarray(el[key], float)
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError('Electrical shape/value mismatch: ' + key)
            el[key] = a.tolist()
        if np.any(np.asarray(el['vmin']) <= 0) or np.any(np.asarray(el['vmin']) > np.asarray(el['vmax'])):
            raise ValueError('Invalid voltage bounds')
        if not np.isfinite(el['vslack']) or not 0 < el['vslack'] < 2:
            raise ValueError('Invalid slack voltage')
        eopts = {(r['line'], r['option']): r for r in el['branches']}
        if len(eopts) != len(el['branches']):
            raise ValueError('Duplicate electrical option')
        for e in self.lines:
            if e['i'] not in self.bi or e['j'] not in self.bi or e['i'] == e['j'] or len(e['options'])<2 or [o['id'] for o in e['options']] != list(range(len(e['options']))):
                raise ValueError('Invalid consecutive-option branch catalogue')
            for o in e['options']:
                r = eopts[e['id'],o['id']]
                if not all(np.isfinite(r[key]) for key in ('r','x','b')) or r['r'] < 0 or abs(complex(r['r'],r['x'])) == 0:
                    raise ValueError('Invalid branch impedance')
                for key in ('capacity_MW','nominal_current_A'):
                    if not np.isfinite(o[key]) or o[key] <= 0:
                        raise ValueError('Invalid thermal rating')
                for key in ('capex','maintenance_per_year'):
                    contracts.number(o[key],key)
                if len(o['failure_rate']) != 2 or len(o['repair_hours']) != 2 or any(not np.isfinite(f) or f < 0 for f in o['failure_rate']) or any(not np.isfinite(r) or not 0 <= r < 8760 for r in o['repair_hours']):
                    raise ValueError('Invalid directional reliability data')
        self.eopts = eopts
        for u in self.units:
            if u['bus'] not in self.bi:
                raise ValueError('Unknown BESS site')
            for key in ('energy_MWh','power_MW','eta_charge','eta_discharge'):
                if not np.isfinite(u[key]) or u[key] <= 0:
                    raise ValueError('Invalid BESS capability')
            if max(u['eta_charge'],u['eta_discharge']) > 1:
                raise ValueError('Efficiency greater than one')
            for key in ('capex','maintenance_per_year'):
                contracts.number(u[key],key)
        for g in d['generators']:
            if g['bus'] not in self.bi:
                raise ValueError('Unknown existing generator bus')
            for key in ('price','curtail_price'):
                contracts.number(g[key],key)
            qlo,qhi = el['generator_q'][str(g['id'])]
            if not np.isfinite(qlo+qhi) or qlo > qhi:
                raise ValueError('Invalid Q limits')
            for s in range(self.S):
                a = self.av[g['id'],s]
                if not all(np.isfinite(a[x]) for x in ('minimum','maximum','emissions')) or not 0 <= a['minimum'] <= a['maximum'] or a['emissions'] < 0:
                    raise ValueError('Invalid active capability/emissions')
        if 'physical' in el:
            from .physical import validate_catalogue
            validate_catalogue(self)
        # Ownership/scope guard only: does not change coefficients or fingerprints.
        from .economic_policy import validate_study_incidence
        validate_study_incidence(d)
        self.input_id = fingerprint(dict(version=VERSION, data=self.data, electrical=self.electrical))
        # Derived arrays also affect decoding, physics and cost. Protect them,
        # not just the source dictionaries, against stale identities/cache hits.
        self.runtime_id = fingerprint(vars(self))

    def check_unchanged(self):
        if fingerprint({k:v for k,v in vars(self).items() if k!='runtime_id'}) != self.runtime_id:
            raise ValueError('Study or derived data changed during its lifetime')

    def shape_check(self, candidate):
        from .physical import validate_candidate
        validate_candidate(self,candidate)
        shapes = dict(existing_p=(self.K,self.S,self.G), existing_q=(self.K,self.S,self.G),
                      new_p=(self.K,self.S,self.N), soc=(self.K,self.U,self.S),
                      conventional=(self.K,self.B,self.S), ev=(self.K,self.B,self.S), cut=(self.K,self.B,self.S))
        for key, shape in [('new_capacity',(self.K,self.N)), *shapes.items()]:
            a = candidate.new_capacity if key == 'new_capacity' else getattr(candidate.operation,key)
            if not isinstance(a,np.ndarray) or a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError('Invalid candidate array: ' + key)

    def active_capacity(self, candidate):
        return np.array([[sum(candidate.new_capacity[j,n] for j,y in enumerate(self.years) if contracts.active(g,y,year)) for n,g in enumerate(self.new)] for year in self.years]).reshape(self.K,self.N)

    def storage_active(self, plan):
        purchases = dict(plan.storage_buys)
        return np.array([[u['id'] in purchases and purchases[u['id']] <= k for u in self.units] for k in range(self.K)], bool).reshape(self.K,self.U)

    def existing_bounds(self):
        lo = np.array([[[self.av[g['id'],s]['minimum']*self.c['generation_growth_by_stage'][k] for g in self.old] for s in range(self.S)] for k in range(self.K)]).reshape(self.K,self.S,self.G)
        hi = np.array([[[self.av[g['id'],s]['maximum']*self.c['generation_growth_by_stage'][k] for g in self.old] for s in range(self.S)] for k in range(self.K)]).reshape(lo.shape)
        qlo = np.zeros(lo.shape); qhi = np.zeros(lo.shape)
        for j,g in enumerate(self.old):
            qlo[:,:,j],qhi[:,:,j] = [np.where(hi[:,:,j] > 1e-9, bound, 0.) for bound in self.electrical['generator_q'][str(g['id'])]]
        return lo,hi,qlo,qhi
