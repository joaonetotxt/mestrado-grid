"""Isolated hybrid: finite resource neighbourhoods + risk-directed joint LP.

Only the frozen complete AC evaluator ranks candidates. Tail weights are a
local subgradient, NOT an exact optimizer of CVaR. Full repricing handles tail
changes. No relaxed reliability ceiling, no historical finalist import.
"""
from common import *
from joint_dispatch_base import JointRefiner
from resource_base import apply as apply_resource
from meta_planning.guided_blocks import cycle_path
from collections import Counter

FAMILIES=['dispatch','topology','cable','BESS_ESS1','BESS_ESS2','new_PV',
          'new_WT','new_BIO','capacitor','regulator','retrofit','planned_reduce',
          'storage_remove','device_controls']

def tail_weights(costs,p,alpha,beta):
    """Subgradient of (1-beta) E + beta CVaR with fractional probability mass."""
    _,tail=economics.tail(costs,p,alpha)
    w=(1-beta)*np.asarray(p,float)
    for r in tail:w[r['state_id']]+=beta*r['tail_mass']/(1-alpha)
    assert np.isclose(w.sum(),1.) and np.min(w)>=0
    return w

class RiskRefiner(JointRefiner):
    def __init__(self,st):super().__init__(st);self.scenario_weights=np.asarray(st.probabilities)
    def propose(self,c,k,cy,rows,radius=.5,record=None):
        if record is not None and record.get('metrics'):
            f=record['metrics']['finance']
            costs=f.get('conditional_costs',record['metrics'].get('conditional_costs_VP'))
            self.scenario_weights=tail_weights(costs,self.st.probabilities,self.st.c['alpha'],self.st.c['beta'])
        return super().propose(c,k,cy,rows,radius)

def decorate(r):
    if r.get('metrics') and r.get('failed_cases')==0:
        r['search_selection']='feasibility_violation_exact_cost_tie_v1'
    return r

def opportunities(st,c,r,seed,visit):
    """Finite adaptive menu. All catalogue sites eligible across revisits.

    Includes partial reliability improvements from an infeasible MAT start;
    feasibility is never required by this proposal screen.
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,visit,8107]))
    ph=st.electrical['physical'];rows=r.get('cases',[])
    low={b:2. for b in st.buses if b!=st.data['root']}
    for row in rows:
        if not row.get('converged'):continue
        for b,v in zip(row['bus_order'],np.hypot(row['voltage_real'],row['voltage_imag'])):
            if b in low:low[b]=min(low[b],float(v))
    sites=list(low);rng.shuffle(sites);sites=list(dict.fromkeys(sorted(low,key=low.get)[:2]+sites))
    selected=sites[:4];out=[]
    def add(f,**d):out.append(dict(family=f,**d))
    for bus in selected:
        occupied=any(next(u for u in st.units if u['id']==uid)['bus']==bus for uid,k in c.plan.storage_buys)
        if not occupied:
            for u in st.units:
                if u['bus']==bus:
                    for stage in range(st.K):add('BESS_'+u['id'].split('@')[0],unit=u['id'],stage=stage,q=True)
        for j,g in enumerate(st.new):
            if g['bus']!=bus:continue
            remaining=g['max_MW']-c.new_capacity[:,j].sum()
            for amount in (.1,.5,remaining*.25):
                if remaining>0:add('new_'+g['technology'],j=j,amount=min(amount,remaining),stage=int(rng.integers(st.K)))
        if bus in ph['capacitor_sites']:
            j=ph['capacitor_sites'].index(bus)
            if not c.physical['cap_choice'][j]:
                for opt in range(len(ph['capacitors'])):add('capacitor',j=j,option=opt,stage=int(rng.integers(st.K)))
    installed,_=line_schedule(st.data,c.plan)
    edges=st.lines.copy();rng.shuffle(edges)
    for e in edges[:6]:
        for stage in range(st.K):
            if installed[e['id'],stage] is None:continue
            for o in range(len(e['options'])):
                if o!=installed[e['id'],stage] and not(e['existing'] and o==0):add('cable',edge=e['id'],option=o,stage=stage)
        j=ph['regulator_sites'].index(e['id'])
        if not c.physical['reg_choice'][j]:
            for opt in range(len(ph['regulators'])):
                add('regulator',j=j,option=opt,stage=int(rng.integers(st.K)),tap=int(rng.choice([-4,-2,2,4])))
    gens=list(range(st.G));rng.shuffle(gens)
    for j in gens[:12]:
        if not c.physical['retro_choice'][j]:add('retrofit',j=j,stage=int(rng.integers(st.K)))
    for j in range(st.N):
        if np.any(c.new_capacity[:,j]):add('planned_reduce',j=j,fraction=.5)
    for uid,k in c.plan.storage_buys:add('storage_remove',unit=uid)
    for choice,control in [('cap_choice','cap_state'),('reg_choice','tap')]:
        for j,installed_choice in enumerate(c.physical[choice]):
            if installed_choice:add('device_controls',control=control,j=j,fraction=.5)
    # Reliability screen only: every cycle opening of a seed-varied edge subset.
    # No AC solves or undisclosed optimized topology is used in the screen.
    top=[]
    for stage in range(st.K):
        closing=[e for e in edges if e['id'] not in c.plan.trees[stage]][:8]
        for e in closing:
            for drop in cycle_path(st.data,c.plan.trees[stage],e['id']):
                z=ca.exchange(st,c,e['id'],drop,stage)
                rr=reliability(st.data,z.plan)
                score=sum(max(0,x['SAIDI']/st.c['SAIDI_limits'][k]-1)+max(0,x['SAIFI']/st.c['SAIFI_limits'][k]-1) for k,x in enumerate(rr))
                cost=sum(economics.deterministic(st,z).values())
                top.append((score,cost,dict(family='topology',close=e['id'],drop=drop,stage=stage)))
    out.extend(d for score,cost,d in sorted(top,key=lambda x:x[:2])[:12])
    return out

def apply(st,c,d,r):
    f=d['family']
    if f=='topology':return ca.exchange(st,c,d['close'],d['drop'],d['stage'])
    if f in ('planned_reduce','storage_remove','device_controls'):
        z=copy.deepcopy(c)
        if f=='planned_reduce':
            j=d['j'];z.new_capacity[:,j]*=d['fraction'];z.operation.new_p[:,:,j]*=d['fraction']
            q=np.asarray(z.physical['new_q']);q[:,:,j]*=d['fraction'];z.physical['new_q']=q.tolist()
        elif f=='storage_remove':
            j=next(j for j,u in enumerate(st.units) if u['id']==d['unit'])
            z.plan.storage_buys=[x for x in z.plan.storage_buys if x[0]!=d['unit']]
            z.physical['bess_choice'][j]=0
            z.operation.soc[:,j,:]=0;q=np.asarray(z.physical['bess_q']);q[:,:,j]=0;z.physical['bess_q']=q.tolist()
        else:
            a=np.asarray(z.physical[d['control']]);a[:,:,d['j']]=np.rint(a[:,:,d['j']]*d['fraction']);z.physical[d['control']]=a.tolist()
        normalize(st,z);return z
    return apply_resource(st,c,d,r)

def day_order(st,r):
    if not r.get('metrics'):return [(k,list(cy)) for k in range(st.K) for cy in st.cycles]
    f=r['metrics']['finance'];costs=f.get('conditional_costs',r['metrics']['conditional_costs_VP'])
    w=tail_weights(costs,st.probabilities,st.c['alpha'],st.c['beta'])
    rows={(x['stage'],x['state_id']):x for x in r['cases']}
    scored=[]
    for k in range(st.K):
        for cy in st.cycles:
            violation=sum(sum(rows[k,s]['normalized_violations'].values()) for s in cy)
            expense=sum(w[s]*sum(rows[k,s]['hourly_costs'].values()) for s in cy)
            scored.append((-violation,-expense,k,list(cy)))
    return [(k,cy) for v,c,k,cy in sorted(scored)]
