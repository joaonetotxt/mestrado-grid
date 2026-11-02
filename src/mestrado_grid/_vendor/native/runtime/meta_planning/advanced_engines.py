"""QPSO Type 2 and SHADE (2013) operators; see ADVANCED_METHODS.md.

Only evaluated positions enter memories. No physical/economic information is
used to propose coordinates. Constraints enter through the common comparator.
"""
import copy
import numpy as np


def qpso_alpha(engine):
    p=engine.config.parameters
    progress=engine.budget_progress if engine.config.progress_schedule=='budget' else (engine.generation-1)/max(1,engine.config.generations-1)
    return p['alpha_start']+(p['alpha_end']-p['alpha_start'])*np.clip(progress,0,1)


def propose_qpso(engine):
    rng=engine.rng;shape=engine.population.shape
    phi=rng.random(shape)
    attractor=phi*engine.personal+(1-phi)*engine.best['x']
    mbest=engine.personal.mean(axis=0)
    u=np.maximum(rng.random(shape),np.finfo(float).tiny)
    sign=np.where(rng.random(shape)<.5,-1.,1.)
    return np.clip(attractor+sign*qpso_alpha(engine)*abs(mbest-engine.population)*(-np.log(u)),0,1)


def initialize_shade(h,n):
    return dict(memory_f=[.5]*h,memory_cr=[.5]*h,index=0,updates=0,
                archive=[],trial_f=[.5]*n,trial_cr=[.5]*n,last_weight_mode=None)


def propose_shade(engine,key):
    rng=engine.rng;state=engine.shade;pop=engine.population;n,d=pop.shape
    order=sorted(range(n),key=lambda j:key(engine.records[j]))
    archive=np.asarray(state['archive'],float).reshape(-1,d)
    union=np.vstack((pop,archive));out=np.empty_like(pop)
    pmin=2/n;pmax=max(pmin,engine.config.parameters['p_max'])
    for i in range(n):
        r=int(rng.integers(len(state['memory_f'])))
        cr=float(np.clip(rng.normal(state['memory_cr'][r],.1),0,1))
        f=state['memory_f'][r]+.1*rng.standard_cauchy()
        while not np.isfinite(f) or f<=0:f=state['memory_f'][r]+.1*rng.standard_cauchy()
        f=float(min(f,1.))
        p=float(rng.uniform(pmin,pmax))
        top=min(n,max(2,int(np.floor(n*p+.5))))
        pbest=order[int(rng.integers(top))]
        r1=int(rng.choice([j for j in range(n) if j!=i]))
        r2=int(rng.choice([j for j in range(len(union)) if j not in (i,r1)]))
        mutant=pop[i]+f*(pop[pbest]-pop[i])+f*(pop[r1]-union[r2])
        # JADE/SHADE boundary repair: midpoint of parent and violated bound.
        mutant=np.where(mutant<0,pop[i]/2,np.where(mutant>1,(pop[i]+1)/2,mutant))
        mask=rng.random(d)<cr;mask[int(rng.integers(d))]=True
        out[i]=np.where(mask,mutant,pop[i])
        state['trial_f'][i]=f;state['trial_cr'][i]=cr
    return out


def success_weights(pairs):
    """Recover paper's objective weights for feasible pairs.

    Infeasible improvements use measured violation only when all successes are
    in that same rank class. Transitions/mixed classes use equal weights, never
    a subtraction of currency from normalized violation.
    """
    classes={a[0] for a,b in pairs}|{b[0] for a,b in pairs}
    mode='objective' if classes=={0} else 'violation' if classes=={1} else 'uniform_rank_transition'
    if len(classes)==1 and next(iter(classes)) in (0,1):
        # Scale before subtraction to avoid overflowing finite objective values.
        scale=max(max(abs(a[1]),abs(b[1])) for a,b in pairs) or 1.
        delta=np.array([a[1]/scale-b[1]/scale for a,b in pairs])
        if np.all(delta>0) and np.isfinite(delta).all() and delta.sum()>0:return delta/delta.sum(),mode
        mode='uniform_roundoff'
    return np.full(len(pairs),1/len(pairs)),mode


def accept_shade(engine,key):
    state=engine.shade;success=[];pairs=[]
    for i,record in enumerate(engine.received):
        old=key(engine.records[i]);new=key(record)
        if engine.restart_pending[i] is not None:continue
        if new<old:
            state['archive'].append(engine.population[i].tolist())
            if engine.native_pending_credit[i]:success.append(i);pairs.append((old,new))
        if new<=old:
            engine.population[i]=engine.pending[i]
            engine.records[i]=copy.deepcopy(record)
    if len(state['archive'])>engine.n:
        selected=engine.rng.choice(len(state['archive']),engine.n,replace=False)
        state['archive']=[state['archive'][int(j)] for j in selected]
    if success:
        weights,mode=success_weights(pairs)
        f=np.asarray(state['trial_f'])[success];cr=np.asarray(state['trial_cr'])[success]
        k=state['index']
        state['memory_f'][k]=float(np.clip(np.sum(weights*f*f)/np.sum(weights*f),np.finfo(float).tiny,1))
        state['memory_cr'][k]=float(np.clip(np.sum(weights*cr),0,1))
        state['updates']+=1;state['index']=state['updates']%len(state['memory_f'])
        state['last_weight_mode']=mode


def validate_shade(engine):
    s=engine.shade;h=engine.config.parameters['memory_size'];n=engine.n
    if not isinstance(s,dict) or set(s)!=set(initialize_shade(h,n)):raise ValueError('Invalid SHADE memory schema')
    for name,size,positive in [('memory_f',h,True),('memory_cr',h,False),('trial_f',n,True),('trial_cr',n,False)]:
        v=s[name]
        if not isinstance(v,list) or len(v)!=size or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in v):raise ValueError('Invalid SHADE '+name)
        a=np.asarray(v,float)
        if not np.isfinite(a).all() or np.any(a>1) or np.any(a<=0 if positive else a<0):raise ValueError('Invalid SHADE '+name)
    if type(s['updates'])!=int or not 0<=s['updates']<=max(0,engine.generation-1):raise ValueError('Invalid SHADE updates')
    if type(s['index'])!=int or s['index']!=s['updates']%h:raise ValueError('Invalid SHADE memory index')
    if not isinstance(s['archive'],list) or len(s['archive'])>n:raise ValueError('Invalid SHADE archive size')
    if s['archive']:
        a=np.asarray(s['archive'],float)
        if a.shape!=(len(s['archive']),engine.d) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid SHADE archive')
    if (s['updates']==0)!=(s['last_weight_mode'] is None) or s['last_weight_mode'] not in (None,'objective','violation','uniform_rank_transition','uniform_roundoff'):raise ValueError('Invalid SHADE weight mode')
    if s['updates']==0 and any(x!=.5 for x in s['memory_f']+s['memory_cr']):raise ValueError('Premature SHADE memory')
