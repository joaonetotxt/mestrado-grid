"""Declared auxiliary search: local AC sensitivities, exact AC acceptance.

This is NOT a globally optimal AC OPF or a native metaheuristic gain. The
linear program only proposes a simultaneous interbus redispatch. Unchanged
nonlinear AC, reliability, temporal constraints and financial rank accept it.
"""
import copy
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from . import physical, economics
from .physical_guidance import response
from .evaluator import Evaluator, rank
from .temporal import storage_dispatch
from .topology import line_schedule, reliability, repair_trees
from .guided_blocks import cycle_path


def prepare_normal(st, candidate, pairs):
    """Explicit normal-counterfactual profiles, preserving complete day cycles.

    pairs is a caller-supplied zero-based extreme -> matching normal mapping.
    Probabilities/hours are preserved, NOT conditional deletion of five states.
    """
    from .schema import Study
    from .temporal import normalize
    data=copy.deepcopy(st.data);el=copy.deepcopy(st.electrical);c=copy.deepcopy(candidate)
    if not pairs or any(t==s or t not in range(st.S) or s not in range(st.S) for t,s in pairs.items()):
        raise ValueError('Explicit valid counterpart mapping required')
    for t,s in pairs.items():
        if st.hours[t]!=st.hours[s]:raise ValueError('Counterpart has a different duration')
    for name,key in [('operations','bus'),('availability','generator')]:
        lookup={(r[key],r['scenario']):r for r in data[name]}
        for r in data[name]:
            if r['scenario'] in pairs:
                t=r['scenario'];r.update(copy.deepcopy(lookup[r[key],pairs[t]]));r['scenario']=t
    for g in data['extension']['generators']:
        for t,s in pairs.items():g['availability'][str(t)]=g['availability'][str(s)]
    a=np.asarray(el['q_ratio']);
    for t,s in pairs.items():a[:,t]=a[:,s]
    el['q_ratio']=a.tolist()
    for key in ('new_sources','old_sources','bess_sources','capacitors','regulators'):
        for r in el.get('physical',{}).get(key,[]):
            for t,s in pairs.items():r['electrical_available'][t]=r['electrical_available'][s]
    # Copy whole matched operating days, not disconnected SOC/DR boundaries.
    controls={}
    for cy in st.cycles:
        affected=[t for t in cy if t in pairs]
        if not affected:continue
        offsets={pairs[t]-t for t in affected}
        if len(offsets)!=1:raise ValueError('No consistent counterpart day')
        offset=offsets.pop();source=[t+offset for t in cy]
        if source not in st.cycles:raise ValueError('Counterpart is not a complete day')
        controls.update(zip(cy,source))
    for name in ('existing_p','existing_q','new_p'):
        for t,s in controls.items():getattr(c.operation,name)[:,t]=getattr(candidate.operation,name)[:,s]
    for name in ('conventional','ev','cut','soc'):
        for t,s in controls.items():getattr(c.operation,name)[:,:,t]=getattr(candidate.operation,name)[:,:,s]
    if c.physical:
        for name in ('new_q','bess_q','cap_state','tap'):
            a=np.asarray(c.physical[name])
            for t,s in controls.items():a[:,t]=np.asarray(candidate.physical[name])[:,s]
            c.physical[name]=a.tolist()
    data['normal_counterfactual']=dict(mapping={str(t+1):s+1 for t,s in pairs.items()},
        probability_policy='unchanged complete-day occurrences',not_conditional_27_state_repricing=True,
        initial_controls='complete counterpart days, preserving SOC and daily demand energy')
    data['castro_extreme_overlay']['active']=False
    new=Study(data,el);repairs=normalize(new,c)
    return new,c,repairs


def operating_bounds(st,c):
    return (*physical.existing_bounds(st,c),physical.q_bounds(st,c))


def linear_step(st,c,row,radius=.5):
    """LP in physical MW/MVAr. Both service increase and decrease are eligible."""
    k,s=row['stage'],row['state_id'];op=c.operation
    V=np.array(row['voltage_real'])+1j*np.array(row['voltage_imag'])
    model=response(st,c,k,s,V);n=st.B-1;bybus={b:i for i,b in enumerate(model['buses'])}
    lower,upper,*_=physical.existing_bounds(st,c);ql,qh,nl,nh,bl,bh=physical.q_bounds(st,c)
    active=st.active_capacity(c)[k]*physical.availability(st)[s]
    cols=[];bounds=[];direct=[];names=[]
    def add(name,j,bus,p,q,lo,hi,cost):
        if hi-lo<1e-12:return
        col=np.zeros(2*n);i=bybus[bus]
        if i==0:raise ValueError('Local root resource needs explicit slack conversion')
        col[i-1]=p;col[n+i-1]=q;cols.append(col);bounds.append((float(lo),float(hi)))
        direct.append(float(cost));names.append((name,j))
    for b,bus in enumerate(st.buses):
        if bus==st.data['root']:continue
        load=op.conventional[k,b,s];cut=op.cut[k,b,s]
        add('restore',b,bus,-1,-st.electrical['q_ratio'][b][s],
            -min(load-cut,radius*max(load,.01)),min(cut,radius*max(load,.01)),-st.c['unserved_energy_per_MWh'])
    for j,g in enumerate(st.old):
        p=op.existing_p[k,s,j];span=max(upper[k,s,j],.01)*radius
        emission=st.av[g['id'],s]['emissions']
        cost=g['price']+st.c['carbon_price_per_ton']*emission-(g['curtail_price'] if g.get('renewable') else 0.)
        add('existing_p',j,g['bus'],1,0,max(lower[k,s,j]-p,-span),min(upper[k,s,j]-p,span),cost)
        q=op.existing_q[k,s,j]
        add('existing_q',j,g['bus'],0,1,max(ql[k,s,j]-q,-span),min(qh[k,s,j]-q,span),0.)
    for j,g in enumerate(st.new):
        p=op.new_p[k,s,j];span=max(active[j],.01)*radius
        cost=g['variable_per_MWh']+st.c['carbon_price_per_ton']*g['emissions_ton_per_MWh']-(g['curtail_per_MWh'] if g['technology'] in ('PV','WT') else 0.)
        add('new_p',j,g['bus'],1,0,max(-p,-span),min(active[j]-p,span),cost)
        q=c.physical['new_q'][k][s][j]
        add('new_q',j,g['bus'],0,1,max(nl[k,s,j]-q,-span),min(nh[k,s,j]-q,span),0.)
    if not cols:return None,dict(reason='no_free_controls')
    A=np.array(cols).T;d=model['derivatives'];base=model['base'];ineq=[];rhs=[]
    def limits(der,low,high,value):
        der=np.atleast_2d(der)@A;value=np.atleast_1d(value)
        ineq.extend(der);rhs.extend(np.broadcast_to(high,value.shape)-value)
        ineq.extend(-der);rhs.extend(value-np.broadcast_to(low,value.shape))
    limits(d['voltage'],model['voltage_lower'],model['voltage_upper'],base['voltage'])
    limits(d['branch_I'],0.,.999,base['branch_I'])
    limits(d['branch_S'],0.,.999,base['branch_S'])
    pl,ph,qlo,qhi,sm=physical.sub_limits(st,k,s)
    sp,sq=row['sub_P_MW'],row['sub_Q_MVAr'];mag=np.hypot(sp,sq)
    limits(d['sub_P'],pl,ph,sp);limits(d['sub_Q'],qlo,qhi,sq)
    limits((sp*d['sub_P']+sq*d['sub_Q'])/max(mag,1e-9),0.,sm*.999,mag)
    subprice=st.sub['price']+st.c['carbon_price_per_ton']*st.av[st.sub['id'],s]['emissions']
    objective=np.array(direct)+(subprice*d['sub_P']+st.c['loss_cost_per_MWh']*d['loss'])@A
    result=linprog(objective/3000.,A_ub=np.array(ineq),b_ub=np.array(rhs),bounds=bounds,method='highs')
    meta=dict(lp_success=result.success,lp_status=result.status,variables=len(names),radius=radius,
              jacobian_residual=model['jacobian_residual'],role='auxiliary_linear_proposal_exact_AC_required')
    if not result.success:return None,meta
    meta['predicted_hourly_gain']=-float(objective@result.x)
    return (names,result.x),meta


def apply_step(st,c,k,s,step,fraction):
    out=copy.deepcopy(c);names,delta=step
    for (name,j),dx in zip(names,delta*fraction):
        if name=='restore':out.operation.cut[k,j,s]-=dx
        elif name=='new_q':out.physical[name][k][s][j]+=dx
        else:getattr(out.operation,name)[k,s,j]+=dx
    out.operation.cut=np.clip(out.operation.cut,0.,out.operation.conventional)
    lo,hi,*_=physical.existing_bounds(st,out)
    out.operation.existing_p=np.clip(out.operation.existing_p,lo,hi)
    out.operation.new_p=np.clip(out.operation.new_p,0.,st.active_capacity(out)[:,None,:]*physical.availability(st)[None,:,:])
    ql,qh,nl,nh,_,_=physical.q_bounds(st,out)
    out.operation.existing_q=np.clip(out.operation.existing_q,ql,qh)
    out.physical['new_q']=np.clip(out.physical['new_q'],nl,nh).tolist()
    return out


class ACRefiner:
    def __init__(self,st,policy=None):
        self.study=st;self.evaluator=Evaluator(st,policy);self.counts=Counter();self.history=[]

    def evaluate(self,c):
        r=self.evaluator.evaluate(c);self.counts['full_evaluations']+=1
        self.counts['AC_cases']+=self.study.K*self.study.S
        return r

    def refine(self,c,rounds=2,cases=None,radius=.5,callback=None):
        """Acceptance uses the unmodified global economic objective.

        Local hourly steps are merely proposals; the completed portfolio must
        improve the original rank, also when beta is positive.
        """
        st=self.study;c=copy.deepcopy(c);initial=self.evaluate(c)
        if initial['status'] not in ('feasible','violated') or not all(r['converged'] for r in initial['cases']):
            return c,initial,dict(reason='no_converged_AC_linearization')
        if any(v>self.evaluator.policy.temporal_tolerance for v in initial['temporal_violations'].values()) or not all(r['passed'] for r in initial['reliability']):
            return c,initial,dict(reason='discrete_or_temporal_violation')
        best=copy.deepcopy(c);best_r=initial;rows={(r['stage'],r['state_id']):r for r in initial['cases']}
        # A topology or storage change can violate previously quiet states.
        # Always include every affected failed AC case in the restoration pass.
        selected=list(rows) if cases is None else list(dict.fromkeys(list(cases)+[ks for ks,row in rows.items() if not row['passed']]))
        for iteration in range(rounds):
            accepted=0
            # Probability is used only for ordering, never to exclude a state.
            selected.sort(key=lambda ks:-st.probabilities[ks[1]]*sum(rows[ks]['hourly_costs'].values()))
            for k,s in selected:
                row=rows[k,s];step,meta=linear_step(st,c,row,radius)
                self.counts['LP_proposals']+=1
                # Restoring an infeasible case may require more expensive service
                # or shedding. The complete unchanged feasibility rank accepts it.
                if step is None or (row['passed'] and meta.get('predicted_hourly_gain',0)<1e-5):continue
                before=sum(row['hourly_costs'].values());outcome=False
                for fraction in (1.,.5,.25,.125,.0625):
                    trial=apply_step(st,c,k,s,step,fraction)
                    installed,_=line_schedule(st.data,trial.plan);top=self.evaluator._edges(trial,k,installed)
                    ch,di=storage_dispatch(st,trial)
                    actual=self.evaluator._case(trial,k,s,top,ch,di,operating_bounds(st,trial))
                    self.counts['AC_cases']+=1;self.counts['backtracking_evaluations']+=1
                    improved_cost=actual.get('hourly_costs') and sum(actual['hourly_costs'].values())<before-1e-7
                    restored_feasibility=not row['passed'] and actual.get('passed',False)
                    reduced_violation=(not row['passed'] and bool(actual.get('normalized_violations'))
                        and sum(actual['normalized_violations'].values())<sum(row['normalized_violations'].values()))
                    if actual['converged'] and ((actual['passed'] and (improved_cost or restored_feasibility)) or reduced_violation):
                        c=trial;rows[k,s]=actual;accepted+=1;outcome=True;break
                self.history.append(dict(stage=k,state=s,iteration=iteration,accepted=outcome,
                    fraction=fraction,**meta))
                if callback:callback(self,k,s)
            result=self.evaluate(c)
            if rank(result)<rank(best_r):best=copy.deepcopy(c);best_r=result
            else:c=copy.deepcopy(best);rows={(r['stage'],r['state_id']):r for r in best_r['cases']}
            if not accepted:break
        return best,best_r,dict(counts=dict(self.counts),native_gain=False)


def replace_line(st,c,edge,option,stage):
    out=copy.deepcopy(c)
    # Edit exactly one dated purchase. Nominal IDs must never erase later
    # investments; validate the resulting complete chronology explicitly.
    out.plan.line_buys=[p for p in out.plan.line_buys if not (p[0]==edge and p[2]==stage)]
    out.plan.line_buys.append((edge,option,stage));line_schedule(st.data,out.plan)
    return out


def exchange(st,c,close,drop,stage):
    out=copy.deepcopy(c);trees=copy.deepcopy(c.plan.trees)
    for k in range(stage,st.K):
        if close not in trees[k] and drop in cycle_path(st.data,trees[k],close):
            trees[k]=[e for e in trees[k] if e!=drop]+[close]
    out.plan=repair_trees(st.data,out.plan,trees)
    return out


def storage_cycle(st,c,unit_id,stage,donor,target,energy_fraction=1.):
    """Install+charge+hold+discharge in complete cycles, no free SOC energy."""
    if not 0<=energy_fraction<=1:raise ValueError('Invalid energy fraction')
    out=copy.deepcopy(c);u=next(j for j,v in enumerate(st.units) if v['id']==unit_id);unit=st.units[u]
    if any(st.units[next(j for j,v in enumerate(st.units) if v['id']==name)]['bus']==unit['bus'] for name,_ in out.plan.storage_buys):
        raise ValueError('Storage already installed at this site')
    cy=next(cy for cy in st.cycles if donor in cy)
    if target not in cy or donor==target:raise ValueError('Pulse must close inside one original day')
    out.plan.storage_buys.append((unit_id,stage));line_schedule(st.data,out.plan)
    amount=energy_fraction*min(unit['energy_MWh'],unit['power_MW']*unit['eta_charge']*st.hours[donor],unit['power_MW']*st.hours[target]/unit['eta_discharge'])
    arc=[];j=(cy.index(donor)+1)%len(cy)
    while True:
        arc.append(cy[j])
        if cy[j]==target:break
        j=(j+1)%len(cy)
    out.operation.soc[stage:,u,arc]=amount
    b=st.bi[unit['bus']]
    for k in range(stage,st.K):
        delivered=amount*unit['eta_discharge']/st.hours[target]
        out.operation.cut[k,b,target]-=min(delivered,out.operation.cut[k,b,target])
    return out


def compare_investment(refiner,reference,proposal,cases=None,rounds=1):
    """Equal case scope and local-iteration cap for an investment comparison.

    Actual LP, nonlinear backtracking and case counts can still differ and
    must be reported. This is a finite local comparison, not causal proof or
    an optimal investment valuation.
    """
    preview=refiner.evaluate(proposal)
    if preview['status'] not in ('feasible','violated') or not preview.get('cases'):
        return reference,refiner.evaluate(reference),dict(rejected=preview['status'])
    selected=list(dict.fromkeys((list(cases) if cases is not None else [(k,s) for k in range(refiner.study.K) for s in range(refiner.study.S)])+
        [(r['stage'],r['state_id']) for r in preview['cases'] if not r['passed']]))
    before=Counter(refiner.counts)
    control,rc,_=refiner.refine(reference,rounds=rounds,cases=selected)
    control_calls=dict(refiner.counts-before);before=Counter(refiner.counts)
    candidate,rp,_=refiner.refine(proposal,rounds=rounds,cases=selected)
    proposal_calls=dict(refiner.counts-before);accepted=rank(rp)<rank(rc)
    def value(r):return r['metrics']['selection_objective'] if r['status']=='feasible' else None
    audit=dict(cases=selected,rounds=rounds,control_FO=value(rc),proposal_FO=value(rp),
        accepted=accepted,control_counts=control_calls,proposal_counts=proposal_calls,
        control_status=rc['status'],proposal_status=rp['status'],cut_zero_requirement=False)
    return (candidate,rp,audit) if accepted else (control,rc,audit)


def main():
    import argparse,json
    from pathlib import Path
    from .schema import Study,Candidate,plain
    p=argparse.ArgumentParser(description='Explicit auxiliary AC-guided redispatch; original economic objective')
    p.add_argument('--study',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    if args.rounds<1:raise ValueError('At least one finite pass is required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(Path(path).read_text('utf-8-sig'))
    st=Study(**read(args.study));c=Candidate.from_dict(read(args.candidate))
    refiner=ACRefiner(st);c,r,meta=refiner.refine(c,rounds=args.rounds)
    fresh=Evaluator(st).evaluate(c)
    if fresh['status']!='feasible':raise RuntimeError('No feasible final candidate; do not rank financially')
    if abs(fresh['metrics']['selection_objective']-r['metrics']['selection_objective'])>1e-5:raise RuntimeError('Independent objective mismatch')
    meta['independent_AC_cases']=st.K*st.S
    for name,value in [('candidate',c.to_dict()),('validation',fresh),('counts',meta),('history',refiner.history)]:
        (out/(name+'.json')).write_text(json.dumps(plain(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(status=fresh['status'],FO=fresh['metrics']['selection_objective'],
        independent_AC_cases=fresh['expected_cases'],origin='auxiliary_AC_redispatch',cut_zero_required=False)))


if __name__=='__main__':main()
