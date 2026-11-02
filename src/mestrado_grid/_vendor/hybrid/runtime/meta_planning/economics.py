"""Direct operator ledger and fractional-tail CVaR. No optimization variables."""
import numpy as np
from . import contracts
from .topology import line_schedule


def weights(st):
    end=st.c['horizon_years'];r=st.c['discount_rate'];ys=st.years
    return np.array([sum((1+r)**-y for y in range(start+1,(ys[k+1] if k+1<len(ys) else end)+1)) for k,start in enumerate(ys)])


def tail(costs,probabilities,alpha):
    costs=np.asarray(costs,float);p=np.asarray(probabilities,float)
    if costs.shape!=p.shape or not np.all(np.isfinite(costs)) or not np.all(np.isfinite(p)) or np.any(p<0) or abs(p.sum()-1)>1e-10 or not 0<alpha<1:
        raise ValueError('Invalid CVaR inputs')
    remaining=1-alpha;rows=[];total=0.
    for s in sorted(range(len(p)),key=lambda s:(-costs[s],s)):
        take=min(remaining,p[s])
        if take>0:
            contribution=take*costs[s]/(1-alpha);total+=contribution
            rows.append(dict(state_id=s,probability=float(p[s]),tail_mass=float(take),cost_VP=float(costs[s]),contribution_VP=float(contribution)))
            remaining-=take
        if remaining<1e-14:break
    return float(total),rows


def deterministic(st,candidate):
    w=weights(st);r=st.c['discount_rate'];lines={e['id']:e for e in st.lines};units={u['id']:u for u in st.units}
    installed,_=line_schedule(st.data,candidate.plan)
    parts=dict(CAPEX_network=0.,maintenance_network=0.,CAPEX_BESS=0.,maintenance_BESS=0.,CAPEX_new=0.,maintenance_new=0.)
    for e,o,k in candidate.plan.line_buys:
        parts['CAPEX_network']+=lines[e]['options'][o]['capex']*(1+r)**-st.years[k]
    for (e,k),o in installed.items():
        if o is not None:parts['maintenance_network']+=lines[e]['options'][o]['maintenance_per_year']*w[k]
    for u,k in candidate.plan.storage_buys:
        parts['CAPEX_BESS']+=units[u]['capex']*(1+r)**-st.years[k]
        parts['maintenance_BESS']+=units[u]['maintenance_per_year']*sum(w[k:])
    active=st.active_capacity(candidate)
    for n,g in enumerate(st.new):
        parts['CAPEX_new']+=sum(candidate.new_capacity[k,n]*g['capex_per_MW']*(1+r)**-year for k,year in enumerate(st.years))
        parts['maintenance_new']+=sum(active[k,n]*g['fixed_per_MW_year']*w[k] for k in range(st.K))
    if 'physical' in st.electrical:
        from .physical import ledger
        parts.update(ledger(st,candidate)['operator_parts_VP'])
    return parts


def hourly(st,candidate,k,s,sub_p,loss,for_power,charge,discharge):
    op=candidate.operation;parts={}
    def add(name,value):parts[name]=parts.get(name,0.)+float(value)
    from .physical import existing_connected
    connected=dict(zip((g['id'] for g in st.old),existing_connected(st,candidate,k)[s]))
    for g,power in [(st.sub,sub_p),*zip(st.old,op.existing_p[k,s])]:
        av=st.av[g['id'],s];tech=g['technology'];maximum=av['maximum']*st.c['generation_growth_by_stage'][k]
        if not connected.get(g['id'],True):maximum=0.
        add('generation_existing_'+tech,power*g['price'])
        add('carbon_existing_'+tech,power*av['emissions']*st.c['carbon_price_per_ton'])
        if g['renewable']:add('curtailment_existing_'+tech,(maximum-power)*g['curtail_price'])
    active=st.active_capacity(candidate)[k]
    from .physical import availability
    available=availability(st)[s]
    for n,g in enumerate(st.new):
        power=op.new_p[k,s,n];maximum=active[n]*available[n];tech=g['technology']
        add('energy_new_'+tech,power*g['variable_per_MWh'])
        add('carbon_new_'+tech,power*g['emissions_ton_per_MWh']*st.c['carbon_price_per_ton'])
        if tech in ('PV','WT'):add('curtailment_new_'+tech,(maximum-power)*g['curtail_per_MWh'])
    add('BESS_discharge',sum(discharge[k,:,s])*st.c['prices_per_MWh']['ESS'])
    add('BESS_carbon',sum(discharge[k,:,s])*st.c['emissions_ton_per_MWh']['ESS']*st.c['carbon_price_per_ton'])
    flex=st.data['extension'].get('flex',{})
    add('DR_compensation',sum(np.maximum(st.base_load[k,:,s]-op.conventional[k,:,s],0))*flex.get('shift_compensation_per_MWh',0))
    add('EV_compensation',sum(np.maximum(st.base_ev[k,:,s]-op.ev[k,:,s],0))*flex.get('ev_compensation_per_MWh',0))
    add('conventional_cut',sum(op.cut[k,:,s])*st.c['unserved_energy_per_MWh'])
    add('AC_losses',loss*st.c['loss_cost_per_MWh'])
    add('FOR_proxy',for_power*st.c['EENS_cost_per_MWh'])
    return parts


def portfolio_vp30(st,candidate,cases):
    if len(cases)!=st.K*st.S or any(not r['converged'] for r in cases):
        raise ValueError('Incomplete numerical portfolio cannot have an aggregate cost')
    w=weights(st);p=st.probabilities;H=st.c['annual_hours'];costs=np.zeros(st.S)
    parts=deterministic(st,candidate);fixed=sum(parts.values());annual=[]
    by_case={(r['stage'],r['state_id']):r for r in cases}
    if set(by_case)!={(k,s) for k in range(st.K) for s in range(st.S)}:
        raise ValueError('Duplicate or missing portfolio case')
    for k,year in enumerate(st.years):
        energies=dict(year=year,cut_MWh_year=0.,loss_MWh_year=0.,load_MWh_year=0.,EV_MWh_year=0.)
        for s in range(st.S):
            row=by_case[k,s];costs[s]+=H*w[k]*sum(row['hourly_costs'].values())
            for name,value in row['hourly_costs'].items():parts[name]=parts.get(name,0.)+H*w[k]*p[s]*value
            energies['cut_MWh_year']+=H*p[s]*sum(candidate.operation.cut[k,:,s])
            energies['loss_MWh_year']+=H*p[s]*row['loss_MW']
            energies['load_MWh_year']+=H*p[s]*sum(candidate.operation.conventional[k,:,s]+candidate.operation.ev[k,:,s])
            energies['EV_MWh_year']+=H*p[s]*sum(candidate.operation.ev[k,:,s])
        annual.append(energies)
    cv,rows=tail(costs,p,st.c['alpha']);expected_op=float(p@costs)
    # Existing pure contract ledger provides producer/cohort attribution. The
    # operator cost above is independently assembled, never added a second time.
    values={f'dg_buy[{g["id"]},{k}]':float(candidate.new_capacity[k,n]) for n,g in enumerate(st.new) for k in range(st.K)}
    active=st.active_capacity(candidate)
    from .physical import availability
    av=availability(st)
    dispatch={(g['id'],k,s):(float(candidate.operation.new_p[k,s,n]),float(active[k,n]*av[s,n]-candidate.operation.new_p[k,s,n])) for n,g in enumerate(st.new) for k in range(st.K) for s in range(st.S)}
    ledger_data=st.data
    if 'physical' in st.electrical:
        import copy
        ledger_data=copy.deepcopy(st.data)
        for g in ledger_data['extension']['generators']:
            n=next(n for n,r in enumerate(st.new) if r['id']==g['id'])
            g['availability']={str(s):float(av[s,n]) for s in range(st.S)}
    ledger=contracts.payments(ledger_data,values,dispatch)
    paid=sum(H*w[k]*p[s]*(dispatch[g['id'],k,s][0]*g['variable_per_MWh']+dispatch[g['id'],k,s][1]*g['curtail_per_MWh'])
             for g in st.new if 'contract' in g for k in range(st.K) for s in range(st.S))
    if abs(paid-ledger['expected_payments_VP'])>1e-4:
        raise ValueError('Operator/producer transfer reconciliation failed')
    result=dict(deterministic_VP=fixed,expected_operation_VP=expected_op,expected_VP=fixed+expected_op,
                CVaR_operation_VP=cv,CVaR_total_VP=fixed+cv,
                objective_VP=fixed+(1-st.c['beta'])*expected_op+st.c['beta']*cv,
                cut_equivalent_MWh=sum(w[k]*annual[k]['cut_MWh_year'] for k in range(st.K))/sum(w),
                conditional_costs_VP=costs.tolist(),tail=rows,annual=annual,parts_VP=parts,
                contracts=ledger,accounting_residual_VP=float(sum(parts.values())-fixed-expected_op))
    if 'physical' in st.electrical:
        from .physical import ledger as physical_ledger
        result['physical_ledger']=physical_ledger(st,candidate)
        result['physical_ledger']['producer_parts_VP']['Q_auxiliary_energy']=float(sum(H*w[k]*p[s]*by_case[k,s]['physical']['auxiliary_reimbursement_per_hour'] for k in range(st.K) for s in range(st.S)))
    return result


def portfolio(st,candidate,cases):
    """Select the configured financial view and retain the unchanged VP30 ledger."""
    from .financial_policy import apply
    return apply(st,candidate,cases,portfolio_vp30(st,candidate,cases))
