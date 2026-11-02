from common import *
from collections import defaultdict,Counter
from meta_planning.guided_blocks import cycle_path

def catalogue(st,c,r):
    """Every family retained, finite explicit geographically varied shortlist."""
    ph=st.electrical['physical'];rows=r['cases'];lower={b:2. for b in st.buses};upper={b:0. for b in st.buses};flows=defaultdict(float);losses=defaultdict(float)
    for row in rows:
        for b,v in zip(row['bus_order'],np.hypot(row['voltage_real'],row['voltage_imag'])):lower[b]=min(lower[b],v);upper[b]=max(upper[b],v)
        for e in row['branches']:
            flows[e['id']]=max(flows[e['id']],abs(e['P_from_MW'])+abs(e['Q_from_MVAr']))
            losses[e['id']]+=st.probabilities[row['state_id']]*max(0.,e['P_from_MW']+e['P_to_MW'])
    load={b:float(st.base_load[:,st.bi[b]].sum()) for b in st.buses};curt={b:0. for b in st.buses}
    for j,g in enumerate(st.old):
        if not g['renewable']:continue
        for k in range(st.K):
            for s in range(st.S):curt[g['bus']]+=st.probabilities[s]*max(0,st.av[g['id'],s]['maximum']-c.operation.existing_p[k,s,j])
    low=sorted((b for b in st.buses if b!=st.data['root']),key=lambda b:lower[b]);high=sorted(low,key=lambda b:-upper[b]);demand=sorted(low,key=lambda b:-load[b]);renew=sorted(low,key=lambda b:-curt[b])
    selected=list(dict.fromkeys([low[0],low[len(low)//2],high[0],demand[0],renew[0],renew[1]]))
    out=[]
    def add(family,**d):out.append(dict(family=family,**d))
    # Two storage technologies, geographically distinct sites, every entry year,
    # Q converter both off/on. One full cyclic pulse per day initializes SOC.
    for bus in list(dict.fromkeys([renew[0],demand[0],low[0]])):
        if any(next(u for u in st.units if u['id']==uid)['bus']==bus for uid,k in c.plan.storage_buys):continue
        for u in [u for u in st.units if u['bus']==bus]:
            for stage in range(st.K):
                for q in (False,True):add('BESS_'+u['id'].split('@')[0],unit=u['id'],stage=stage,q=q)
    for tech in ('WT','PV','BIO'):
        sites=[j for j,g in enumerate(st.new) if g['technology']==tech]
        orders=[sorted(sites,key=lambda j:lower[st.new[j]['bus']]),sorted(sites,key=lambda j:-load[st.new[j]['bus']]),sorted(sites,key=lambda j:-curt[st.new[j]['bus']])]
        shortlist=list(dict.fromkeys([j for a in orders for j in a[:2]]))
        for j in shortlist:
            available=st.new[j]['max_MW']-c.new_capacity[:,j].sum()
            for fraction in (.25,.75):
                if available<=1e-10:continue
                for stage in (0,2):add('new_'+tech,j=j,amount=available*fraction,stage=stage)
    for bus in list(dict.fromkeys([low[0],low[1],demand[0]])):
        j=ph['capacitor_sites'].index(bus)
        if c.physical['cap_choice'][j]:continue
        for opt in range(len(ph['capacitors'])):
            add('capacitor',j=j,option=opt,stage=0)
    edges=list(dict.fromkeys([e for e in sorted(flows,key=lambda e:-flows[e])[:2]]+[max(losses,key=losses.get)]))
    for eid in edges:
        j=ph['regulator_sites'].index(eid)
        if c.physical['reg_choice'][j]:continue
        for opt in range(len(ph['regulators'])):
            for tap in (-8,-4,4,8):add('regulator',j=j,option=opt,stage=0,tap=tap)
    installed,_=line_schedule(st.data,c.plan)
    cable_edges=list(dict.fromkeys(sorted(losses,key=lambda e:-losses[e])[:5]+sorted(flows,key=lambda e:-flows[e])[:5]))
    for eid in cable_edges:
        edge=next(e for e in st.lines if e['id']==eid)
        for stage in (0,1,2):
            for option in range(len(edge['options'])):
                if option==installed[eid,stage] or (edge['existing'] and option==0):continue
                add('cable',edge=eid,option=option,stage=stage)
    for tech in ('WT','PV','BIO'):
        gg=[j for j,g in enumerate(st.old) if g['technology']==tech]
        gg.sort(key=lambda j:lower[st.old[j]['bus']])
        for j in gg[:3]:
            if c.physical['retro_choice'][j]:continue
            for stage in (0,2):add('retrofit',j=j,stage=stage)
    # All available closing edges and their cycle opening choices are inspected
    # electrically later only for the explicit reliability-qualified shortlist.
    top=[];screened=0
    for stage in range(st.K):
        for edge in st.lines:
            if edge['id'] in c.plan.trees[stage]:continue
            choices=[]
            for drop in cycle_path(st.data,c.plan.trees[stage],edge['id']):
                z=ca.exchange(st,c,edge['id'],drop,stage);rr=reliability(st.data,z.plan);screened+=1
                if not all(x['passed'] for x in rr):continue
                score=sum(x['SAIDI']/st.c['SAIDI_limits'][k]+x['SAIFI']/st.c['SAIFI_limits'][k] for k,x in enumerate(rr))
                choices.append((score,drop))
            if choices:
                score,drop=min(choices);top.append((score,dict(family='reconfiguration' if edge['existing'] else 'new_route',close=edge['id'],drop=drop,stage=stage)))
    for family in ('reconfiguration','new_route'):
        out.extend(d for score,d in sorted(top,key=lambda x:x[0]) if d['family']==family) # trim below per family
    kept=[];nt=Counter()
    for d in out:
        if d['family'] in ('reconfiguration','new_route'):
            nt[d['family']]+=1
            if nt[d['family']]>8:continue
        kept.append(d)
    metadata=dict(sites_considered=st.buses,bus_voltage_min=lower,bus_voltage_max=upper,bus_load_score=load,bus_curtailment_score=curt,
        topology_reliability_screens=screened,shortlist_counts=dict(Counter(x['family'] for x in kept)),
        scope='All families free; finite physically ranked sites, sizes and stage choices. Not all site/technology combinations receive AC.')
    return kept,metadata

def apply(st,base,d,r):
    c=copy.deepcopy(base);f=d['family'];ph=st.electrical['physical'];stage=d.get('stage',0)
    if f.startswith('BESS'):
        j=next(j for j,u in enumerate(st.units) if u['id']==d['unit']);u=st.units[j]
        if any(next(x for x in st.units if x['id']==uid)['bus']==u['bus'] for uid,k in c.plan.storage_buys):raise ValueError('BESS site already occupied')
        c.plan.storage_buys.append((u['id'],stage))
        if d['q']:c.physical['bess_choice'][j]=physical.choice_options(st,'bess_choice').index((0,stage))
        for k in range(stage,st.K):
            costs={row['state_id']:sum(row['hourly_costs'].values()) for row in r['cases'] if row['stage']==k}
            for cy in st.cycles:
                donor=min(cy,key=lambda s:costs[s]);target=max(cy,key=lambda s:costs[s])
                if donor==target:continue
                energy=min(u['energy_MWh'],u['power_MW']*u['eta_charge']*st.hours[donor],u['power_MW']*st.hours[target]/u['eta_discharge'])
                a=(cy.index(donor)+1)%len(cy)
                while True:
                    s=cy[a];c.operation.soc[k,j,s]=energy
                    if s==target:break
                    a=(a+1)%len(cy)
    elif f.startswith('new_') and f!='new_route':
        j=d['j'];amount=min(d['amount'],st.new[j]['max_MW']-c.new_capacity[:,j].sum())
        if amount<=1e-10:raise ValueError('No remaining planned capacity')
        c.new_capacity[stage,j]+=amount;c.operation.new_p[stage:,:,j]+=.5*amount*physical.availability(st)[None,:,j]
    elif f=='capacitor':
        j=d['j'];o=ph['capacitors'][d['option']]
        if c.physical['cap_choice'][j]:raise ValueError('Capacitor already installed')
        c.physical['cap_choice'][j]=physical.choice_options(st,'cap_choice').index((d['option'],stage))
        cs=np.asarray(c.physical['cap_state']);cs[stage:,:,j]=np.array(o['electrical_available'])[None,:]*o['steps'];c.physical['cap_state']=cs.tolist()
    elif f=='regulator':
        j=d['j'];o=ph['regulators'][d['option']]
        if c.physical['reg_choice'][j]:raise ValueError('Regulator already installed')
        c.physical['reg_choice'][j]=physical.choice_options(st,'reg_choice').index((d['option'],stage));t=np.asarray(c.physical['tap'])
        for k in range(stage,st.K):
            if ph['regulator_sites'][j] in c.plan.trees[k]:t[k,:,j]=d['tap']*np.array(o['electrical_available'])
        c.physical['tap']=t.tolist()
    elif f=='retrofit':
        j=d['j'];c.physical['retro_choice'][j]=physical.choice_options(st,'retro_choice').index((0,stage))
    elif f=='cable':c=ca.replace_line(st,c,d['edge'],d['option'],stage)
    elif f in ('reconfiguration','new_route'):c=ca.exchange(st,c,d['close'],d['drop'],stage)
    else:raise ValueError(f)
    normalize(st,c);lo,hi,*_=physical.existing_bounds(st,c);c.operation.existing_p=np.clip(c.operation.existing_p,lo,hi)
    ql,qh,nl,nh,bl,bh=physical.q_bounds(st,c);c.operation.existing_q=np.clip(c.operation.existing_q,ql,qh)
    c.physical['new_q']=np.clip(c.physical['new_q'],nl,nh).tolist();c.physical['bess_q']=np.clip(c.physical['bess_q'],bl,bh).tolist()
    return c
