"""Unrestricted seeded catalogue revisits shared by both hybrid variants.

The pilot's physically ranked menu is retained for ordinary visits. Forced
visits also draw across all catalogue sites/types/stages, including changing
or cancelling planned devices. These proposals receive the same exact AC gate.
"""
from hybrid_core import *

def broad_proposal(st,c,r,family,seed,visit):
    rng=np.random.default_rng(np.random.SeedSequence([seed,visit,251019]))
    stage=int(rng.integers(st.K));ph=st.electrical['physical'];z=copy.deepcopy(c)
    d=dict(family=family,scope='unrestricted_catalogue_draw',stage=stage)
    if family=='topology':
        choices=[e for e in st.lines if e['id'] not in c.plan.trees[stage]]
        if not choices:return None,None
        e=choices[int(rng.integers(len(choices)))];path=cycle_path(st.data,c.plan.trees[stage],e['id'])
        d.update(close=e['id'],drop=path[int(rng.integers(len(path)))])
        return apply(st,c,d,r),d
    if family=='cable':
        if c.plan.line_buys and rng.random()<.25:
            purchase=c.plan.line_buys[int(rng.integers(len(c.plan.line_buys)))]
            z.plan.line_buys=[p for p in c.plan.line_buys if tuple(p)!=tuple(purchase)]
            line_schedule(st.data,z.plan)
            d.update(cancel_planned_purchase=list(purchase));return z,d
        installed,_=line_schedule(st.data,c.plan)
        choices=[e for e in st.lines if installed[e['id'],stage] is not None]
        e=choices[int(rng.integers(len(choices)))];opts=[i for i in range(len(e['options'])) if i!=installed[e['id'],stage] and not(e['existing'] and i==0)]
        if not opts:return None,None
        d.update(edge=e['id'],option=int(rng.choice(opts)));return apply(st,c,d,r),d
    if family in ('new_PV','new_WT','new_BIO'):
        ids=[j for j,g in enumerate(st.new) if g['technology']==family[4:] and g['max_MW']-c.new_capacity[:,j].sum()>1e-10]
        if not ids:return None,None
        j=int(rng.choice(ids));remaining=st.new[j]['max_MW']-c.new_capacity[:,j].sum()
        d.update(j=j,amount=remaining*float(rng.choice([.05,.25,.5,1.])))
        return apply(st,c,d,r),d
    if family in ('capacitor','regulator','retrofit'):
        choice={'capacitor':'cap_choice','regulator':'reg_choice','retrofit':'retro_choice'}[family]
        j=int(rng.integers(len(z.physical[choice])));options=physical.choice_options(st,choice)
        selected=int(rng.integers(len(options)));z.physical[choice][j]=selected
        d.update(j=j,choice_index=selected)
        if family in ('capacitor','regulator'):
            control='cap_state' if family=='capacitor' else 'tap';a=np.asarray(z.physical[control]);a[:,:,j]=0
            if selected:
                typ,start=options[selected];o=ph['capacitors' if family=='capacitor' else 'regulators'][typ]
                value=o['steps'] if family=='capacitor' else int(rng.integers(-16,17))
                for k in range(start,st.K):
                    if family=='capacitor' or ph['regulator_sites'][j] in c.plan.trees[k]:
                        a[k,:,j]=value*np.asarray(o['electrical_available'])
                d.update(option=typ,stage=start,control=value)
            z.physical[control]=a.tolist()
    elif family in ('BESS_ESS1','BESS_ESS2'):
        units=[u for u in st.units if u['id'].split('@')[0]==family[5:]]
        u=units[int(rng.integers(len(units)))];j=next(j for j,x in enumerate(st.units) if x['id']==u['id'])
        occupied=[uid for uid,k in z.plan.storage_buys if next(x for x in st.units if x['id']==uid)['bus']==u['bus']]
        for uid in occupied:z=apply(st,z,dict(family='storage_remove',unit=uid),r)
        d.update(unit=u['id'],q=bool(rng.integers(2)))
        z=apply(st,z,d,r)
        # Varied donor/receiver and depth give both cyclic directions exposure.
        z.operation.soc[:,j,:]=0
        for k in range(stage,st.K):
            for cy in st.cycles:
                donor,target=map(int,rng.choice(cy,2,replace=False))
                energy=min(u['energy_MWh'],u['power_MW']*u['eta_charge']*st.hours[donor],u['power_MW']*st.hours[target]/u['eta_discharge'])*float(rng.choice([.1,.25,.5,1.]))
                a=(cy.index(donor)+1)%len(cy)
                while True:
                    s=cy[a];z.operation.soc[k,j,s]=energy
                    if s==target:break
                    a=(a+1)%len(cy)
    elif family=='planned_reduce':
        ids=[j for j in range(st.N) if np.any(c.new_capacity[:,j])]
        if not ids:return None,None
        d.update(j=int(rng.choice(ids)),fraction=float(rng.choice([0.,.25,.5,.9])))
        return apply(st,c,d,r),d
    elif family=='storage_remove':
        if not c.plan.storage_buys:return None,None
        d['unit']=c.plan.storage_buys[int(rng.integers(len(c.plan.storage_buys)))][0]
        return apply(st,c,d,r),d
    elif family=='device_controls':
        installed=[(key,j) for key in ('cap_choice','reg_choice') for j,x in enumerate(c.physical[key]) if x]
        if not installed:return None,None
        key,j=installed[int(rng.integers(len(installed)))];typ,start=physical.choice_options(st,key)[c.physical[key][j]]
        k=int(rng.integers(start,st.K));cy=st.cycles[int(rng.integers(len(st.cycles)))];control='cap_state' if key=='cap_choice' else 'tap'
        a=np.asarray(z.physical[control]);o=ph['capacitors' if key=='cap_choice' else 'regulators'][typ]
        value=int(rng.integers(o['steps']+1)) if key=='cap_choice' else int(rng.integers(-16,17))
        if key=='reg_choice' and ph['regulator_sites'][j] not in c.plan.trees[k]:value=0
        a[k,cy,j]=value*np.asarray(o['electrical_available'])[cy];z.physical[control]=a.tolist()
        d.update(control=control,j=j,cycle=list(cy),stage=k,value=value)
    else:return None,None
    normalize(st,z)
    lo,hi,*_=physical.existing_bounds(st,z);z.operation.existing_p=np.clip(z.operation.existing_p,lo,hi)
    ql,qh,nl,nh,bl,bh=physical.q_bounds(st,z);z.operation.existing_q=np.clip(z.operation.existing_q,ql,qh)
    z.physical['new_q']=np.clip(z.physical['new_q'],nl,nh).tolist();z.physical['bess_q']=np.clip(z.physical['bess_q'],bl,bh).tolist()
    return z,d
