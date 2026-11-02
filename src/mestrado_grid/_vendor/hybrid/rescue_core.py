"""Isolated search mechanisms. Never changes physical or financial acceptance."""
from campaign_support import *
from hybrid_core import FAMILIES, day_order, apply


def coordinates(r):
    v=r.get('normalized_violations') or {}
    return (sum(v.get(k,0.) for k in ('SAIDI','SAIFI')),
            sum(x for k,x in v.items() if k not in ('SAIDI','SAIFI')),
            (r.get('metrics') or {}).get('selection_objective',float('inf')))


class Portfolio:
    """Small Pareto archive of measured alternatives, independent of the incumbent."""
    def __init__(self,limit=8):self.limit=limit;self.entries={};self.uses={}
    def offer(self,c,r,origin):
        if r.get('failed_cases')!=0 or r.get('status') not in ('feasible','violated'):return
        assert c.candidate_id==r['candidate_id']
        self.entries[c.candidate_id]=(copy.deepcopy(c),copy.deepcopy(r),copy.deepcopy(origin))
        items=list(self.entries.items());front=[]
        for key,item in items:
            a=coordinates(item[1])
            if not any(all(x<=y for x,y in zip(coordinates(other[1]),a)) and
                       any(x<y for x,y in zip(coordinates(other[1]),a)) for k,other in items if k!=key):
                front.append((key,item))
        keep=[]
        for score in (lambda a:(a[0],a[1],a[2]),lambda a:(a[1],a[0],a[2]),
                      lambda a:(a[0]+a[1],a[2]),lambda a:(a[2],a[0]+a[1])):
            if front:
                k,_=min(front,key=lambda x:(score(coordinates(x[1][1])),x[0]))
                if k not in keep:keep.append(k)
        # Preserve different plans before filling with operation variants.
        ordered=sorted(front,key=lambda x:(sum(coordinates(x[1][1])[:2]),coordinates(x[1][1])[2],x[0]))
        plans={self.entries[k][0].plan_id for k in keep}
        for k,e in ordered:
            if len(keep)>=self.limit:break
            if k not in keep and e[0].plan_id not in plans:keep.append(k);plans.add(e[0].plan_id)
        for k,e in ordered:
            if len(keep)>=self.limit:break
            if k not in keep:keep.append(k)
        self.entries={k:self.entries[k] for k in keep[:self.limit]}
    def repair_source(self,incumbent):
        rel,other,_=coordinates(incumbent)
        choices=[e for e in self.entries.values() if coordinates(e[1])[0]<rel and coordinates(e[1])[1]>0]
        if not choices:return None
        e=min(choices,key=lambda e:(coordinates(e[1])[0],coordinates(e[1])[1],self.uses.get(e[0].candidate_id,0),coordinates(e[1])[2]))
        self.uses[e[0].candidate_id]=self.uses.get(e[0].candidate_id,0)+1
        return copy.deepcopy(e)
    def export(self):
        return dict(limit=self.limit,uses=self.uses,entries=[dict(candidate=c.to_dict(),record=r,origin=o) for c,r,o in self.entries.values()])
    @classmethod
    def restore(cls,x):
        obj=cls(x['limit']);obj.uses=x['uses']
        for e in x['entries']:
            c=Candidate.from_dict(e['candidate']);assert c.candidate_id==e['record']['candidate_id']
            obj.entries[c.candidate_id]=(c,e['record'],e['origin'])
        return obj


def critical_cycles(st,record):
    """Re-rank using the trial's own full AC cases, retaining complete daily cycles."""
    return day_order(st,record)


def default_policy(record,feedback):
    rel,other,_=coordinates(record);weights={f:1 for f in FAMILIES}
    if rel>0:weights.update(topology=6,cable=3)
    if other>0:weights.update(dispatch=5,capacitor=3,regulator=3,cable=3,device_controls=2)
    if rel==0 and other==0:weights.update(dispatch=5,planned_reduce=2,new_PV=2,new_WT=2,BESS_ESS1=2,BESS_ESS2=2)
    for f,row in feedback.items():
        if row.get('improving_calls',0)>0:weights[f]=min(6,weights[f]+1)
    return dict(family_weights=weights,repair_slots=3,radius=.125 if other>0 else .5,
                reason='Explicit deterministic constraint-aware allocation; own-arm measured feedback only.')


def choose_family(policy,spent,visit):
    # Coverage guard shared by both upgraded variants. All fourteen families remain eligible.
    if visit%4==3:return FAMILIES[(visit//4)%len(FAMILIES)],True
    weights=policy['family_weights']
    return min(FAMILIES,key=lambda f:(spent.get(f,0)/weights[f],-weights[f],FAMILIES.index(f))),False


def critical_proposal(st,c,r,family,seed,visit):
    """Finite same-candidate voltage interventions; original catalogue and AC gate."""
    if family not in ('capacitor','regulator','device_controls','cable'):return None,None
    bad=[x for x in r.get('cases',[]) if x.get('converged') and x.get('normalized_violations',{}).get('voltage',0)>0]
    if not bad:return None,None
    row=max(bad,key=lambda x:x['normalized_violations']['voltage']);k=row['stage']
    volts=np.hypot(row['voltage_real'],row['voltage_imag']);i=int(np.argmax(np.abs(volts-1.)))
    bus=row['bus_order'][i];low=volts[i]<1.;ph=st.electrical['physical'];z=copy.deepcopy(c)
    rng=np.random.default_rng(np.random.SeedSequence([seed,visit,718]))
    direction=orient_tree(st.data,c.plan.trees[k]);edges={e['id']:e for e in st.lines};parent={}
    for eid,di in direction.items():
        e=edges[eid];a,b=(e['i'],e['j']) if di==0 else (e['j'],e['i']);parent[b]=(a,eid)
    path=[];nodes=[bus];node=bus
    while node in parent:
        node,eid=parent[node];path.append(eid);nodes.append(node)
    d=dict(family=family,stage=k,scope='trial_critical_voltage',critical_bus=bus,critical_state=row['state_id'])
    if family=='capacitor':
        sites=[b for b in nodes if b in ph['capacitor_sites']]
        if not sites:return None,None
        j=ph['capacitor_sites'].index(sites[0]);d['j']=j
        if c.physical['cap_choice'][j]:
            option,start=physical.choice_options(st,'cap_choice')[c.physical['cap_choice'][j]]
            a=np.asarray(z.physical['cap_state']);a[k,:,j]=(ph['capacitors'][option]['steps'] if low else 0)*np.asarray(ph['capacitors'][option]['electrical_available']);z.physical['cap_state']=a.tolist()
        else:
            if not low:return None,None
            d['option']=int(rng.integers(len(ph['capacitors'])));z=apply(st,c,d,r)
    elif family=='regulator':
        choices=[e for e in path if e in ph['regulator_sites']]
        if not choices:return None,None
        # Revisit different locations along the actual failing path.
        eid=choices[(visit//4)%min(len(choices),4)];j=ph['regulator_sites'].index(eid);d.update(j=j,edge=eid)
        tap=(4 if low else -4)*(1 if visit%2==0 else -1)
        if c.physical['reg_choice'][j]:
            a=np.asarray(z.physical['tap']);a[k,:,j]=np.clip(a[k,:,j]+tap,-16,16);z.physical['tap']=a.tolist()
        else:d.update(option=0,tap=tap);z=apply(st,c,d,r)
    elif family=='device_controls':
        options=[e for e in path if e in ph['regulator_sites'] and c.physical['reg_choice'][ph['regulator_sites'].index(e)]]
        if not options:return None,None
        j=ph['regulator_sites'].index(options[0]);a=np.asarray(z.physical['tap']);a[k,:,j]=np.clip(a[k,:,j]+(2 if low else -2),-16,16);z.physical['tap']=a.tolist();d['j']=j
    else:
        if not path:return None,None
        eid=path[int(rng.integers(min(len(path),4)))];installed,_=line_schedule(st.data,c.plan)
        opts=[j for j in range(len(edges[eid]['options'])) if j!=installed[eid,k] and not(edges[eid]['existing'] and j==0)]
        if not opts:return None,None
        d.update(edge=eid,option=int(rng.choice(opts)));z=apply(st,c,d,r)
    normalize(st,z)
    return z,d
