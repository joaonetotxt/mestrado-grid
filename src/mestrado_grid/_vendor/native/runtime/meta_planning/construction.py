"""Feasibility-oriented catalogue heuristic, used ONLY for the common start.

No historical vector, price ranking, AC search, LP or OPF is consulted. Exact
dominance avoids paying for an identical, costlier existing-line replacement.
Directed
shortest paths prioritize reliability; local generation and a proportional reserve
rule balance a deliberately conservative served load. This is not AC feasibility
certification and cannot prove infeasibility if its candidate fails.
"""
import heapq
import numpy as np
from .schema import Plan


def initial_line_option(st,line):
    """Retain an installed line when replacing it has no modeled benefit.

    This is exact dominance under the current catalogue, not an economic search
    or a decoder restriction. All options remain representable by the engines.
    Any physical/reliability difference prevents this simplification.
    """
    base,upgrade=line['options'][:2]
    same=all(base[k]==upgrade[k] for k in ('capacity_MW','nominal_current_A','failure_rate','repair_hours'))
    same= same and all(st.eopts[line['id'],0][k]==st.eopts[line['id'],1][k] for k in ('r','x','b'))
    dominated=(upgrade['maintenance_per_year']>=base['maintenance_per_year'] and
               (upgrade['capex']>0 or upgrade['maintenance_per_year']>base['maintenance_per_year']))
    return 0 if line['existing'] and same and dominated else 1


def reliability_tree(st,rng):
    adjacency={b:[] for b in st.buses}
    choices={e['id']:initial_line_option(st,e) for e in st.lines}
    for e in st.lines:
        o=e['options'][choices[e['id']]]
        for direction,(i,j) in enumerate(((e['i'],e['j']),(e['j'],e['i']))):
            f=o['failure_rate'][direction];r=o['repair_hours'][direction]
            weight=f/max(1.,st.c['SAIFI_limits'][-1])+f*r/max(1.,st.c['SAIDI_limits'][-1])
            # Strictly positive costs avoid zero-cost predecessor cycles.
            adjacency[i].append((j,e['id'],max(1e-10,weight)*(1+rng.uniform(0,.015))))
    root=st.data['root'];dist={root:0.};parent={};queue=[(0.,root)]
    while queue:
        value,i=heapq.heappop(queue)
        if value!=dist[i]:continue
        for j,e,w in adjacency[i]:
            candidate=value+w
            if candidate<dist.get(j,float('inf')):
                dist[j]=candidate;parent[j]=e;heapq.heappush(queue,(candidate,j))
    if len(dist)!=st.B:raise ValueError('Catalogue is disconnected')
    tree=sorted(parent.values())
    return Plan([tree.copy() for _ in range(st.K)],[(e,choices[e],0) for e in tree if choices[e]!=0],[])


def original_mat_plan(st):
    """Exact imported operational tree; never infer closed switches from existing.

    A malformed/missing tree is an error, not permission to construct another.
    New investments are absent here and remain eligible in the search codec.
    """
    from .topology import orient_tree, line_schedule
    if any(type(e.get('initial_closed')) is not bool for e in st.lines):
        raise ValueError('MAT initialization requires explicit initial_closed on every line')
    tree=sorted(e['id'] for e in st.lines if e['initial_closed'])
    if any(not e['existing'] for e in st.lines if e['initial_closed']):
        raise ValueError('MAT initial tree contains an uninstalled line')
    orient_tree(st.data,tree)
    plan=Plan([tree.copy() for _ in range(st.K)],[],[])
    line_schedule(st.data,plan)
    return plan


def original_mat_start(codec,rng):
    from .resource_construction import restore_resources
    base,meta=physical_start(codec,rng,original_mat=True)
    return restore_resources(codec,base,rng,meta)


def physical_start(codec,rng,original_mat=False):
    st=codec.study;plan=original_mat_plan(st) if original_mat else reliability_tree(st,rng)
    # Small cyclic SOC variations keep usable variation for archive methods.
    # Their actual charge/discharge is included in the active balance below.
    if not original_mat:
        for choices in codec.plan_codec.storage_choices:
            choice=choices[int(rng.integers(len(choices)))]
            plan.storage_buys.extend(choice)
    blocks=codec.unpack(np.full(codec.dimension,.5))
    if 'physical' in st.electrical:
        for name in blocks:
            if name.startswith('phys_'):blocks[name][:]=0.5 if name.endswith('_q') else 0.
    blocks['plan']=codec.plan_codec.encode(plan)
    blocks['new_capacity'][:]=0.
    if not original_mat:blocks['new_capacity'][0]=rng.uniform(.65,.95,st.N)
    blocks['soc'][:]=.5+.002*(rng.random(blocks['soc'].shape)-.5)
    from .temporal import demand_bounds,balance_profile,storage_dispatch
    clo,chi,elo,ehi=demand_bounds(st)
    ev_profile=st.base_ev.copy()
    for cy in st.cycles:
        target=np.sum(st.base_ev[:,:,cy]*st.hours[cy],axis=-1)
        uniform=np.broadcast_to((target/24.)[:,:,None],ev_profile[:,:,cy].shape)*(1+.02*(rng.random(ev_profile[:,:,cy].shape)-.5))
        ev_profile[:,:,cy]=balance_profile(uniform,st.hours[cy],elo[:,:,cy],ehi[:,:,cy],target)
    for name,base,low,high in [('conventional',st.base_load,clo,chi),('ev',ev_profile,elo,ehi)]:
        if name=='conventional':base=base*(1+.004*(rng.random(base.shape)-.5))
        blocks[name]=np.divide(base-low,high-low,out=np.zeros(base.shape),where=high>low)
        blocks[name]=np.clip(blocks[name],0,1)
    candidate,_=codec.decode(codec.pack(blocks));op=candidate.operation
    if candidate.physical is not None:
        for name in ('new_q','bess_q'):candidate.physical[name]=np.zeros_like(candidate.physical[name],dtype=float).tolist()
    lo,hi,qlo,qhi=st.existing_bounds();active=st.active_capacity(candidate)
    from .physical import availability,sub_limits
    avail=availability(st)
    gen_buses=np.array([st.bi[g['bus']] for g in st.old+st.new],int)
    qratio=np.asarray(st.electrical['q_ratio'])
    margin=float(rng.uniform(.35,.55))
    active_margin=float(rng.uniform(.25,.35))
    charge,discharge=storage_dispatch(st,candidate)
    for k in range(st.K):
        for s in range(st.S):
            sublo,subhi,_,qmax,subS=sub_limits(st,k,s)
            subq=max(0.,qmax)
            # Q of new resources remains zero. Ignore potential old Q support in
            # this conservative load rule; use it locally below when available.
            qbudget=min(subq*margin,subS*.22)
            totalq=float(np.dot(op.conventional[k,:,s],qratio[:,s]))
            serve=min(.45,qbudget/totalq) if totalq>0 else .45
            totalp=float(op.conventional[k,:,s].sum())
            if totalp>0:serve=min(serve,active_margin*subS/totalp)
            op.cut[k,:,s]=op.conventional[k,:,s]*(1-serve)
            demand=op.conventional[k,:,s]-op.cut[k,:,s]+op.ev[k,:,s]
            for j,u in enumerate(st.units):demand[st.bi[u['bus']]]+=charge[k,j,s]-discharge[k,j,s]
            lower=np.r_[lo[k,s],np.zeros(st.N)]
            upper=np.r_[hi[k,s],active[k]*avail[s]]
            gen=lower.copy()
            for b in range(st.B):
                ids=np.flatnonzero(gen_buses==b);spare=upper[ids]-lower[ids]
                need=max(0.,demand[b]-lower[ids].sum())
                if spare.sum()>0:gen[ids]+=spare*min(1.,need/spare.sum())
            # Additional generation covers non-local EV/served demand while
            # reserving positive SUB import for unmodelled losses.
            target=max(sublo,min(subhi*.15,subS*.15,3.))
            desired=max(0.,demand.sum()-target)
            if gen.sum()<desired:
                spare=upper-gen
                if spare.sum()>0:gen+=spare*min(1.,(desired-gen.sum())/spare.sum())
            elif gen.sum()>desired:
                reducible=gen-lower
                if reducible.sum()>0:gen-=reducible*min(1.,(gen.sum()-desired)/reducible.sum())
            op.existing_p[k,s]=gen[:st.G];op.new_p[k,s]=gen[st.G:]
            op.existing_q[k,s]=qlo[k,s]
            for b in range(st.B):
                ids=np.flatnonzero(gen_buses[:st.G]==b);spare=qhi[k,s,ids]-qlo[k,s,ids]
                demandq=(op.conventional[k,b,s]-op.cut[k,b,s])*qratio[b,s]
                need=max(0.,demandq-qlo[k,s,ids].sum())
                if spare.sum()>0:op.existing_q[k,s,ids]+=spare*min(1.,need/spare.sum())
    if original_mat:
        assert candidate.plan.canonical()==plan.canonical()
        assert not np.any(candidate.new_capacity)
        assert candidate.physical is None or all(not np.any(v) for name,v in candidate.physical.items() if name.endswith('_choice'))
    return candidate,dict(rule=('original_MAT_tree_existing_assets_local_PQ_reserve' if original_mat else 'directed_reliability_tree_local_PQ_reserve_v3_exact_dominance'),reactive_margin=margin,active_margin=active_margin,
                         scope='initial_population_only',AC_evaluations_in_construction=0,
                         guarantee='structural_temporal_only_full_AC_required')
