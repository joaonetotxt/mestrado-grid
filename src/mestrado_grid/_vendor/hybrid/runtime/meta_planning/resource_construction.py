"""Explicit, price-blind proposals using resource balances, not an AC optimizer.

Lossless subtree sums and first-order voltage drops are construction estimates.
Only the unchanged nonlinear evaluator can approve a candidate. No PF trials,
objective evaluations or iterative economic search occur in this module.
"""
import copy
import numpy as np
from .temporal import storage_dispatch
from .topology import line_schedule, orient_tree


def allocate(values, lower, upper, target):
    """One proportional headroom adjustment; a target outside bounds saturates."""
    out=np.clip(np.asarray(values,float),lower,upper).copy()
    delta=float(target-out.sum())
    spare=upper-out if delta>=0 else out-lower
    if spare.sum()>0:out+=np.sign(delta)*spare*min(1.,abs(delta)/spare.sum())
    return out


def resources(st,candidate):
    lo,hi,ql,qh=st.existing_bounds()
    from .physical import availability
    av=availability(st)
    upper=np.concatenate((hi,st.active_capacity(candidate)[:,None,:]*av[None,:,:]),axis=2)
    lower=np.concatenate((lo,np.zeros((st.K,st.S,st.N))),axis=2)
    buses=np.array([st.bi[g['bus']] for g in st.old+st.new],int)
    ch,di=storage_dispatch(st,candidate);fixed=candidate.operation.ev.copy()
    for j,u in enumerate(st.units):fixed[:,st.bi[u['bus']],:]+=ch[:,j,:]-di[:,j,:]
    return lower,upper,ql,qh,buses,fixed


def sub_limits(st,k,s):
    if 'physical' in st.electrical:
        from .physical import sub_limits as limits
        return limits(st,k,s)
    av=st.av[st.sub['id'],s];growth=st.c['generation_growth_by_stage'][k]
    hi=av['maximum']*growth;lo=max(0.,av['minimum']*growth)
    qlo,qhi=st.electrical['generator_q'][str(st.sub['id'])] if hi>1e-9 else (0.,0.)
    return lo,hi,qlo,qhi,st.c['substation_rated_MVA']*hi/st.c['substation_active_limit_MW']


def local_dispatch(demand,lower,upper,buses,target):
    gen=lower.copy()
    for b in range(len(demand)):
        ids=np.flatnonzero(buses==b)
        if len(ids):gen[ids]=allocate(gen[ids],lower[ids],upper[ids],max(0.,demand[b]))
    return allocate(gen,lower,upper,target)


def network_estimate(st,candidate,k,pnet,qnet,installed):
    """Return subtree apparent flows and approximate voltage, in catalogue order."""
    byid={e['id']:e for e in st.lines};direction=orient_tree(st.data,candidate.plan.trees[k])
    children={b:[] for b in st.buses}
    for eid,di in direction.items():
        e=byid[eid];a,b=(e['i'],e['j']) if di==0 else (e['j'],e['i'])
        children[a].append((b,eid))
    order=[st.data['root']];arcs=[]
    for a in order:
        for b,e in children[a]:order.append(b);arcs.append((st.bi[a],st.bi[b],e))
    total=pnet.astype(complex)+1j*qnet;flow={}
    for a,b,e in reversed(arcs):flow[e]=total[b];total[a]+=total[b]
    voltage=np.full(st.B,st.electrical['vslack']);limits=[];flows=[]
    for a,b,e in arcs:
        o=installed[e,k];raw=st.eopts[e,o];cat=byid[e]['options'][o]
        voltage[b]=voltage[a]-(raw['r']*flow[e].real+raw['x']*flow[e].imag)/st.electrical['vslack']
        # Current rating converted at the minimum permitted terminal voltage.
        vmin=min(st.electrical['vmin'][a],st.electrical['vmin'][b])
        limits.append(min(cat['capacity_MW'],cat['nominal_current_A']/st.electrical['Ibase_A']*vmin))
        flows.append(flow[e])
    return np.asarray(flows),np.asarray(limits),voltage,total[st.bi[st.data['root']]]


def circle_step(a,b,limit):
    """Largest segment fraction within circles, provided its start is inside."""
    a=np.asarray(a);d=np.asarray(b)-a;limit=np.asarray(limit)
    if np.any(abs(a)>limit+1e-10):return 0.
    aa=abs(d)**2;bb=2*np.real(np.conj(a)*d);cc=abs(a)**2-limit**2
    roots=np.divide(-bb+np.sqrt(np.maximum(0.,bb*bb-4*aa*cc)),2*aa,out=np.ones_like(aa,float),where=aa>1e-20)
    return float(np.clip(np.min(roots,initial=1.),0,1))


def resource_start(codec,rng):
    from .construction import physical_start
    base,meta=physical_start(codec,rng)
    return restore_resources(codec,base,rng,meta)


def restore_resources(codec,base,rng,base_metadata):
    """Construct additional service using a supplied plan, without changing assets.

    The caller's base can be the exact MAT tree. No evaluated historical plan is
    fetched and no physical/financial feasibility is certified by the estimate.
    """
    meta=base_metadata;out=copy.deepcopy(base);st=codec.study
    lower,upper,ql,qh,buses,fixed=resources(st,base);qr=np.asarray(st.electrical['q_ratio'])
    installed,_=line_schedule(st.data,base.plan)
    margin=float(rng.uniform(.75,.85));fractions=[]
    for k in range(st.K):
        for s in range(st.S):
            op=out.operation;old=base.operation;conv=op.conventional[k,:,s]
            pl,ph,qlo,qhi,ss=sub_limits(st,k,s)
            # Import targets leave a loss allowance; all distributed P/Q counts.
            pt=max(pl,min(.35*ph,.35*ss,10.))
            qt=max(qlo,min(.5*qhi,.35*ss))
            f=min(1.,(upper[k,s].sum()+margin*min(ph,ss)-fixed[k,:,s].sum())/max(1e-12,conv.sum()))
            tq=float(np.dot(conv,qr[:,s]))
            if tq>0:f=min(f,(np.maximum(qh[k,s],0).sum()+margin*min(qhi,ss))/tq)
            served=np.maximum(conv-old.cut[k,:,s],conv*np.clip(f,0,1))
            p=local_dispatch(served+fixed[k,:,s],lower[k,s],upper[k,s],buses,(served+fixed[k,:,s]).sum()-pt)
            q=local_dispatch(served*qr[:,s],ql[k,s],qh[k,s],buses[:st.G],float(np.dot(served,qr[:,s]))-qt)
            def net(served,p,q):
                pn=served+fixed[k,:,s]-np.bincount(buses,weights=p,minlength=st.B)
                qn=served*qr[:,s]-np.bincount(buses[:st.G],weights=q,minlength=st.B)
                return network_estimate(st,base,k,pn,qn,installed)
            a=net(conv-old.cut[k,:,s],np.r_[old.existing_p[k,s],old.new_p[k,s]],old.existing_q[k,s])
            b=net(served,p,q)
            fraction=circle_step(a[0],b[0],margin*a[1])
            fraction=min(fraction,circle_step(np.array([a[3]]),np.array([b[3]]),np.array([margin*ss])))
            # Affine voltage and Q bounds constrain the segment, without PF trials.
            vl=np.asarray(st.electrical['vmin'])+.005;vh=np.asarray(st.electrical['vmax'])-.002
            root=st.bi[st.data['root']];vl[root]=st.electrical['vmin'][root];vh[root]=st.electrical['vmax'][root]
            for start,end,low,high in ((a[2],b[2],vl,vh),
                    (np.array([a[3].imag]),np.array([b[3].imag]),np.array([qlo]),np.array([margin*qhi]))):
                if np.any(start<low-1e-10) or np.any(start>high+1e-10):fraction=0.;continue
                delta=end-start
                bound=np.where(delta>0,high-start,start-low)
                rat=np.divide(bound,abs(delta),out=np.ones_like(delta),where=abs(delta)>1e-15)
                fraction=min(fraction,float(np.clip(np.min(rat,initial=1.),0,1)))
            op.cut[k,:,s]=old.cut[k,:,s]+fraction*(conv-served-old.cut[k,:,s])
            op.existing_p[k,s]=old.existing_p[k,s]+fraction*(p[:st.G]-old.existing_p[k,s])
            op.new_p[k,s]=old.new_p[k,s]+fraction*(p[st.G:]-old.new_p[k,s])
            op.existing_q[k,s]=old.existing_q[k,s]+fraction*(q-old.existing_q[k,s]);fractions.append(fraction)
    return out,dict(rule='available_resources_lossless_network_segment_v1',base=meta,margin=margin,
        minimum_segment_fraction=min(fractions),mean_segment_fraction=float(np.mean(fractions)),
        scope='initial_population_only',AC_evaluations_in_construction=0,
        guarantee='approximate_construction_only_full_AC_required')
