"""Explicit local P/Q restoration proposals; no objective or flow inside moves.

The engine supplies the amplitude. Full AC and temporal evaluation determines
acceptance in runner. This is a declared common variant, not a native engine.
"""
import copy
import numpy as np
from .physical import availability, existing_bounds, q_bounds


def restore(st, reference, amplitude, affected, mode='renewable'):
    if isinstance(amplitude, bool) or not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError('Amplitude must be in [0,1]')
    if mode not in ('renewable', 'import'):raise ValueError('Unknown restoration mode')
    affected=list(affected)
    if len(set(affected))!=len(affected) or any(type(k)!=int or type(s)!=int or not 0<=k<st.K or not 0<=s<st.S for k,s in affected):
        raise ValueError('Invalid stage/state selection')
    c=copy.deepcopy(reference)
    if amplitude==0:return c
    _,hi,_,_=existing_bounds(st,c);cap=st.active_capacity(c);av=availability(st)
    for k,s in affected:
        if mode=='import':c.operation.cut[k,:,s]*=1-amplitude
        else:
            for origin,gens in [('old',st.old),('new',st.new)]:
                for j,g in enumerate(gens):
                    if g['technology'] not in ('PV','WT'):continue
                    b=st.bi[g['bus']]
                    a=c.operation.existing_p if origin=='old' else c.operation.new_p
                    maximum=hi[k,s,j] if origin=='old' else cap[k,j]*av[s,j]
                    delta=amplitude*max(0.,min(maximum-a[k,s,j],c.operation.cut[k,b,s]))
                    a[k,s,j]+=delta;c.operation.cut[k,b,s]-=delta
    # Recompute equipment envelopes AFTER P changed. First project Q into the
    # new envelope, then share the added local demand among same-bus sources.
    physical='physical' in st.electrical
    ol,oh,nl,nh,_,_=q_bounds(st,c) if physical else (*st.existing_bounds()[2:],None,None,None,None)
    newq=np.asarray(c.physical['new_q'],float) if physical else None
    for k,s in affected:
        oldq=c.operation.existing_q[k,s]
        np.clip(oldq,ol[k,s],oh[k,s],out=oldq)
        if physical:newq[k,s]=np.clip(newq[k,s],nl[k,s],nh[k,s])
        for b,bus in enumerate(st.buses):
            oj=[j for j,g in enumerate(st.old) if g['bus']==bus]
            nj=[j for j,g in enumerate(st.new) if g['bus']==bus] if physical else []
            extra=(reference.operation.cut[k,b,s]-c.operation.cut[k,b,s])*st.electrical['q_ratio'][b][s]
            before=sum(reference.operation.existing_q[k,s,j] for j in oj)
            if physical:before+=sum(reference.physical['new_q'][k][s][j] for j in nj)
            current=sum(oldq[j] for j in oj)+sum(newq[k,s,j] for j in nj)
            remaining=before+extra-current
            for arr,j,lower,upper in [(oldq,j,ol[k,s,j],oh[k,s,j]) for j in oj]+[(newq[k,s],j,nl[k,s,j],nh[k,s,j]) for j in nj]:
                delta=float(np.clip(remaining,lower-arr[j],upper-arr[j]));arr[j]+=delta;remaining-=delta
    if physical:c.physical['new_q']=newq.tolist()
    return c


def local_proposal(codec, genome, anchor, ordinal, step):
    if type(ordinal)!=int or ordinal<0:raise ValueError('Invalid ordinal')
    if isinstance(step,bool) or not np.isfinite(step) or not 0<step<=1:raise ValueError('Invalid step')
    st=codec.study;ref,_=codec.decode(anchor)
    raw=codec.unpack(genome);base=codec.unpack(anchor)
    amplitude=min(step,max(float(np.max(abs(raw[n]-base[n]),initial=0.)) for n in ('cut','existing_p','new_p')))
    # Both modes visit every stage/state, without a permanent top-four filter.
    index=ordinal//2;mode=('renewable','import')[ordinal%2]
    scores=ref.operation.cut.sum(axis=1)*st.probabilities[None,:]
    order=sorted(((k,s) for k in range(st.K) for s in range(st.S)),key=lambda ks:(-scores[ks],ks))
    k,s=order[index%len(order)]
    candidate=restore(st,ref,amplitude,[(k,s)],mode)
    x=np.asarray(anchor,float).copy() if amplitude==0 else codec.encode(candidate)
    return x,dict(rule='local_service_v1',move=mode,ordinal=ordinal,stage=k,state=s+1,
        amplitude=amplitude,restored_cut_MW=float(ref.operation.cut.sum()-candidate.operation.cut.sum()),
        AC_evaluations=0,objective_evaluations=0,role='explicit_search_operator_not_feasibility_repair')
