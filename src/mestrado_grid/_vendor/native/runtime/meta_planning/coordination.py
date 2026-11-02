"""Declared common load/PQ proposal, outside the unchanged direct evaluator.

Half of eligible offspring may rebalance their own generation against their own
load, preserving the anchor's lossless SUB import. No price/PF/objective is read.
Other offspring keep the original full-domain proposal. Bounds can prevent the
balance; this never labels an unevaluated candidate feasible.
"""
import numpy as np
from .resource_construction import resources,allocate,sub_limits


def coordinate_load(codec,genome,anchor):
    st=codec.study;c,repairs=codec.decode(genome);ref,_=codec.decode(anchor)
    low,high,ql,qh,buses,fixed=resources(st,c)
    _,_,_,_,_,ref_fixed=resources(st,ref)
    qr=np.asarray(st.electrical['q_ratio']);unmetP=unmetQ=0.
    for k in range(st.K):
        for s in range(st.S):
            served=c.operation.conventional[k,:,s]-c.operation.cut[k,:,s]
            refserved=ref.operation.conventional[k,:,s]-ref.operation.cut[k,:,s]
            pl,ph,qlo,qhi,ss=sub_limits(st,k,s)
            # SUB residual is estimated from the anchor's actual controls, never
            # from an optimized historical dispatch or an AC correction trial.
            targetP=float((refserved+ref_fixed[k,:,s]).sum()-ref.operation.existing_p[k,s].sum()-ref.operation.new_p[k,s].sum())
            targetQ=float(np.dot(refserved,qr[:,s])-ref.operation.existing_q[k,s].sum())
            targetP=np.clip(targetP,pl,ph);targetQ=np.clip(targetQ,qlo,qhi)
            desiredP=float((served+fixed[k,:,s]).sum()-targetP)
            desiredQ=float(np.dot(served,qr[:,s])-targetQ)
            # Preserve the proposed distribution as far as one bounded
            # proportional adjustment allows; do not rank technologies by cost.
            p=allocate(np.r_[c.operation.existing_p[k,s],c.operation.new_p[k,s]],low[k,s],high[k,s],desiredP)
            q=allocate(c.operation.existing_q[k,s],ql[k,s],qh[k,s],desiredQ)
            c.operation.existing_p[k,s]=p[:st.G];c.operation.new_p[k,s]=p[st.G:];c.operation.existing_q[k,s]=q
            unmetP=max(unmetP,abs(float(p.sum()-desiredP)));unmetQ=max(unmetQ,abs(float(q.sum()-desiredQ)))
    encoded=codec.encode(c)
    return encoded,dict(rule='own_load_PQ_proportional_headroom_v1',anchor_id=ref.candidate_id,
        changed_coordinates=int(np.count_nonzero(abs(encoded-genome)>1e-12)),
        maximum_unmet_P_MW=unmetP,maximum_unmet_Q_MVAr=unmetQ,
        AC_evaluations=0,objective_evaluations=0,decode_repairs=repairs)
