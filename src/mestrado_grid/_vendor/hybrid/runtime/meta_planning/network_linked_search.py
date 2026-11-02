"""Native interbus dispatch/service coordinates; no LP or economic repair.

Engines choose four signed amplitudes. Existing independent controls and
investment domains remain reachable through the complementary scheduler.
The separate coordinated_ac refiner is never silently invoked here.
"""
import copy
import numpy as np
from .native_blocks import Field,endpoint_value,endpoint_key
from .energy_linked_search import EnergyLinkedScheduler
from . import physical
from .resource_construction import allocate


class NetworkTransferBlock:
    def __init__(self,codec,base,reference,case,mass=.25):
        self.codec=codec;self.base=base;self.reference=np.asarray(reference).copy()
        self.case=tuple(case);self.mass=mass;self.dimension=4
        names=('cut','existing_p','new_p','existing_q','phys_new_q')
        ids=np.unique(np.concatenate([base.ids(n,[self.case]) for n in names]))
        self.fields=[Field('interbus_dispatch_service',ids)]
        self.metadata=dict(scope='interbus_dispatch_service',case=list(case),case_groups=[[list(case)]],
            allocation='all load buses and all eligible sources in this state',
            auxiliary_optimizer=False,representation='native service amplitude, source mix, supply fraction and Q fraction; interbus balance linkage, full AC acceptance')

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(4,) or np.any(~np.isfinite(z)) or np.any((z<0)|(z>1)):raise ValueError('Invalid native coordinates')
        d=2*endpoint_value(z,self.mass)-1
        if np.all(d==0):return self.reference.copy()
        st=self.codec.study;k,s=self.case;c=self.codec.decode(self.reference)[0]
        anchor=copy.deepcopy(c);lo,hi,*_=physical.existing_bounds(st,c)
        def signed(a,low,high,x):return a+x*(high-a if x>=0 else a-low)
        # Restoration sign is positive; the opposite direction is still allowed.
        cut=c.operation.cut[k,:,s];load=c.operation.conventional[k,:,s]
        c.operation.cut[k,:,s]=signed(cut,0.,load,-d[0])
        upper=st.active_capacity(c)[k]*physical.availability(st)[s]
        # Positive generation compensates restored load anywhere in the graph.
        # Native source-mix weights alter the allocation, never the catalogue.
        restored=float((anchor.operation.cut[k,:,s]-c.operation.cut[k,:,s]).sum())
        p=np.r_[c.operation.existing_p[k,s],c.operation.new_p[k,s]]
        low=np.r_[lo[k,s],np.zeros(st.N)];high=np.r_[hi[k,s],upper]
        room=high-p if restored>=0 else p-low
        weights=np.r_[np.full(st.G,max(.01,1.-d[1])),np.full(st.N,max(.01,1.+d[1]))]
        amount=abs(restored)*float(.5+z[2]);direction=np.sign(restored)
        change=np.zeros_like(p)
        for _ in range(len(p)+1):
            free=np.maximum(0.,room-change);weighted=free*weights
            if amount<=1e-12 or weighted.sum()<=1e-12:break
            extra=np.minimum(free,amount*weighted/weighted.sum());change+=extra;amount-=extra.sum()
        p+=direction*change
        c.operation.existing_p[k,s]=p[:st.G];c.operation.new_p[k,s]=p[st.G:]
        ql,qh,nl,nh,_,_=physical.q_bounds(st,c)
        qnew=np.asarray(c.physical['new_q']);q=np.r_[c.operation.existing_q[k,s],qnew[k,s]]
        qlow=np.r_[ql[k,s],nl[k,s]];qhigh=np.r_[qh[k,s],nh[k,s]]
        requested=float((anchor.operation.cut[k,:,s]-c.operation.cut[k,:,s])@np.array(st.electrical['q_ratio'])[:,s])
        q=allocate(q,qlow,qhigh,float(q.sum()+requested*(.5+z[3])))
        c.operation.existing_q[k,s]=q[:st.G];qnew[k,s]=q[st.G:];c.physical['new_q']=qnew.tolist()
        result=self.reference.copy();encoded=self.codec.encode(c)
        result[self.fields[0].indices]=encoded[self.fields[0].indices]
        return result

    def project(self,full):
        # Preserve meaningful engine memory when an outside context changes.
        # Saturated directions are only approximate and are always re-evaluated.
        if np.array_equal(full,self.reference):return np.full(4,.5)
        st=self.codec.study;k,s=self.case
        a=self.codec.decode(self.reference)[0];b=self.codec.decode(full)[0]
        restored=float((a.operation.cut[k,:,s]-b.operation.cut[k,:,s]).sum())
        room=float((a.operation.cut[k,:,s] if restored>=0 else a.operation.conventional[k,:,s]-a.operation.cut[k,:,s]).sum())
        d0=np.clip(restored/max(room,1e-12),-1,1)
        dpold=float((b.operation.existing_p[k,s]-a.operation.existing_p[k,s]).sum())
        dpnew=float((b.operation.new_p[k,s]-a.operation.new_p[k,s]).sum())
        lo,hi,*_=physical.existing_bounds(st,a);upper=st.active_capacity(a)[k]*physical.availability(st)[s]
        ro=float(((hi[k,s]-a.operation.existing_p[k,s]) if restored>=0 else (a.operation.existing_p[k,s]-lo[k,s])).sum())
        rn=float(((upper-a.operation.new_p[k,s]) if restored>=0 else a.operation.new_p[k,s]).sum())
        t=abs(dpnew)*ro;u=abs(dpold)*rn;d1=(t-u)/(t+u) if t+u>1e-12 else 0.
        z2=np.clip((dpold+dpnew)/restored-.5,0,1) if abs(restored)>1e-12 else .5
        qr=float((a.operation.cut[k,:,s]-b.operation.cut[k,:,s])@np.array(st.electrical['q_ratio'])[:,s])
        dq=float((b.operation.existing_q[k,s]-a.operation.existing_q[k,s]).sum()+
                 (np.asarray(b.physical['new_q'])[k,s]-np.asarray(a.physical['new_q'])[k,s]).sum())
        z3=np.clip(dq/qr-.5,0,1) if abs(qr)>1e-12 else .5
        return np.r_[endpoint_key((d0+1)/2,self.mass),endpoint_key((d1+1)/2,self.mass),z2,z3]

    def describe(self):
        return dict(self.metadata,dimension=4,physical_coordinates=len(self.fields[0].indices),
                    fields=[dict(name=self.fields[0].name,kind='native_signed',coordinates=len(self.fields[0].indices))])


class NetworkLinkedScheduler(EnergyLinkedScheduler):
    def __init__(self,*args):super().__init__(*args);self.network_visits=0
    def choose(self,reference,contexts):
        self.network_visits+=1
        if self.network_visits%2:
            case=divmod(self.base.case_order[((self.network_visits-1)//2)%len(self.base.case_order)],self.codec.study.S)
            return None,NetworkTransferBlock(self.codec,self.base,reference,case,self.base.mass)
        return super().choose(reference,contexts)
    def report(self):
        out=super().report();out['network_visits']=self.network_visits
        out['native_transfer_note']='native interbus amplitudes; auxiliary economic refiner is separate'
        return out
