"""Native signed generation changes whose net balance is supplied by AC slack.

An internal transfer conserves total non-slack P. This extra coordinate makes
net generation independently movable even with zero load shedding. No slack
setpoint, cost sorting, dispatch solver, or feasibility shortcut is introduced.
"""
import copy
import numpy as np
from . import physical
from .native_blocks import endpoint_value
from .consolidated_search import distribute


class SlackExchangeBlock:
    def __init__(self,inner):
        self.inner=inner;self.codec=inner.codec;self.base=inner.base
        self.reference=inner.reference.copy();self.fields=inner.fields
        self.mass=inner.mass;self.radius=inner.radius
        self.dimension=inner.dimension+len(inner.groups)
        self.categorical_indices=list(inner.categorical_indices)
        self.metadata=copy.deepcopy(inner.metadata)
        self.metadata.update(net_generation_exchange=True,slack='calculated by unchanged AC evaluator',
            net_generation_sign='both increase and decrease native; no preferred technology',
            auxiliary_optimizer=False)

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid slack exchange coordinates')
        self.inner.reference=self.reference.copy()
        full=self.inner.decode(z[:self.inner.dimension])
        signed=(2*endpoint_value(z[self.inner.dimension:],self.mass)-1)*self.radius
        if not np.any(signed):return full
        st=self.codec.study;c=self.codec.decode(full)[0]
        lo,hi,*_=physical.existing_bounds(st,c)
        capacity=st.active_capacity(c);availability=physical.availability(st)
        controls=z[self.inner.prefix:self.inner.dimension].reshape(-1,8)
        for group,v,d in zip(self.inner.groups,controls,signed):
            source=self.inner.sources[min(int(v[5]*len(self.inner.sources)),len(self.inner.sources)-1)]
            for k,s in group:
                p=np.r_[c.operation.existing_p[k,s],c.operation.new_p[k,s]]
                low=np.r_[lo[k,s],np.zeros(st.N)]
                high=np.r_[hi[k,s],capacity[k]*availability[s]]
                room=np.maximum(0,high[source]-p[source] if d>=0 else p[source]-low[source])
                p[source]=distribute(p[source],low[source],high[source],float(d*room.sum()))
                c.operation.existing_p[k,s]=p[:st.G];c.operation.new_p[k,s]=p[st.G:]
        # P has changed, so the old absolute Q can be outside the new envelope.
        # Encode only P; retain Q's normalized native coordinates. The unchanged
        # physical decoder then maps Q into the envelope for the proposed P.
        # This is exactly the previous output wherever its full encode succeeded.
        blocks=self.codec.unpack(full)
        for name,values,low,high in (
            ('existing_p',c.operation.existing_p,lo,hi),
            ('new_p',c.operation.new_p,0.,capacity[:,None,:]*availability[None,:,:])):
            low,high=np.broadcast_arrays(low,high)
            if np.any(values<low-1e-8) or np.any(values>high+1e-8):
                raise ValueError('Active power outside its representable bounds')
            blocks[name]=np.clip(np.divide(values-low,high-low,
                out=np.zeros_like(values),where=high>low),0,1)
        encoded=self.codec.pack(blocks)
        for name in ('existing_p','new_p'):
            ids=self.base.ids(name,self.inner.cases);full[ids]=encoded[ids]
        return full

    def project(self,full):
        self.inner.reference=self.reference.copy()
        # Projection is approximate and always re-evaluated by the runner.
        return np.r_[self.inner.project(full),np.full(len(self.inner.groups),.5)]

    def describe(self):
        result=self.inner.describe();result.update(self.metadata)
        result.update(dimension=self.dimension,extra_native_coordinates=len(self.inner.groups))
        return result
