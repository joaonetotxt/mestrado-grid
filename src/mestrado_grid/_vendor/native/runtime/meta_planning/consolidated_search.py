"""Native balanced coordinates, including zero-cut redispatch and asset use.

This is a search representation, not an economic optimizer. Engine coordinates
choose signs, magnitudes, source groups, assets and SOC. No AC/LP call occurs in
decode. Original full AC evaluation and financial selection remain authoritative.
"""
import copy
import numpy as np
from . import physical
from .native_blocks import Block, Field, Scheduler, endpoint_value, endpoint_key
from .guided_blocks import TreeExchangeBlock
from .network_linked_search import NetworkLinkedScheduler
from .resource_construction import allocate
from .temporal import storage_dispatch


def distribute(p, low, high, amount, weights=None):
    """Bounded proportional headroom allocation, with no economic ranking."""
    weights = np.ones(len(p)) if weights is None else np.asarray(weights)
    direction = np.sign(amount)
    room = np.maximum(0., high-p if amount >= 0 else p-low)
    change = np.zeros_like(p); remaining = abs(amount)
    for _ in range(len(p)+1):
        free = np.maximum(0., room-change); w = free*weights
        if remaining <= 1e-12 or w.sum() <= 1e-12: break
        extra = np.minimum(free, remaining*w/w.sum())
        change += extra; remaining -= extra.sum()
    return p+direction*change


class CapacityPairBlock:
    """Native choice of two catalogue resources and signed purchases by stage.

    Existing plants are immutable. Reduction means cancellation/reduction of
    planned additions, not physical retirement. The engine chooses both resource
    indices and all changes. Duplicate selections apply only the first change.
    """
    def __init__(self,codec,base,reference,mass=.25,radius=1.):
        self.codec=codec; self.base=base; self.reference=np.asarray(reference).copy()
        self.mass=mass; self.radius=radius; self.dimension=2+2*codec.study.K
        self.fields=[Field('new_capacity',base.ids('new_capacity'))]
        self.metadata=dict(scope='native_capacity_pair',family='new_capacity',
            all_catalogue_resources_eligible=True,all_entry_stages_eligible=True,
            preserve_absolute_dispatch=True,duplicate_resource='second change ignored',
            auxiliary_optimizer=False)

    def decode(self,z):
        z=np.asarray(z,float); st=self.codec.study; out=self.reference.copy()
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid capacity coordinates')
        selected=[min(int(v*st.N),st.N-1) for v in z[:2]]
        for slot,j in enumerate(selected):
            if slot and j==selected[0]:continue
            ids=self.base.maps['new_capacity'][:,j]; valid=ids>=0; ids=ids[valid]
            d=(2*endpoint_value(z[2+slot*st.K:2+(slot+1)*st.K],self.mass)-1)*self.radius
            before=out[ids]; direction=d[valid]
            out[ids]=before+direction*np.where(direction>=0,1-before,before)
        return out

    def project(self,full):
        st=self.codec.study; out=np.full(self.dimension,.5)
        changed=[]
        for j in range(st.N):
            ids=self.base.maps['new_capacity'][:,j]; ids=ids[ids>=0]
            if np.any(np.asarray(full)[ids]!=self.reference[ids]):changed.append(j)
        for slot,j in enumerate((changed+[0,0])[:2]):
            out[slot]=(j+.5)/st.N
            ids=self.base.maps['new_capacity'][:,j]; valid=ids>=0; ids=ids[valid]
            before=self.reference[ids]; delta=np.asarray(full)[ids]-before
            room=np.where(delta>=0,1-before,before)*self.radius
            d=np.divide(delta,room,out=np.zeros_like(delta),where=room>0)
            out[2+slot*st.K:2+(slot+1)*st.K][valid]=endpoint_key((1+np.clip(d,-1,1))/2,self.mass)
        return out


class CoordinatedBlock:
    """Eight native coordinates per group of states plus optional asset keys.

    Coordinates: service, source mix, supply/Q compensation, signed balanced
    redispatch, two categorical source groups, and independent signed Q support.
    Source groups contain every existing/additional technology. Their order does
    not encode a preferred source, location or topology. Saturation never implies
    physical feasibility. Opposite directions and neutral proposals are legal.
    """
    def __init__(self, codec, base, reference, groups, mass=.25, radius=1., asset=None):
        if not 0 < radius <= 1: raise ValueError('Invalid radius')
        self.codec=codec; self.base=base; self.reference=np.asarray(reference).copy()
        self.groups=[list(map(tuple,g)) for g in groups]; self.mass=mass; self.radius=radius
        self.asset=asset; self.prefix=0 if asset is None else asset.dimension
        self.dimension=self.prefix+8*len(self.groups)
        self.cases=[c for g in self.groups for c in g]
        if len(set(self.cases))!=len(self.cases): raise ValueError('Repeated state')
        fields=[] if asset is None else asset.fields
        names=('cut','existing_p','new_p','existing_q','phys_new_q')
        ids=[f.indices for f in fields]+[base.ids(n,self.cases) for n in names]
        self.fields=[Field('coordinated_native',np.unique(np.concatenate(ids)))]
        st=codec.study
        labels=[('existing',g['technology']) for g in st.old]+[('additional',g['technology']) for g in st.new]
        self.sources=[np.array([i for i,v in enumerate(labels) if v==label],int) for label in sorted(set(labels))]
        self.metadata=dict(scope='coordinated_native',case_groups=self.groups,
            auxiliary_optimizer=False,asset=None if asset is None else asset.metadata,
            radius=radius,redispatch_without_load_change=True,
            balance='aggregate P linkage including charge/discharge; full AC still required')
        self.categorical_indices=[self.prefix+8*i+j for i in range(len(self.groups)) for j in (5,6)]
        if asset is not None:
            self.categorical_indices+=list(getattr(asset,'categorical_indices',[]))
        self.metadata['categorical_key_indices']=self.categorical_indices

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid native coordinates')
        full=self.reference.copy()
        if self.asset is not None:
            self.asset.reference=self.reference.copy(); full=self.asset.decode(z[:self.prefix])
        controls=z[self.prefix:].reshape(-1,8)
        if np.array_equal(full,self.reference) and np.all(controls[:,[0,4,7]]==.5): return full
        st=self.codec.study; anchor=self.codec.decode(self.reference)[0]
        c=self.codec.decode(full)[0]
        charge0,dis0=storage_dispatch(st,anchor); charge,dis=storage_dispatch(st,c)
        lo,hi,*_=physical.existing_bounds(st,c)
        availability=physical.availability(st); capacity=st.active_capacity(c)
        preserve_dispatch=self.asset is not None and self.asset.metadata.get('preserve_absolute_dispatch',False)
        if preserve_dispatch:
            # Capacity genes otherwise rescale utilization automatically. Keep
            # the actual MW where possible; compensate only a genuine deficit.
            c.operation.new_p=np.minimum(anchor.operation.new_p,capacity[:,None,:]*availability[None,:,:])
            c.physical['new_q']=np.asarray(anchor.physical['new_q']).copy()
        requested_q={}
        for group,v in zip(self.groups,controls):
            d=(2*endpoint_value(v,self.mass)-1)*self.radius
            for k,s in group:
                before=c.operation.cut[k,:,s].copy(); load=c.operation.conventional[k,:,s]
                c.operation.cut[k,:,s]=before-d[0]*(before if d[0]>=0 else load-before)
                # Include the asset prefix's own service coordinate (e.g. tree
                # exchange), not only the subsequent dispatch-coordinate change.
                restored=anchor.operation.cut[k,:,s]-c.operation.cut[k,:,s]
                p=np.r_[c.operation.existing_p[k,s],c.operation.new_p[k,s]]
                low=np.r_[lo[k,s],np.zeros(st.N)]; high=np.r_[hi[k,s],capacity[k]*availability[s]]
                weights=np.r_[np.full(st.G,max(.01,1-d[1])),np.full(st.N,max(.01,1+d[1]))]
                net_bess=float(((dis-charge)-(dis0-charge0))[k,:,s].sum())
                demand=float(restored.sum())-net_bess
                if preserve_dispatch:
                    demand-=float(p.sum()-anchor.operation.existing_p[k,s].sum()-anchor.operation.new_p[k,s].sum())
                p=distribute(p,low,high,demand*(.5+v[2]),weights)
                # Balanced transfer can operate with zero cut and no restoration.
                a=self.sources[min(int(v[5]*len(self.sources)),len(self.sources)-1)]
                b=self.sources[min(int(v[6]*len(self.sources)),len(self.sources)-1)]
                donor,receiver=(a,b) if d[4]>=0 else (b,a)
                if not np.array_equal(donor,receiver):
                    amount=abs(d[4])*min(np.maximum(0,p[donor]-low[donor]).sum(),np.maximum(0,high[receiver]-p[receiver]).sum())
                    p[donor]=distribute(p[donor],low[donor],high[donor],-amount)
                    p[receiver]=distribute(p[receiver],low[receiver],high[receiver],amount)
                c.operation.existing_p[k,s]=p[:st.G]; c.operation.new_p[k,s]=p[st.G:]
                requested_q[k,s]=float(restored@np.asarray(st.electrical['q_ratio'])[:,s])*(.5+v[3])
        ql,qh,nl,nh,_,_=physical.q_bounds(st,c); newq=np.asarray(c.physical['new_q'])
        for group,v in zip(self.groups,controls):
            d7=float(2*endpoint_value(v[7],self.mass)-1)*self.radius
            for k,s in group:
                q=np.r_[c.operation.existing_q[k,s],newq[k,s]]
                low=np.r_[ql[k,s],nl[k,s]]; high=np.r_[qh[k,s],nh[k,s]]
                q=allocate(q,low,high,float(q.sum()+requested_q[k,s]))
                q=distribute(q,low,high,d7*float((high-q if d7>=0 else q-low).sum()))
                c.operation.existing_q[k,s]=q[:st.G]; newq[k,s]=q[st.G:]
        c.physical['new_q']=newq.tolist(); encoded=self.codec.encode(c)
        # Preserve categorical asset keys. Re-encode only physical operation;
        # canonical decoding/repair is the existing codec's responsibility.
        for name in ('cut','existing_p','new_p','existing_q','phys_new_q'):
            ids=self.base.ids(name,self.cases); full[ids]=encoded[ids]
        return full

    def project(self,full):
        # Approximate projection is a proposal, never a fitness transfer. Native
        # context populations keep their own coordinates across revisits.
        out=np.full(self.dimension,.5)
        if self.asset is not None:
            self.asset.reference=self.reference.copy(); out[:self.prefix]=self.asset.project(full)
        st=self.codec.study; a=self.codec.decode(self.reference)[0]; b=self.codec.decode(full)[0]
        for i,g in enumerate(self.groups):
            delta=sum(float((a.operation.cut[k,:,s]-b.operation.cut[k,:,s]).sum()) for k,s in g)
            room=sum(float((a.operation.cut[k,:,s] if delta>=0 else a.operation.conventional[k,:,s]-a.operation.cut[k,:,s]).sum()) for k,s in g)
            d=np.clip(delta/max(room*self.radius,1e-12),-1,1)
            out[self.prefix+8*i]=endpoint_key((1+d)/2,self.mass)
        return out

    def describe(self):
        return dict(self.metadata,dimension=self.dimension,physical_coordinates=len(self.fields[0].indices),
            projection='approximate; always independently re-evaluated',source_groups=len(self.sources),
            fields=[dict(name='coordinated_native',kind='native_balanced',coordinates=len(self.fields[0].indices))])


class ConsolidatedScheduler(NetworkLinkedScheduler):
    """One new balanced visit alternating with the unchanged broad scheduler.

    New visits rotate: uniform state, economic state, capacity, cable, topology, storage.
    All asset sites rotate using seed decks, independent of economic scores.
    The economic visit chooses a state only, never its decision values. All
    previous families, direct genes and native context revisits remain reachable.
    """
    def __init__(self,*args):
        super().__init__(*args); self.consolidated_visits=0; self.new_visits=0
        self.protected=Scheduler(args[0],args[1]+4817,args[2])

    def choose(self,reference,contexts):
        self.consolidated_visits+=1
        if self.consolidated_visits%2==0: return super().choose(reference,contexts)
        st=self.codec.study; base=self.base; slot=self.new_visits%6
        radius=self.options.radii[(self.new_visits//6)%len(self.options.radii)]
        self.new_visits+=1; asset=None
        if slot in (0,1):
            uniform_case=divmod(self.protected.draw('consolidated_case',st.K*st.S),st.S)
            if slot==1 and self.options.economic_allocation and self.feedback and not self.coverage_preflight:
                case=max(self.feedback,key=lambda c:self.feedback[c].get('economic_opportunity',0)/(1+self.case_visits[c]))
            else: case=uniform_case
            groups=[[case]]
        elif slot==2 and st.N:
            asset=CapacityPairBlock(self.codec,base,reference,base.mass,radius)
            groups=[[(k,s) for s in range(st.S)] for k in range(st.K)]
        elif slot in (2,3):
            d=next(d for d in base.categories if d['name']=='line_schedule')
            j=self.protected.draw('consolidated_cable',len(d['sizes']))
            asset=Block(reference,[Field('line_schedule',np.array([d['start']+j]),'direct',False)],
                dict(scope='native_cable',site_index=j),base.mass)
            groups=[[(k,s) for s in range(st.S)] for k in range(st.K)]
        elif slot==4:
            k=self.protected.draw('consolidated_tree_stage',st.K)
            # Small existing operational prefix is native too; both parts are
            # evaluated jointly. A noop tree is included in the category set.
            fields=[Field('tree_service',base.ids('cut',[(k,0)]))]
            asset=TreeExchangeBlock(self.codec,reference,fields,dict(scope='native_tree',stage=k),base.mass,radius)
            groups=[[(k,s) for s in range(st.S)]]
        else:
            d=next(d for d in base.categories if d['name']=='storage_purchase')
            j=self.protected.draw('consolidated_storage_site',len(d['sizes']))
            site=self.codec.storage_sites.index(self.codec.plan_codec.sites[j])
            ids=base.maps['soc'][:,site,:].ravel(); ids=ids[ids>=0]
            # Every complete representative day, including no-cut/curtail-only
            # days. SOC power, losses and closure use the unchanged decoder.
            fields=[Field('storage_purchase',np.array([d['start']+j]),'direct',False),Field('soc',ids,'direct')]
            asset=Block(reference,fields,dict(scope='native_storage_cycle',site_index=j,
                complete_cycles=True,curtail_only_days_eligible=True),base.mass)
            groups=[[(k,s) for s in range(st.S)] for k in range(st.K)]
        block=CoordinatedBlock(self.codec,base,reference,groups,base.mass,radius,asset)
        block.metadata['allocation']='economic_state' if slot==1 and self.options.economic_allocation else 'protected_seed_deck'
        block.metadata['scheduler']='consolidated'
        if slot in (0,1) and not self.coverage_preflight:
            # Record the actual AC response at each visited single state. This
            # measurement alone does not select setpoints or add an optimizer.
            measurement=self.model(reference,case)
            block.metadata['measurement_anchor']=None if measurement is None else self.codec.decode(reference)[0].candidate_id
            block.metadata['measurement_role']='diagnostic response; no setpoint optimization'
        return None,block

    def report(self):
        out=super().report(); out.update(consolidated_visits=self.consolidated_visits,
            native_auxiliary_calls=0,storage_scope='all complete days at each eligible site; finite visited subset')
        return out
