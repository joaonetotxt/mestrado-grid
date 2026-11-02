"""Common mixed representation; physical arrays keep explicit stage/state axes."""
from dataclasses import dataclass
import numpy as np
from .schema import Candidate, Operation, Plan
from .topology import PlanCodec, line_schedule


@dataclass(frozen=True)
class Block:
    start: int
    stop: int
    shape: tuple


class Codec:
    def __init__(self, study):
        self.study = study
        self.plan_codec = PlanCodec(study.data)
        st = study
        shapes = dict(plan=(self.plan_codec.dimension,), new_capacity=(st.K,st.N),
                      existing_p=(st.K,st.S,st.G),existing_q=(st.K,st.S,st.G),
                      new_p=(st.K,st.S,st.N),soc=(st.K,st.U,st.S),
                      conventional=(st.K,st.B,st.S),ev=(st.K,st.B,st.S),cut=(st.K,st.B,st.S))
        if 'physical' in st.electrical:
            from .physical import shapes as physical_shapes
            shapes.update({'phys_'+n:s for n,s in physical_shapes(st).items()})
            self.storage_sites=sorted({u['bus'] for u in st.units})
            self.unit_site=np.array([self.storage_sites.index(u['bus']) for u in st.units],int)
            shapes['soc']=(st.K,len(self.storage_sites),st.S)
            shapes['phys_bess_q']=(st.K,st.S,len(self.storage_sites))
        self.indices={n:np.arange(int(np.prod(s))) for n,s in shapes.items()}
        if 'physical' in st.electrical:
            # Remove only coordinates whose domain is invariant for every plan.
            # Inactive but purchasable equipment retains its control coordinates.
            from .temporal import demand_bounds
            from .physical import choice_options
            lo,hi,ql,qh=st.existing_bounds();cl,ch,el,eh=demand_bounds(st)
            masks={'existing_p':hi>lo,'conventional':ch>cl,'ev':eh>el,'cut':ch>0}
            if not st.electrical['physical']['enabled']['retrofit']:masks['existing_q']=qh>ql
            av=np.array([[g['availability'][str(s)] for g in st.new] for s in range(st.S)])
            masks['new_p']=np.broadcast_to(av[None,:,:]>0,shapes['new_p'])
            flags=st.electrical['physical']['enabled']
            for n in physical_shapes(st):
                enabled=flags['new_q'] if n=='new_q' else flags['bess_q'] if n=='bess_q' else flags['capacitors'] if n=='cap_state' else flags['regulators'] if n=='tap' else len(choice_options(st,n))>1
                if not enabled:masks['phys_'+n]=np.zeros(shapes['phys_'+n],bool)
            for n,m in masks.items():self.indices[n]=np.flatnonzero(m)
        self.blocks = {}; offset = 0
        for name,shape in shapes.items():
            size = len(self.indices[name]); self.blocks[name] = Block(offset,offset+size,shape); offset += size
        self.dimension = offset

    def unpack(self, genome):
        x = np.asarray(genome,float)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)) or np.any((x<0)|(x>1)):
            raise ValueError('Genome must be finite, correctly sized and in [0,1]')
        out={}
        for name,b in self.blocks.items():
            a=np.zeros(int(np.prod(b.shape)));a[self.indices[name]]=x[b.start:b.stop];out[name]=a.reshape(b.shape)
        return out

    def pack(self, blocks):
        if set(blocks) != set(self.blocks):
            raise ValueError('Missing or unexpected genome blocks')
        for n,b in self.blocks.items():
            if np.asarray(blocks[n]).shape != b.shape:
                raise ValueError('Block shape mismatch: ' + n)
        x = np.concatenate([np.asarray(blocks[n]).ravel()[self.indices[n]] for n in self.blocks])
        self.unpack(x)
        return x

    def describe(self):
        return dict(dimension=self.dimension,axes='K=stage,S=state,G=existing non-SUB,N=new,U=BESS option,B=bus',
                    blocks={name:dict(start=b.start,stop=b.stop,shape=b.shape,size=b.stop-b.start) for name,b in self.blocks.items()})

    def search_groups(self):
        """Four complete disjoint coordinate groups, without freezing decisions."""
        names=(('plan','new_capacity'),('existing_p','existing_q','new_p','cut'),
               ('soc',),('conventional','ev'))
        if 'physical' in self.study.electrical:
            names=(names[0]+('phys_cap_choice','phys_reg_choice','phys_retro_choice','phys_bess_choice'),
                   names[1]+('phys_new_q','phys_cap_state','phys_tap'),names[2]+('phys_bess_q',),names[3])
        return [[[self.blocks[n].start,self.blocks[n].stop] for n in group
                 if self.blocks[n].stop>self.blocks[n].start] for group in names
                if any(self.blocks[n].stop>self.blocks[n].start for n in group)]

    def categorical_spec(self):
        """Search domains, not prices or a preferred investment schedule.

        Tree priority keys are ordinal rankings, not independent categories.
        Capacitor switching has plan-dependent bins; zero denotes a uniform
        key reset (its physical level is established only by the decoder).
        """
        p=self.plan_codec;offset=self.blocks['plan'].start;out=[]
        def add(name,start,sizes):
            if len(sizes):out.append(dict(name=name,start=start,sizes=[int(n) for n in sizes]))
        add('line_schedule',offset,[len(x) for x in p.line_choices])
        add('storage_purchase',offset+len(p.lines),[len(x) for x in p.storage_choices])
        if 'physical' in self.study.electrical:
            from .physical import choice_options
            for name in ('cap_choice','reg_choice','retro_choice','bess_choice'):
                b=self.blocks['phys_'+name]
                add(name,b.start,[len(choice_options(self.study,name))]*(b.stop-b.start))
            for name,size in [('tap',33),('cap_state',0)]:
                b=self.blocks['phys_'+name];add(name,b.start,[size]*(b.stop-b.start))
        return out

    def exploration_spec(self):
        """Remaining keys plus conditional Q links, without costs or AC trials."""
        categorical=self.categorical_spec();taken=np.zeros(self.dimension,bool)
        for row in categorical:taken[row['start']:row['start']+len(row['sizes'])]=True
        continuous=[]
        for name,b in self.blocks.items():
            ids=np.flatnonzero(~taken[b.start:b.stop])+b.start
            if not len(ids):continue
            # Only the plan block has a categorical prefix (tree priority suffix).
            if not np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):raise ValueError('Noncontiguous continuous domain')
            continuous.append(dict(name='tree_priority' if name=='plan' else name,start=int(ids[0]),sizes=[0]*len(ids)))
        links=[]
        if 'physical' in self.study.electrical:
            from .physical import choice_options
            for choice_name,target,sites in [('retro_choice','existing_q',range(self.study.G)),
                                             ('bess_choice','phys_bess_q',self.unit_site)]:
                choice=self.blocks['phys_'+choice_name];control=self.blocks[target]
                if choice.stop==choice.start or control.stop==control.start:continue
                inverse=np.full(int(np.prod(control.shape)),-1,dtype=int)
                inverse[self.indices[target]]=np.arange(control.start,control.stop)
                inverse=inverse.reshape(control.shape)
                stages=[None if o is None else o[1] for o in choice_options(self.study,choice_name)]
                for j,site in enumerate(sites):
                    links.append(dict(choice_index=choice.start+j,stages=stages,family=target,
                        by_stage=[[int(i) for i in inverse[k,:,site] if i>=0] for k in range(self.study.K)]))
        return dict(continuous=continuous,links=links)

    def decode(self, genome):
        from .temporal import normalize, demand_bounds
        st = self.study; blocks = self.unpack(genome)
        plan, repairs = self.plan_codec.decode(blocks['plan'])
        # A capped simplex by radial scaling: any admissible purchase vector is
        # unchanged; no automatic renewal after expiry and no repeated site cap.
        caps = np.array([g['max_MW'] for g in st.new])
        buys = blocks['new_capacity']*caps
        if st.data['investment_mode']=='single':
            buys[1:] = 0
        factor = np.maximum(1.,np.divide(buys.sum(axis=0),caps,out=np.zeros(st.N),where=caps>0))
        buys /= factor
        lo,hi,qlo,qhi = st.existing_bounds()
        clo,chi,elo,ehi = demand_bounds(st)
        soc_keys=blocks['soc'][:,self.unit_site,:] if 'physical' in st.electrical else blocks['soc']
        operation = Operation(lo+blocks['existing_p']*(hi-lo), qlo+blocks['existing_q']*(qhi-qlo),
                              np.zeros((st.K,st.S,st.N)),soc_keys*np.array([u['energy_MWh'] for u in st.units])[None,:,None],
                              clo+blocks['conventional']*(chi-clo),elo+blocks['ev']*(ehi-elo),np.zeros((st.K,st.B,st.S)))
        candidate = Candidate(plan,buys,operation)
        active = st.active_capacity(candidate)
        availability = np.array([[g['availability'][str(s)] for g in st.new] for s in range(st.S)]).reshape(st.S,st.N)
        operation.new_p = blocks['new_p']*active[:,None,:]*availability[None,:,:]
        repairs.update(normalize(st,candidate))
        operation.cut = blocks['cut']*operation.conventional
        if 'physical' in st.electrical:
            from .physical import decode
            expanded=dict(blocks);expanded['phys_bess_q']=blocks['phys_bess_q'][:,:,self.unit_site]
            decode(st,candidate,expanded)
        # Zero signed zeros for stable identities; impossible/inactive genes do
        # not create different physically evaluated candidates.
        for a in [buys,*vars(operation).values()]:
            a[a==0.] = 0.
        repairs['capacity_scaled_columns'] = int(np.count_nonzero(factor>1))
        st.shape_check(candidate)
        return candidate,repairs

    def encode(self, candidate):
        from .temporal import demand_bounds, violations
        st = self.study; st.shape_check(candidate); line_schedule(st.data,candidate.plan)
        if max(violations(st,candidate).values(), default=0) > 1e-8:
            raise ValueError('Cannot encode a temporally invalid physical candidate')
        def fraction(x,lo,hi):
            lo,hi = np.broadcast_arrays(lo,hi)
            lo = np.broadcast_to(lo,x.shape); hi = np.broadcast_to(hi,x.shape)
            if np.any(x < lo-1e-8) or np.any(x > hi+1e-8):
                raise ValueError('Physical control is outside its representable bounds')
            return np.clip(np.divide(x-lo,hi-lo,out=np.zeros(x.shape),where=hi>lo),0,1)
        lo,hi,qlo,qhi = st.existing_bounds(); clo,chi,elo,ehi = demand_bounds(st)
        op = candidate.operation; caps = np.array([g['max_MW'] for g in st.new])
        if np.any(candidate.new_capacity.sum(axis=0) > caps+1e-8) or (st.data['investment_mode']=='single' and np.any(candidate.new_capacity[1:]!=0)):
            raise ValueError('Capacity schedule outside the catalogue')
        from .physical import availability
        avail = availability(st)
        if 'physical' in st.electrical:
            from .physical import q_bounds,existing_bounds
            lo,hi,_,_=existing_bounds(st,candidate)
            qlo,qhi,*_=q_bounds(st,candidate)
        blocks=dict(plan=self.plan_codec.encode(candidate.plan),new_capacity=fraction(candidate.new_capacity,0.,caps),
            existing_p=fraction(op.existing_p,lo,hi),existing_q=fraction(op.existing_q,qlo,qhi),
            new_p=fraction(op.new_p,0.,st.active_capacity(candidate)[:,None,:]*avail[None,:,:]),
            soc=fraction(op.soc,0.,st.storage_active(candidate.plan)[:,:,None]*np.array([u['energy_MWh'] for u in st.units])[None,:,None]),
            conventional=fraction(op.conventional,clo,chi),ev=fraction(op.ev,elo,ehi),cut=fraction(op.cut,0.,op.conventional))
        if 'physical' in st.electrical:
            from .physical import encode,structural_violations
            # Switching frequency is an operational constraint, not a missing
            # coordinate. Preserve the proposed schedule for the AC evaluator
            # to mark violated; do not abort the whole search or repair it here.
            if structural_violations(st,candidate)['physical_installation']>0:raise ValueError('Unrepresentable physical controls')
            blocks.update(encode(st,candidate,fraction))
            # Mutually exclusive BESS types share site keys without losing any
            # admissible SOC or Q controls of the chosen equipment.
            soc=np.zeros(self.blocks['soc'].shape);q=np.full(self.blocks['phys_bess_q'].shape,.5)
            active=st.storage_active(candidate.plan)
            for j,site in enumerate(self.unit_site):
                soc[:,site,:]+=blocks['soc'][:,j,:]
                for k in range(st.K):
                    if active[k,j]:q[k,:,site]=blocks['phys_bess_q'][k,:,j]
            blocks['soc']=soc;blocks['phys_bess_q']=q
        return self.pack(blocks)
