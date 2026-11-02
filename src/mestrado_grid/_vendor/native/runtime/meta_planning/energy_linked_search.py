"""Declared native investment/service coordinates; no inner optimizer.

An engine selects capacity changes. A linked coordinate simultaneously changes
local generation and served load at the same bus, over every available state
during the asset lifetime. No prices, feasibility tests or search select these
setpoints. This is a problem-specific constructive representation, NOT the
unmodified native algorithm. Independent controls remain in other visits.
"""
import numpy as np
from .native_blocks import Field
from .guided_blocks import WindowBlock
from .productive_search import ProductiveScheduler
from . import physical


class EnergyLinkedBlock:
    def __init__(self,codec,base,reference,stage,mass=.25):
        self.codec=codec;self.base=base;self.reference=np.asarray(reference).copy()
        self.stage=stage;self.mass=mass
        st=codec.study
        groups=[]
        for technology in sorted({g['technology'] for g in st.new}):
            ids=np.array([base.maps['new_capacity'][stage,j] for j,g in enumerate(st.new)
                          if g['technology']==technology],int)
            groups.append(Field('new_capacity:'+technology,ids[ids>=0]))
        self.capacity=WindowBlock(reference,groups,{},mass,1.)
        self.dimension=self.capacity.dimension
        # The shared decoder may rescale a site's whole purchase schedule when
        # its lifetime capacity cap is reached. Preserve that exact schedule,
        # not only the newly edited stage (which would scale it a second time).
        controlled=np.unique(np.concatenate([base.ids('new_capacity'),
            base.ids('new_p'),base.ids('phys_new_q'),base.ids('cut')]))
        self.fields=[Field('native_capacity_with_local_energy_balance',controlled)]
        self.metadata=dict(scope='energy_linked',scheduler='energy_linked',stage=stage,
            allocation='all_catalogue_sites_and_technologies',
            representation='native capacity; bounded local P/service linkage and clipped Q compensation',
            auxiliary_optimizer=False,global_AC_acceptance=True)

    def project(self,full):
        self.capacity.reference=self.reference.copy()
        return self.capacity.project(full)

    def decode(self,z):
        self.capacity.reference=self.reference.copy()
        raw=self.capacity.decode(z)
        if np.array_equal(raw,self.reference):return raw
        st=self.codec.study;anchor=self.codec.decode(self.reference)[0]
        candidate=self.codec.decode(raw)[0]
        # Availability and lifetime are the unchanged catalogue definitions.
        av=physical.availability(st)
        delta=(st.active_capacity(candidate)-st.active_capacity(anchor))[:,None,:]*av[None,:,:]
        delta=np.maximum(delta,-anchor.operation.new_p)
        new_upper=st.active_capacity(candidate)[:,None,:]*av[None,:,:]
        delta=np.minimum(delta,new_upper-anchor.operation.new_p)
        # Simultaneous sources at one bus share room proportionally. Ordering
        # of sites or technologies cannot decide who receives the restored load.
        for bus in sorted({g['bus'] for g in st.new}):
            js=[j for j,g in enumerate(st.new) if g['bus']==bus];b=st.bi[bus]
            d=delta[:,:,js];positive=np.maximum(d,0.);negative=np.minimum(d,0.)
            room=anchor.operation.cut[:,b,:]-negative.sum(axis=2)
            scale=np.minimum(1.,np.divide(room,positive.sum(axis=2),out=np.ones_like(room),where=positive.sum(axis=2)>0))
            d=negative+positive*scale[:,:,None]
            # A reduction may restore shed demand only up to the original
            # conventional demand. Other changes are evaluated, never repaired.
            deficit=np.maximum(-d,0.).sum(axis=2)
            spare=anchor.operation.conventional[:,b,:]-anchor.operation.cut[:,b,:]+np.maximum(d,0.).sum(axis=2)
            ratio=np.minimum(1.,np.divide(spare,deficit,out=np.ones_like(spare),where=deficit>0))
            d=np.maximum(d,0.)+np.minimum(d,0.)*ratio[:,:,None]
            delta[:,:,js]=d
            candidate.operation.cut[:,b,:]=np.clip(anchor.operation.cut[:,b,:]-d.sum(axis=2),0.,anchor.operation.conventional[:,b,:])
        candidate.operation.new_p=np.clip(anchor.operation.new_p+delta,0.,new_upper)
        # Compensate the reactive demand restored at the same bus. These are
        # bound conversions, not AC feasibility repair. Envelope/auxiliary and
        # current constraints still go through the unchanged AC evaluator.
        _,_,ql,qh,_,_=physical.q_bounds(st,candidate)
        target=np.asarray(anchor.physical['new_q']).copy()
        for j,g in enumerate(st.new):
            b=st.bi[g['bus']]
            target[:,:,j]+=delta[:,:,j]*np.asarray(st.electrical['q_ratio'])[b][None,:]
        candidate.physical['new_q']=np.clip(target,ql,qh)
        encoded=self.codec.encode(candidate)
        # Preserve every unrelated key exactly, including dormant decisions.
        out=self.reference.copy();ids=self.fields[0].indices;out[ids]=encoded[ids]
        return out

    def describe(self):
        return dict(self.metadata,dimension=self.dimension,physical_coordinates=len(self.fields[0].indices),
            fields=[dict(name=f.name,kind=f.kind,coordinates=len(f.indices),endpoint=f.endpoint) for f in self.capacity.fields])


class EnergyLinkedScheduler(ProductiveScheduler):
    def __init__(self,*args):
        super().__init__(*args);self.link_visits=0

    def choose(self,reference,contexts):
        self.link_visits+=1
        if not self.coverage_preflight and self.codec.study.N and self.link_visits%2:
            stage=((self.link_visits-1)//2)%self.codec.study.K
            return None,EnergyLinkedBlock(self.codec,self.base,reference,stage,self.base.mass)
        return super().choose(reference,contexts)

    def report(self):
        out=super().report();out['linked_visits']=self.link_visits
        out['representation_note']='Native capacity values; deterministic local energy linkage. Guided native variant, not a pure unmodified engine or an auxiliary optimizer.'
        return out
