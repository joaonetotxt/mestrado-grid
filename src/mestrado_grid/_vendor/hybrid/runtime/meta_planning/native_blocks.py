"""Cooperative search representations, not a second proposal optimizer.

The scheduler chooses scopes without inspecting fitness/flows/prices. A local
Engine generates every offspring value. Coarse coordinates move a whole family;
direct coordinates keep every original decision individually reachable. No
fixed endpoint candidates or feasibility-driven splitting in native mode.
"""
from dataclasses import dataclass
import copy
import numpy as np

CASE=('existing_p','existing_q','new_p','phys_new_q','phys_bess_q','cut','phys_cap_state','phys_tap')
TEMPORAL=('soc','conventional','ev')

def endpoint_value(z,mass=.1):
    if not 0<=mass<.5:raise ValueError('Endpoint mass outside domain')
    z=np.asarray(z,float)
    return np.clip((z-mass)/(1-2*mass),0,1)

def endpoint_key(x,mass=.1):
    x=np.asarray(x,float)
    return np.where(x<=0,mass/2,np.where(x>=1,1-mass/2,mass+(1-2*mass)*x))

def relative_value(reference,value):
    step=2*np.asarray(value)-1
    return np.where(step<0,reference*(1+step),reference+(1-reference)*step)

@dataclass
class Field:
    name:str
    indices:np.ndarray
    kind:str='relative'
    endpoint:bool=True

class Block:
    def __init__(self,reference,fields,metadata,mass=.1):
        self.reference=np.asarray(reference,float).copy();self.fields=[f for f in fields if len(f.indices)]
        if not self.fields:raise ValueError('Empty block')
        allids=np.concatenate([f.indices for f in self.fields])
        if len(set(allids.tolist()))!=len(allids):raise ValueError('Overlapping local fields')
        self.metadata=copy.deepcopy(metadata);self.mass=mass;self.slices=[];offset=0
        for f in self.fields:
            size=1 if f.kind=='relative' else len(f.indices)
            self.slices.append(slice(offset,offset+size));offset+=size
        self.dimension=offset

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):raise ValueError('Invalid local genotype')
        out=self.reference.copy()
        for f,s in zip(self.fields,self.slices):
            value=endpoint_value(z[s],self.mass) if f.endpoint else z[s]
            out[f.indices]=relative_value(self.reference[f.indices],value[0]) if f.kind=='relative' else value
        return out

    def project(self,full):
        """Projection seeds are explicitly re-evaluated, never borrowed fitness."""
        full=np.asarray(full,float);out=np.empty(self.dimension)
        for f,s in zip(self.fields,self.slices):
            value=full[f.indices]
            if f.kind=='relative':
                anchor=self.reference[f.indices];delta=value-anchor
                # Orthogonal projection onto each half of the bounded ray.
                candidates=[]
                for denominator,sign in ((anchor,-1),(1-anchor,1)):
                    norm=float(denominator@denominator)
                    magnitude=0. if norm==0 else np.clip(float(delta@(sign*denominator))/norm,0,1)
                    approximation=anchor+sign*magnitude*denominator
                    candidates.append((float(np.sum((approximation-value)**2)),(1+sign*magnitude)/2))
                value=np.array([min(candidates)[1]])
            out[s]=endpoint_key(value,self.mass) if f.endpoint else value
        return out

    def describe(self):
        return dict(**self.metadata,dimension=self.dimension,physical_coordinates=sum(len(f.indices) for f in self.fields),
            endpoint_mass=self.mass,fields=[dict(name=f.name,kind=f.kind,coordinates=len(f.indices),endpoint=f.endpoint) for f in self.fields])

class Scheduler:
    def __init__(self,codec,seed,mass=.1):
        self.codec=codec;self.study=codec.study;self.mass=mass
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,7727]));self.count=0
        self.decks={};self.resolution_visits={};self.maps={}
        for name,b in codec.blocks.items():
            a=np.full(int(np.prod(b.shape)),-1,int);a[codec.indices[name]]=np.arange(b.start,b.stop)
            self.maps[name]=a.reshape(b.shape)
        self.case_names=[n for n in CASE if np.any(self.maps[n]>=0)]
        self.case_order=self.rng.permutation(self.study.K*self.study.S).tolist()
        self.categories=[d for d in codec.categorical_spec() if d['name'] not in ('tap','cap_state')]

    def draw(self,name,count):
        if not self.decks.get(name):self.decks[name]=self.rng.permutation(count).tolist()
        return self.decks[name].pop()

    def ids(self,name,cases=None):
        a=self.maps[name]
        if cases is None:return a[a>=0].copy()
        chunks=[a[k,:,s].ravel() if name=='cut' else a[k,s,:].ravel() for k,s in cases]
        out=np.concatenate(chunks);return out[out>=0]

    def partitions(self,key):
        visit=self.resolution_visits.get(key,0);self.resolution_visits[key]=visit+1
        sizes=[1,2,4,8,16,self.study.K*self.study.S]
        size=sizes[self.draw('resolution',len(sizes))]
        return [[divmod(int(i),self.study.S) for i in part] for part in np.array_split(self.case_order,size)]

    def technology_ids(self,name,cases,technology):
        catalogue=self.study.old if name=='existing_p' else self.study.new
        columns=[j for j,g in enumerate(catalogue) if g['technology']==technology]
        a=self.maps[name];ids=np.concatenate([a[k,s,columns].ravel() for k,s in cases])
        return ids[ids>=0]

    def next(self,reference):
        slot=self.count%8;number=self.count;self.count+=1;fields=[]
        meta=dict(number=number,scope='',scheduler='balanced independent of objective and violations')
        if slot in (0,2,4):
            name=self.case_names[self.draw('case_family',len(self.case_names))];parts=self.partitions(name)
            fields=[Field(name,self.ids(name,cases)) for cases in parts]
            meta.update(scope='state_family',family=name,case_groups=parts)
        elif slot==1:
            parts=self.partitions('joint');names=['existing_p','new_p','cut']
            # Rotate Q participation; no technology or critical-state list.
            if (number//8)%2:names+=['existing_q','phys_new_q','phys_bess_q']
            for cases in parts:
                for name in names:
                    if name in ('existing_p','new_p'):
                        catalogue=self.study.old if name=='existing_p' else self.study.new
                        for technology in sorted({g['technology'] for g in catalogue}):
                            fields.append(Field(name+':'+technology,self.technology_ids(name,cases,technology)))
                    else:fields.append(Field(name,self.ids(name,cases)))
            meta.update(scope='state_joint',case_groups=parts,families=names)
        elif slot==3:
            name=self.case_names[self.draw('direct_family',len(self.case_names))]
            k,s=divmod(self.draw('direct_case',self.study.K*self.study.S),self.study.S)
            fields=[Field(name,self.ids(name,[(k,s)]),'direct')]
            meta.update(scope='state_direct',family=name,cases=[(k,s)])
        elif slot==5:
            name=TEMPORAL[self.draw('temporal_family',len(TEMPORAL))]
            k,cycle=divmod(self.draw('temporal_cycle',self.study.K*len(self.study.cycles)),len(self.study.cycles))
            ids=self.maps[name][k][:,self.study.cycles[cycle]].ravel();ids=ids[ids>=0]
            # Native direct values over a whole cycle; unchanged conservation
            # projection is part of the existing decoder.
            fields=[Field(name,ids,'direct')];meta.update(scope='temporal_cycle',family=name,stage=k,cycle=cycle)
        elif slot==6:
            fields,details=self.investment();meta.update(scope='investment_joint',**details)
        else:
            p=self.codec.plan_codec;k=self.draw('tree_stage',self.study.K)
            start=p.tree_offset+k*len(p.lines);ids=np.arange(start,start+len(p.lines))
            fields=[Field('tree_priority',ids,'direct',False)]
            cycle=self.coupled_operation(fields,[k])
            meta.update(scope='topology',stage=k,coupled_cycle=cycle)
        return Block(reference,fields,meta,self.mass)

    def coupled_operation(self, fields, stages=None):
        """Join enabling decisions to their possible operating benefit.

        The cycle and all magnitudes are seed/native controlled, never inferred
        from prices or AC violations. Subtract existing fields to keep one gene
        in one coordinate. Complete cycles preserve the storage representation.
        """
        st=self.study
        cycle=self.draw('coupled_cycle',len(st.cycles))
        stages=range(st.K) if stages is None else stages
        used=set(int(i) for f in fields for i in f.indices)
        for k in stages:
            cases=[(k,s) for s in st.cycles[cycle]]
            for family in ('existing_p','existing_q','new_p','phys_new_q','cut'):
                if family in ('existing_p','new_p'):
                    catalogue=st.old if family=='existing_p' else st.new
                    groups=[(family+':'+tech,self.technology_ids(family,cases,tech))
                            for tech in sorted({g['technology'] for g in catalogue})]
                else:groups=[(family,self.ids(family,cases))]
                for name,ids in groups:
                    ids=np.array([int(i) for i in ids if int(i) not in used],int)
                    if len(ids):
                        fields.append(Field(name+':stage'+str(k),ids))
                        used.update(ids.tolist())
        return cycle

    def investment(self):
        kind=self.draw('investment_family',len(self.categories)+1);fields=[];st=self.study
        if kind==len(self.categories):
            j=self.draw('new_capacity_site',st.N)
            ids=self.maps['new_capacity'][:,j];fields.append(Field('new_capacity',ids[ids>=0],'direct'))
            for name in ('new_p','phys_new_q'):
                ids=self.maps[name][:,:,j].ravel();fields.append(Field(name,ids[ids>=0]))
            # The enabling asset can operate in every state, but its benefit
            # must not force the same cut movement in all 96 cases. The cycle
            # fields below provide separate service coordinates by stage.
            cycle=self.coupled_operation(fields)
            return fields,dict(family='new_capacity',site_index=j,coupled_cycle=cycle)
        d=self.categories[kind];j=self.draw('investment_'+d['name'],len(d['sizes']))
        fields.append(Field(d['name'],np.array([d['start']+j]),'direct',False))
        name=d['name']
        if name=='storage_purchase':
            site=self.codec.plan_codec.sites[j];site_index=self.codec.storage_sites.index(site)
            for family in ('soc','phys_bess_q'):
                a=self.maps[family];ids=(a[:,site_index,:] if family=='soc' else a[:,:,site_index]).ravel()
                fields.append(Field(family,ids[ids>=0],'direct' if family=='soc' else 'relative'))
        elif name in ('cap_choice','reg_choice','retro_choice','bess_choice'):
            family={'cap_choice':'phys_cap_state','reg_choice':'phys_tap','retro_choice':'existing_q','bess_choice':'phys_bess_q'}[name]
            site_index=int(self.codec.unit_site[j]) if name=='bess_choice' else j
            a=self.maps[family];ids=a[:,:,site_index].ravel();fields.append(Field(family,ids[ids>=0]))
            if name=='bess_choice':
                ids=self.maps['soc'][:,site_index,:].ravel();fields.append(Field('soc',ids[ids>=0],'direct'))
        elif name=='line_schedule':
            p=self.codec.plan_codec
            ids=np.array([p.tree_offset+k*len(p.lines)+j for k in range(st.K)])
            fields.append(Field('tree_priority',ids,'direct',False))
        cycle=self.coupled_operation(fields)
        return fields,dict(family=name,site_index=j,coupled_cycle=cycle)

    def state(self):
        return dict(count=self.count,decks=copy.deepcopy(self.decks),resolution_visits=copy.deepcopy(self.resolution_visits),
                    case_order=self.case_order.copy(),rng=copy.deepcopy(self.rng.bit_generator.state),mass=self.mass)

    @classmethod
    def restore(cls,codec,state):
        obj=cls(codec,0,state['mass']);obj.count=state['count'];obj.decks=copy.deepcopy(state['decks'])
        obj.resolution_visits=copy.deepcopy(state['resolution_visits']);obj.case_order=state['case_order'].copy()
        obj.rng.bit_generator.state=copy.deepcopy(state['rng']);return obj
