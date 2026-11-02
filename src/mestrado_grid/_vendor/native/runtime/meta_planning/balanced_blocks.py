"""Neutral case coverage and selection of distinct, existing physical seeds.

Only the native Engine creates offspring. Neither helper uses costs, cuts,
voltages, feasibility or the diagnostic witness to choose a scope/direction.
"""
import copy, hashlib
import numpy as np

from .native_blocks import Scheduler, Block, Field, TEMPORAL


class BalancedScheduler(Scheduler):
    """Three single-case joint sessions, then one rotating complementary scope.

With 48 cases, the first 64 sessions visit each joint case exactly once.
Complementary scopes retain temporal, direct, investment, topology and coarse
operation moves. Case order depends solely on the seed, never performance.
"""
    COMPLEMENTS=('state_family','state_direct','temporal_cycle',
                 'investment_joint','topology','coarse_joint')

    def __init__(self,codec,seed,mass=.25):
        super().__init__(codec,seed,mass)
        self.joint_visits=0;self.complement_visits=0

    def joint_fields(self,cases):
        fields=[]
        for name in ('existing_p','new_p','cut','existing_q','phys_new_q','phys_bess_q'):
            if name in ('existing_p','new_p'):
                catalogue=self.study.old if name=='existing_p' else self.study.new
                for tech in sorted({g['technology'] for g in catalogue}):
                    fields.append(Field(name+':'+tech,self.technology_ids(name,cases,tech)))
            else:fields.append(Field(name,self.ids(name,cases)))
        return fields

    def next(self,reference):
        number=self.count;self.count+=1
        meta=dict(number=number,scheduler='seed-only balanced joint-case coverage')
        if number%4!=3:
            index=self.case_order[self.joint_visits%len(self.case_order)]
            k,s=divmod(index,self.study.S);self.joint_visits+=1
            fields=self.joint_fields([(k,s)])
            meta.update(scope='state_joint',case_groups=[[(k,s)]],balanced_case=True,
                        coverage_cycle=(self.joint_visits-1)//len(self.case_order),
                        families=['existing_p','new_p','cut','existing_q','phys_new_q','phys_bess_q'])
        else:
            scope=self.COMPLEMENTS[self.complement_visits%len(self.COMPLEMENTS)]
            self.complement_visits+=1;meta['scope']=scope
            if scope=='state_family':
                name=self.case_names[self.draw('case_family',len(self.case_names))]
                parts=self.partitions(name);fields=[Field(name,self.ids(name,cases)) for cases in parts]
                meta.update(family=name,case_groups=parts)
            elif scope=='state_direct':
                name=self.case_names[self.draw('direct_family',len(self.case_names))]
                k,s=divmod(self.draw('direct_case',self.study.K*self.study.S),self.study.S)
                fields=[Field(name,self.ids(name,[(k,s)]),'direct')]
                meta.update(family=name,cases=[(k,s)])
            elif scope=='temporal_cycle':
                name=TEMPORAL[self.draw('temporal_family',len(TEMPORAL))]
                k,cycle=divmod(self.draw('temporal_cycle',self.study.K*len(self.study.cycles)),len(self.study.cycles))
                ids=self.maps[name][k][:,self.study.cycles[cycle]].ravel()
                fields=[Field(name,ids[ids>=0],'direct')];meta.update(family=name,stage=k,cycle=cycle)
            elif scope=='investment_joint':
                fields,details=self.investment();meta.update(details)
            elif scope=='topology':
                p=self.codec.plan_codec;k=self.draw('tree_stage',self.study.K)
                start=p.tree_offset+k*len(p.lines)
                fields=[Field('tree_priority',np.arange(start,start+len(p.lines)),'direct',False)]
                cycle=self.coupled_operation(fields,[k])
                meta.update(stage=k,coupled_cycle=cycle)
            else:
                parts=self.partitions('joint');fields=[]
                for cases in parts:fields.extend(self.joint_fields(cases))
                meta.update(case_groups=parts,families=['existing_p','new_p','cut','existing_q','phys_new_q','phys_bess_q'])
        return Block(reference,fields,meta,self.mass)

    def state(self):
        return dict(**super().state(),joint_visits=self.joint_visits,complement_visits=self.complement_visits)

    @classmethod
    def restore(cls,codec,state):
        obj=cls(codec,0,state['mass'])
        for name in ('count','decks','resolution_visits','case_order','joint_visits','complement_visits'):
            setattr(obj,name,copy.deepcopy(state[name]))
        obj.rng.bit_generator.state=copy.deepcopy(state['rng']);return obj


def physical_key(candidate,tolerance=1e-10):
    """Diversity identity only; never a fitness cache key.

    Quantization avoids treating round-off as useful physical diversity.
    Exact candidate_id remains the evaluation/cache/persistence identity.
    """
    h=hashlib.sha256(candidate.plan_id.encode())
    items=[('new_capacity',candidate.new_capacity)]
    items += sorted(vars(candidate.operation).items())
    items += [('physical:'+name,candidate.physical[name]) for name in ('new_q','bess_q','cap_state','tap')]
    for name,values in items:
        arr=np.asarray(values,dtype=float)
        if not np.isfinite(arr).all():raise ValueError('Nonfinite physical values')
        h.update(name.encode());h.update(str(arr.shape).encode())
        h.update(np.rint(arr/tolerance).astype('<i8').tobytes())
    return h.hexdigest()


def diverse_projection(block,codec,sources,population_size):
    """Keep anchor, then distinct projected physical candidates from sources.

    No invented vectors, AC probes or direction selection. Exhausting existing
    diversity falls back to existing repeated vectors and is explicitly logged.
    """
    selected=[];fallback=[];keys=set();checked=0;duplicates=0
    for source in sources:
        local=block.project(source)
        candidate,_=codec.decode(block.decode(local));identity=physical_key(candidate)
        checked+=1
        if identity not in keys:
            selected.append(local);keys.add(identity)
            if len(selected)==population_size:break
        else:
            duplicates+=1;fallback.append(local)
    unique=len(selected)
    if not selected:raise ValueError('Missing context anchor')
    repeats=fallback or [selected[0]];i=0
    while len(selected)<population_size:
        selected.append(repeats[i%len(repeats)].copy());i+=1
    return np.asarray(selected),dict(physical_unique=unique,checked=checked,
        duplicates_skipped=duplicates,repeated_fallback=population_size-unique,
        source='existing native bank and fixed initial population only')
