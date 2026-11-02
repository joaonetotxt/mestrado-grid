"""Isolated opt-in early native resource coverage, with unchanged evaluation.

One seeded pass of enabling decisions and their operating controls precedes
the protected scheduler, in both feasibility tiers. No candidate is scored to
choose a task/site, and no installation is forced. A completed visit requires
actual native offspring. Opportunity, decoded change and acceptance differ.
"""
import numpy as np
from .risk_native_search import RiskNativeScheduler
from .native_blocks import Field
from .guided_blocks import WindowBlock
from .consolidated_search import CoordinatedBlock


def resource_tasks(codec):
    st=codec.study
    cats={d['name']:d for d in codec.categorical_spec()}
    tasks=[]
    for name in ('line_schedule','storage_purchase','cap_choice','reg_choice','retro_choice'):
        if name in cats and any(n>1 for n in cats[name]['sizes']):tasks.append(name)
    tasks+=['new_capacity:'+t for t in sorted({g['technology'] for g in st.new})]
    tasks+=['existing_PQ:'+t for t in sorted({g['technology'] for g in st.old})]
    tasks+=['topology','service']
    tasks += [name for name in ('conventional','ev') if codec.blocks[name].stop>codec.blocks[name].start]
    # BESS converter selection is coupled to purchase, avoiding an inert visit
    # when its battery has not been installed by the incumbent.
    return tasks


class InitialResourceScheduler(RiskNativeScheduler):
    def __init__(self,codec,seed,mass,options):
        super().__init__(codec,seed,mass,options)
        self.initial_rng=np.random.default_rng(np.random.SeedSequence([seed,70139]))
        tasks=resource_tasks(codec)
        self.initial_order=[tasks[i] for i in self.initial_rng.permutation(len(tasks))]
        self.initial_completed=[];self.initial_attempts=[];self.initial_decks={}

    def draw_initial(self,name,choices):
        choices=list(choices)
        if not choices:raise ValueError('Empty initial coverage domain: '+name)
        if not self.initial_decks.get(name):
            self.initial_decks[name]=[choices[i] for i in self.initial_rng.permutation(len(choices))]
        return self.initial_decks[name].pop()

    def field(self,name,ids,kind='relative',endpoint=True):
        ids=np.asarray(ids).ravel();return Field(name,ids[ids>=0],kind,endpoint)

    def asset_initial(self,reference,task):
        st=self.codec.study;fields=[];family=task.split(':')[0]
        if family=='new_capacity':
            tech=task.split(':',1)[1]
            j=self.draw_initial(task,[i for i,g in enumerate(st.new) if g['technology']==tech])
            fields.append(self.field('new_capacity',self.base.maps['new_capacity'][:,j],'direct'))
            for n in ('new_p','phys_new_q'):fields.append(self.field(n,self.base.maps[n][:,:,j]))
        else:
            d=next(d for d in self.base.categories if d['name']==family)
            j=self.draw_initial(task,[i for i,n in enumerate(d['sizes']) if n>1])
            fields=[Field(family,np.array([d['start']+j]),'direct',False)]
            if family=='storage_purchase':
                bus=self.codec.plan_codec.sites[j];site=self.codec.storage_sites.index(bus)
                fields.append(self.field('soc',self.base.maps['soc'][:,site,:],'direct'))
                fields.append(self.field('phys_bess_q',self.base.maps['phys_bess_q'][:,:,site],'direct'))
                units=np.flatnonzero(self.codec.unit_site==site)
                dchoice=next((d for d in self.base.categories if d['name']=='bess_choice'),None)
                if dchoice is not None:
                    fields.append(Field('bess_choice',dchoice['start']+units,'direct',False))
            elif family in ('cap_choice','reg_choice','retro_choice'):
                n={'cap_choice':'phys_cap_state','reg_choice':'phys_tap','retro_choice':'existing_q'}[family]
                fields.append(self.field(n,self.base.maps[n][:,:,j]))
        fields=[f for f in fields if len(f.indices)]
        asset=WindowBlock(reference,fields,dict(scope='investment_joint',family=family,site_index=j,
            complete_SOC_cycles=family=='storage_purchase',auxiliary_optimizer=False),self.base.mass,.02)
        asset.categorical_indices=[i for f,sl in zip(asset.fields,asset.slices) if not f.endpoint
            for i in range(sl.start,sl.stop)]
        asset.metadata['categorical_key_indices']=asset.categorical_indices
        if family=='storage_purchase':
            # All SOC states can move; balance P/Q against the changed battery
            # on ALL representative cycles, not just one unrelated day.
            groups=[[(k,s) for s in cy] for k in range(st.K) for cy in st.cycles]
        else:
            day=self.draw_initial('asset_day',range(len(st.cycles)))
            groups=[[(k,s) for s in st.cycles[day]] for k in range(st.K)]
        b=CoordinatedBlock(self.codec,self.base,reference,groups,self.base.mass,.02,asset)
        b.metadata.update(family=family,site_index=j,complete_SOC_cycles=family=='storage_purchase')
        return b

    def initial_block(self,reference,task):
        st=self.codec.study
        if task in ('line_schedule','storage_purchase','cap_choice','reg_choice','retro_choice') or task.startswith('new_capacity:'):
            return self.asset_initial(reference,task)
        if task=='topology':return self.tree(reference)
        if task=='service':return self.operation(reference)
        k=self.draw_initial('stage:'+task,range(st.K))
        day=self.draw_initial('day:'+task,range(len(st.cycles)))
        cases=[(k,s) for s in st.cycles[day]]
        if task.startswith('existing_PQ:'):
            tech=task.split(':',1)[1]
            js=[i for i,g in enumerate(st.old) if g['technology']==tech]
            fields=[self.field(n,self.base.maps[n][k][st.cycles[day]][:,js],'direct') for n in ('existing_p','existing_q')]
        else:
            fields=[self.field(task,self.base.maps[task][k][:,st.cycles[day]],'direct')]
        asset=WindowBlock(reference,[f for f in fields if len(f.indices)],dict(scope='initial_cycle',family=task,
            stage=k,cycle=day,auxiliary_optimizer=False),self.base.mass,.1)
        b=CoordinatedBlock(self.codec,self.base,reference,[cases],self.base.mass,.02,asset)
        b.metadata.update(family=task,cycle=day,stage=k)
        return b

    def choose(self,reference,contexts):
        pending=[t for t in self.initial_order if t not in self.initial_completed]
        if not pending:return super().choose(reference,contexts)
        task=pending[0];b=self.initial_block(reference,task)
        b.metadata.update(initial_resource_task=task,initial_resource_index=len(self.initial_completed),
            initial_resource_total=len(self.initial_order),initial_coverage_tier=self._tier(),
            initial_resource_order=self.initial_order,initial_resource_coverage=True,
            risk_role='initial_resource',auxiliary_optimizer=False,
            selection='unchanged evaluator/FO/CVaR/constraints; no forced acceptance')
        self.initial_attempts.append(dict(task=task,site_index=b.metadata.get('site_index')))
        return None,b

    def visit_generations(self,context,default):
        if context.block.metadata.get('initial_resource_coverage'):return min(2,default)
        return super().visit_generations(context,default)

    def finish_visit(self,context):
        task=context.block.metadata.get('initial_resource_task')
        native=self.current_stats.get(context.number,{}).get('native_calls',0)
        super().finish_visit(context)
        if task and native and task not in self.initial_completed:self.initial_completed.append(task)

    def report(self):
        out=super().report()
        out['initial_resource_coverage']=dict(order=self.initial_order,completed=self.initial_completed,
            attempts=self.initial_attempts,complete=len(self.initial_completed)==len(self.initial_order),
            interpretation='Native opportunities, not proof of accepted investment or all sites/combinations',
            before_feasibility=True,auxiliary_evaluations=0)
        return out
