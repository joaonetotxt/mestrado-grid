"""Native single-case P/Q coordinates with protected original exploration.

No solver, objective gradient, prescribed dispatch, or feasibility repair is
used. Resource decks are seed-only. Each engine chooses every signed P/Q value.
The original decoder and complete portfolio evaluator remain authoritative.
"""
import numpy as np
from . import physical
from .guided_blocks import WindowBlock
from .native_blocks import Field, Scheduler
from .active_search import ActiveScheduler


def free_resources(codec, base, reference, case):
    st=codec.study;k,s=case;c=codec.decode(reference)[0]
    lo,hi,*_=physical.existing_bounds(st,c)
    ql,qh,nl,nh,bl,bh=physical.q_bounds(st,c)
    cap=st.active_capacity(c);availability=physical.availability(st)
    rows=[]
    for kind,p_name,q_name,lower,upper,lower_q,upper_q in (
        ('existing','existing_p','existing_q',lo[k,s],hi[k,s],ql[k,s],qh[k,s]),
        ('additional','new_p','phys_new_q',np.zeros(st.N),cap[k]*availability[s],nl[k,s],nh[k,s]),
        ('BESS',None,'phys_bess_q',np.zeros(st.U),np.zeros(st.U),bl[k,s],bh[k,s])):
        for j in range(len(lower)):
            ids=[]
            if p_name is not None and upper[j]-lower[j]>1e-12:
                index=int(base.maps[p_name][k,s,j])
                if index>=0:ids.append(index)
            if upper_q[j]-lower_q[j]>1e-12:
                index=int(base.maps[q_name][k,s,j])
                if index>=0:ids.append(index)
            if ids:rows.append(dict(kind=kind,index=j,ids=ids))
    return rows


def case_block(codec,base,reference,case,selected,radius=.05,mass=.25):
    rows=free_resources(codec,base,reference,case)
    selected=set(selected)
    fields=[Field(f'{r["kind"]}_resource_{r["index"]}_PQ',np.array(r['ids']),kind='direct')
            for r in rows if (r['kind'],r['index']) in selected]
    return WindowBlock(reference,fields,dict(scope='case_native_PQ',scheduler='case_pq_v1',
        case_groups=[[tuple(case)]],selected_resources=[list(x) for x in sorted(selected)],
        resource_selection='seed-only decks',auxiliary_optimizer=False,
        independent_resource_PQ=True,pre_scored_neighbors=0,
        immutable='network, capacities, storage SOC, conventional/EV demand and shedding'),mass,radius)


class CasePQScheduler(ActiveScheduler):
    """One focused visit per three, with two unchanged original visits.

    Reliability-infeasible incumbents use the previous reliability search.
    Feasible incumbents alternate new single-case P/Q visits and native-memory
    revisits. Cases alternate economic feedback and protected seed-only decks.
    Resources rotate independently of costs through every currently free source.
    """
    def __init__(self,*args):
        super().__init__(*args)
        self.pq_visit=0;self.pq_new=0;self.pq_decks=Scheduler(args[0],args[1]+77123,args[2])
        self.pq_case_visits={}

    def choose(self,reference,contexts):
        if self.coverage_preflight:return super().choose(reference,contexts)
        record=self.record or {}
        if record.get('status')!='feasible':return super().choose(reference,contexts)
        slot=self.pq_visit%6;self.pq_visit+=1
        if slot in (1,2,4,5):return super().choose(reference,contexts)
        if slot==3:
            for ctx in reversed(contexts):
                if ctx.block.metadata.get('scope')=='case_native_PQ':return ctx,None
        st=self.codec.study
        adaptive=self.pq_new%2==1 and self.feedback and self.options.economic_allocation
        if adaptive:
            case=max(self.feedback,key=lambda ks:self.feedback[ks].get('economic_opportunity',0)/(1+self.pq_case_visits.get(ks,0)))
        else:
            # Only an actual protected visit consumes its deck position.
            case=divmod(self.pq_decks.draw('pq_states',st.K*st.S),st.S)
        self.pq_case_visits[case]=self.pq_case_visits.get(case,0)+1
        rows=free_resources(self.codec,self.base,reference,case)
        if not rows:return super().choose(reference,contexts)
        # Deck key includes domain identities, so availability/capacity changes
        # cannot leave stale resource indices in a previously created deck.
        domain=tuple((r['kind'],r['index']) for r in rows)
        width=min(8,len(rows));key=('pq_sources',case,domain)
        selected=[]
        while len(selected)<width:
            item=domain[self.pq_decks.draw(key,len(domain))]
            if item not in selected:selected.append(item)
        # Small windows are frequent; radius 1 remains recurrent and the old
        # unrestricted scheduler keeps all catalogue families eligible.
        radius=(.02,.05,.1,.25,1.)[(self.pq_case_visits[case]-1)%5]
        block=case_block(self.codec,self.base,reference,case,selected,radius,self.base.mass)
        block.metadata.update(allocation='economic_case' if adaptive else 'protected_seed_deck',
                              protected_original_scheduler_fraction=2/3,
                              visit_cycle='new_PQ, original, original, revisit_PQ, original, original')
        self.pq_new+=1
        return None,block

    def report(self):
        out=super().report()
        out.update(case_pq_visits=self.pq_visit,case_pq_new=self.pq_new,
            case_pq_cases=[list(ks) for ks in self.pq_case_visits],native_auxiliary_calls=0)
        return out
