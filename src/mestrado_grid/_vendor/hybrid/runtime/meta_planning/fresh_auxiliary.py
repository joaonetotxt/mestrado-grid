"""Bounded preparation from the exact MAT tree, with no historical vectors.

The catalogue supplies all proposals. Exact reliability screens only reject a
known lower bound on violation; unchanged full AC/rank decides every acceptance.
LP redispatch belongs to preparation only, never to the subsequent engines.
"""
import copy
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .prepared_search import SearchState,write,select_population
from .search_deadline import SearchDeadline,SearchDeadlineReached
from .topology import reliability,canonical,line_schedule
from .construction import reliability_tree
from .evaluator import Evaluator,rank
from .coordinated_ac import ACRefiner


def reliability_bound(st,plan):
    rows=reliability(st.data,plan)
    return sum(max(max(0.,r[name]-st.c[name+'_limits'][-1 if st.data['investment_mode']=='single' else k]-1e-8)
        /max(1.,st.c[name+'_limits'][-1 if st.data['investment_mode']=='single' else k])
        for k,r in enumerate(rows)) for name in ('SAIFI','SAIDI'))


def graph_neighbors(st,codec,reference,rng,check=lambda:None,on_bound=lambda:None):
    plan=codec.decode(reference)[0].plan;installed,_=line_schedule(st.data,plan)
    lines=codec.plan_codec.lines;byid={e['id']:e for e in lines};pool=[]
    for k,edges in enumerate(plan.trees):
        tree=set(edges);adj={b:[] for b in st.buses}
        for eid in tree:
            e=byid[eid];adj[e['i']].append((e['j'],eid));adj[e['j']].append((e['i'],eid))
        for j,added in enumerate(lines):
            check()
            if added['id'] in tree:continue
            first,last=added['i'],added['j'];parent={first:(None,None)};queue=[first]
            for b in queue:
                if b==last:break
                for child,eid in adj[b]:
                    if child not in parent:parent[child]=(b,eid);queue.append(child)
            path=[];b=last
            while b!=first:b,eid=parent[b];path.append(eid)
            for removed in path:
                check();p=copy.deepcopy(plan);p.trees[k]=sorted(tree-{removed}|{added['id']})
                if installed[added['id'],k] is None:
                    earliest=min(kk for kk,tt in enumerate(p.trees) if added['id'] in tt)
                    choices=[c for c in codec.plan_codec.line_choices[j] if c and min(x[2] for x in c)<=earliest]
                    choice=choices[int(rng.integers(len(choices)))]
                    p.line_buys=[x for x in p.line_buys if x[0]!=added['id']]+list(choice)
                p=canonical(p);bound=reliability_bound(st,p);on_bound()
                pool.append((bound,float(rng.random()),p,dict(stage=k,added=added['id'],removed=removed)))
    return sorted(pool,key=lambda row:row[:2])


def prepare(study,output,seed=96101,population=8,wall_seconds=3600,direct_calls=512,max_sweeps=20,refine_rounds=4):
    s=SearchState(study,seed,population,output,'mat_original')
    timer=SearchDeadline(wall_seconds,start=s.start);counts=Counter();polls=[];refiners=[]
    s.initialize();s.save_best();rng=np.random.default_rng(np.random.SeedSequence([seed,96211]))
    stopped=False
    def propose(c,event):
        timer.check()
        r=s.evaluate(s.codec.encode(c),dict(origin='heuristic_preparation',**event))
        s.save_best();return r
    def refine_best(rounds):
        nonlocal refiners
        for iteration in range(rounds):
            timer.check();before=copy.deepcopy(s.best);c=s.codec.decode(s.best_x)[0]
            ref=ACRefiner(study);detail=dict(iteration=iteration,counts={},completed=False)
            refiners.append(detail)
            try:
                trial,record,meta=ref.refine(c,rounds=1,callback=lambda *args:timer.check())
                detail.update(completed=True,metadata=meta)
                propose(trial,dict(component='coordinated_AC_redispatch',iteration=iteration,native_credit=False))
            finally:
                detail['counts']=dict(ref.counts)
                write(s.dst/f'auxiliary_refinement_{len(refiners)-1}.json',dict(detail,history=ref.history))
            if rank(s.best)>=rank(before):break
    try:
        for sweep in range(max_sweeps):
            if s.best['status']=='feasible':break
            timer.check();before=copy.deepcopy(s.best)
            s.status(phase='auxiliary_graph_neighborhood')
            # Count even an interrupted pool; each completed bound is one screen.
            def graph_check():timer.check()
            pool=graph_neighbors(study,s.codec,s.best_x,rng,graph_check,
                lambda:counts.update(reliability_neighbor_examinations=1))
            info=dict(sweep=sweep,neighbors=len(pool),fully_evaluated=0,screened=0,
                best_bound=min((x[0] for x in pool),default=None),before=before)
            polls.append(info)
            for bound,_,plan,event in pool:
                if s.best['status']=='violated' and bound>0 and bound>=s.best['normalized_violation']:
                    info['screened']+=1;continue
                candidate=s.codec.decode(s.best_x)[0];candidate.plan=plan
                propose(candidate,dict(component='fundamental_cycle_poll',sweep=sweep,exchange=event,reliability_bound=bound))
                info['fully_evaluated']+=1
                if rank(s.best)<rank(before) or info['fully_evaluated']>=64:break
            info['after']=copy.deepcopy(s.best);write(s.dst/'graph_polls.json',polls)
            if rank(s.best)>=rank(before):break
        # Fresh catalogue-wide tree proposals are explicit auxiliary moves, not
        # a substituted initial seed. Assets/controls still descend from this run.
        if s.best['status']!='feasible':
            for proposal in range(8):
                candidate=s.codec.decode(s.best_x)[0];candidate.plan=reliability_tree(study,rng)
                propose(candidate,dict(component='catalogue_reliability_tree',proposal=proposal))
                refine_best(1)
                if s.best['status']=='feasible':break
        refine_best(refine_rounds)
        for _ in range(direct_calls//population):
            timer.check()
            # Each call is atomic; check between every candidate, not just batch.
            anchor=s.best_x.copy()
            xs,events=s.variation.propose(np.tile(anchor,(population,1)),anchor)
            for x,event in zip(xs,events):
                timer.check();r=s.evaluate(x,dict(origin='heuristic_preparation',component='hierarchical_symmetric',auxiliary_event=event))
                s.variation.observe(event,r)
            s.save_best()
        refine_best(refine_rounds)
    except SearchDeadlineReached:
        stopped=True
    # Handoff contains only independently feasible members; repeat if necessary
    # and declare the number of distinct decisions rather than invent diversity.
    feasible_bank=[b for b in s.bank if b['record']['status']=='feasible']
    if not feasible_bank:
        write(s.dst/'preparation_failure.json',dict(reason='No feasible point in bounded fresh preparation',best=s.best,
            evaluations=s.evaluations,counts=dict(counts),refinement=refiners,timing=timer.report()))
        raise RuntimeError('Fresh MAT preparation exhausted without a feasible point; engines were not started')
    s.bank=feasible_bank
    prep=s.export_preparation(s.evaluations-s.n)
    prep.update(termination_reason='wall_time_limit' if stopped else 'planned_auxiliary_work_completed',
        graph_examinations=dict(counts),refinement=refiners,search_timing=timer.report(),
        historical_vectors_used=False,auxiliary_off_during_native=True,
        count_note='Graph bound examinations include candidates also fully evaluated; refiner full and case evaluations are additional to SearchState proposals.')
    write(s.dst/'preparation.json',prep)
    audits=[];audit_start=time.monotonic()
    for i,x in enumerate(s.population):
        c=s.codec.decode(x)[0];r=Evaluator(study).evaluate(c)
        assert r['status']=='feasible' and r['approved_cases']==study.K*study.S and r['normalized_violation']==0
        assert c.candidate_id==r['candidate_id']
        assert np.isclose(r['metrics']['selection_objective'],s.cache[c.candidate_id]['metrics']['selection_objective'],rtol=1e-13,atol=1e-5)
        write(s.dst/f'population_validation_{i}.json',r)
        audits.append(dict(member=i,candidate_id=c.candidate_id,plan_id=c.plan_id,status=r['status'],approved_cases=r['approved_cases']))
    write(s.dst/'handoff_audit.json',dict(verified=True,members=audits,independent_AC_cases=population*study.K*study.S,
        audit_seconds=time.monotonic()-audit_start,population_sha256=prep['prepared_population_sha256']))
    s.status('completed','population_independently_validated')
    return prep
