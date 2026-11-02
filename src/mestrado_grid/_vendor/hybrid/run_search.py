"""Four independent incumbent searches. No time/call stop in production."""
from campaign_support import *
from hybrid_core import FAMILIES,RiskRefiner,opportunities,apply,day_order
from search_scope import broad_proposal
from autonomous_ai import CodexSupervisor,InferenceInterrupted
from collections import Counter,deque

class PairStopped(Exception):pass
class PhaseChanged(Exception):pass
class SmokeComplete(Exception):pass

class Search:
    def __init__(self,pair,method,section='runs',smoke_calls=None,resume=False):
        if pair not in PAIRS or method not in METHODS:raise ValueError('Unknown arm')
        self.pair=pair;self.method=method;self.section=section;self.smoke_calls=smoke_calls
        self.out=D/section/pair/method;self.out.mkdir(parents=True,exist_ok=resume)
        self.shared=D/section/pair;self.strict,self.relaxed,_=studies()
        self.phase='relaxed' if pair=='relaxed_then_strict' else 'strict'
        self.started=now();self.clock=time.monotonic();self.cpu=time.process_time();self.previous_wall=0.;self.previous_cpu=0.
        self.calls=0;self.ac_cases=0;self.audit_calls=0;self.audit_ac=0;self.visit=0;self.day_cursor=0;self.init_index=0
        self.best=None;self.current=None;self.initial=None;self.counts=Counter();self.recent=deque(maxlen=8);self.pairs=deque(maxlen=4)
        self.order=FAMILIES.copy();self.radius=.5;self.next_ai=8;self.has_ai=False;self.ai_retry_at=0.;self.menu=None;self.menu_plan=None;self.seed=96101
        self.last_audited=None;self.last_periodic=0;self.used=set();self.configure()
        self.sup=CodexSupervisor(self.out/'ai') if method=='HYBRID_AI' else None
        if resume:self.restore()
        save(self.out/'identity.json',dict(pid=os.getpid(),parent_pid=os.getppid(),created_utc=now(),argv=sys.argv,
                executable=sys.executable,pair=pair,method=method,section=section,resume=resume,seed=self.seed,
                strict_input_id=self.strict.input_id,relaxed_input_id=self.relaxed.input_id))

    def configure(self):
        self.st=self.strict if self.phase=='strict' else self.relaxed
        self.ev=CountingEvaluator(self.st);self.ref=RiskRefiner(self.st)

    def stopped(self):
        return (self.shared/'WINNER.json').exists() or (D/'STOP_USER.json').exists() or (self.shared/'STOP_USER.json').exists()

    def check(self):
        if self.stopped():raise PairStopped()
        if self.smoke_calls is not None and self.calls>=self.smoke_calls:raise SmokeComplete()

    def progress(self,state,**extra):
        save(self.out/'status.json',dict(state=state,updated_utc=now(),pid=os.getpid(),pair=self.pair,method=self.method,
            phase=self.phase,strict_input_id=self.strict.input_id,calls=self.calls,AC_cases=self.ac_cases,
            audit_calls=self.audit_calls,audit_AC_cases=self.audit_ac,visit=self.visit,initial_members_evaluated=self.init_index,
            elapsed_s=self.previous_wall+time.monotonic()-self.clock,cpu_s=self.previous_cpu+time.process_time()-self.cpu,
            best=compact(self.best) if self.best else None,coverage=dict(self.counts),
            genuine_AI_policy=self.has_ai,AI_requests=self.sup.calls if self.sup else 0,**extra))

    def checkpoint(self):
        save_gz(self.out/'checkpoint.json.gz',dict(version=1,pair=self.pair,method=self.method,phase=self.phase,
            calls=self.calls,ac_cases=self.ac_cases,audit_calls=self.audit_calls,audit_ac=self.audit_ac,
            visit=self.visit,day_cursor=self.day_cursor,init_index=self.init_index,counts=dict(self.counts),
            best=self.best,current=self.current.to_dict() if self.current else None,initial=self.initial,
            recent=list(self.recent),pairs=list(self.pairs),order=self.order,radius=self.radius,next_ai=self.next_ai,
            has_ai=self.has_ai,used=list(self.used),last_audited=self.last_audited,last_periodic=self.last_periodic,
            wall=self.previous_wall+time.monotonic()-self.clock,cpu=self.previous_cpu+time.process_time()-self.cpu,
            strict_input_id=self.strict.input_id))

    def restore(self):
        x=read_gz(self.out/'checkpoint.json.gz')
        assert x['pair']==self.pair and x['method']==self.method and x['strict_input_id']==self.strict.input_id
        for k in ('phase','calls','ac_cases','audit_calls','audit_ac','visit','day_cursor','init_index','initial','order','radius','next_ai','has_ai','last_audited','last_periodic'):
            setattr(self,k,x[k])
        self.best=x['best'];self.current=Candidate.from_dict(x['current']) if x['current'] else None
        self.counts=Counter(x['counts']);self.used={tuple(z) for z in x['used']};self.recent=deque(x['recent'],maxlen=8);self.pairs=deque(x['pairs'],maxlen=4)
        self.previous_wall=x['wall'];self.previous_cpu=x['cpu'];self.configure()
        if self.current:
            v=self.fresh(self.current,self.st,'resume')
            if not audit_agrees(v,self.best):raise RuntimeError('Checkpoint failed fresh re-evaluation')
            append(self.out/'reproduction_deltas.jsonl',dict(utc=now(),label='resume',saved=compact(self.best),fresh=compact(v)))
            self.best=v
        append(self.out/'resumptions.jsonl',dict(utc=now(),calls=self.calls,checkpoint_sha256=sha(self.out/'checkpoint.json.gz')))

    def fresh(self,c,study,label):
        self.progress('auditing',audit_label=label)
        ev=CountingEvaluator(study);r=ev.evaluate(c);self.audit_calls+=ev.calls;self.audit_ac+=ev.ac_cases
        root=self.out/'audits';root.mkdir(exist_ok=True);path=root/f'{self.audit_calls:06d}_{label}.json.gz'
        save_gz(path,r)
        if r.get('metrics'):finance_checks(r)
        append(self.out/'audits.jsonl',dict(label=label,path=str(path.relative_to(D)),sha256=sha(path),
            input_id=study.input_id,calls=self.calls,result=compact(r),utc=now()))
        self.last_audit_path=path
        return r

    def accept_record(self,c,r,event):
        accepted=self.best is None or rank(r)<rank(self.best)
        if accepted:
            self.best=r;self.current=copy.deepcopy(c)
            save(self.out/'best_candidate.json',c.to_dict());save_gz(self.out/'best_validation.json.gz',r)
            append(self.out/'improvements.jsonl',dict(utc=now(),call=self.calls,phase=self.phase,event=event,result=compact(r)))
        row=dict(utc=now(),call=self.calls,phase=self.phase,accepted=accepted,event=event,result=compact(r))
        self.recent.append(row);append(self.out/'history.jsonl',row)
        self.progress('running');return accepted

    def evaluate(self,c,event):
        self.check();t=time.monotonic();before=self.ev.ac_cases;r=self.ev.evaluate(c)
        self.calls+=1;self.ac_cases+=self.ev.ac_cases-before
        self.counts[self.phase+'_full_calls']+=1;event=event|{'elapsed_s':time.monotonic()-t}
        self.accept_record(c,r,event)
        if self.phase=='relaxed' and valid(r):
            v=self.fresh(c,self.relaxed,'first_relaxed_feasible')
            if not valid(v) or not audit_agrees(v,r):raise RuntimeError('Relaxed handoff failed independent audit')
            self.phase='strict';self.configure();self.best=None;self.current=None;self.menu=None;self.used.clear()
            self.order=FAMILIES.copy();self.next_ai=self.calls;self.has_ai=False;self.visit=0;self.day_cursor=0
            # Context retest is charged separately, never an optimization gain.
            strict=self.fresh(c,self.strict,'transition_strict_retest')
            self.accept_record(c,strict,dict(family='context_retest',from_phase='relaxed'))
            save(self.out/'transition.json',dict(utc=now(),at_call=self.calls,candidate_id=c.candidate_id,
                relaxed=compact(v),strict=compact(strict),immediate=True,ranking_and_schedule_reset=True,
                strict_input_id=self.strict.input_id,relaxed_input_id=self.relaxed.input_id))
            if strict_success(strict,'strict'):self.publish(c,strict)
            self.checkpoint();raise PhaseChanged()
        if strict_success(r,self.phase):self.publish(c,r)
        return r

    def publish(self,c,r):
        self.check();v=self.fresh(c,self.strict,'target')
        if (not strict_success(v,'strict') or not audit_agrees(v,r)
                or v.get('signature')!=Evaluator(self.strict).signature):
            raise RuntimeError('Target failed fresh strict audit or strict evaluation signature')
        verify_frozen();finance=finance_checks(v)
        candidate=self.out/'target_candidate.json';save(candidate,c.to_dict())
        item=dict(utc=now(),pair=self.pair,method=self.method,strict_input_id=self.strict.input_id,
            target=TARGET,result=compact(v),finance=finance,calls=self.calls,
            audit_path=str(self.last_audit_path.relative_to(D)),audit_sha256=sha(self.last_audit_path),
            candidate_path=str(candidate.relative_to(D)),candidate_sha256=sha(candidate))
        # Only the first completely written, verified marker wins the pair.
        pending=self.out/'winner_ready.json';save(pending,item)
        try:
            # Windows rename is atomic and refuses an existing destination.
            # Hard links are unnecessarily restricted by the Windows sandbox.
            if os.name=='nt':os.rename(pending,self.shared/'WINNER.json')
            else:os.link(pending,self.shared/'WINNER.json')
        except FileExistsError:pass
        raise PairStopped()

    def joint(self,c,r,k,cy,radius,event):
        self.check();rows={(x['stage'],x['state_id']):x for x in r.get('cases',[])}
        z=copy.deepcopy(c);meta={};t=time.monotonic()
        if all(rows.get((k,s),{}).get('converged') for s in cy):
            step,meta=self.ref.propose(c,k,cy,rows,radius=radius,record=r)
            if step:z=self.ref.apply(c,k,cy,step,1.)
        self.counts['LP_wall_ms']+=round((time.monotonic()-t)*1000)
        rr=self.evaluate(z,event|dict(stage=k,cycle=list(cy),LP=meta))
        return (z,rr) if rank(rr)<rank(r) else (c,r)

    def decide(self):
        if self.calls<self.next_ai:return
        if not self.sup:
            self.order=(['topology','dispatch'] if not valid(self.best) else ['dispatch','topology'])+[f for f in FAMILIES if f not in ('dispatch','topology')]
            self.radius=.5;self.next_ai=self.calls+128;return
        if time.monotonic()<self.ai_retry_at and self.has_ai:return
        observation=dict(phase=self.phase,regime='extremo',alpha=self.st.c['alpha'],beta=self.st.c['beta'],calls=self.calls,visit=self.visit,
            families=FAMILIES,current=compact(self.best),recent=list(self.recent),paired_controls=list(self.pairs),coverage=dict(self.counts),
            rules='Feasibility then exact cost. All resources eligible. P/Q, new generation and BESS have no direct SAIDI/SAIFI credit. No candidate or solution exchange. Every fourth visit rotates unrestricted catalogue draws. Each asset has two equal dispatch rounds on asset and no-change control. Stop only strict feasible FO <25 million. Choose only order/radius.')
        while True:
            self.check();self.progress('waiting_for_ai')
            try:
                d=self.sup.choose(observation,self.stopped,lambda:self.progress('waiting_for_ai'))
                self.order=d['family_order'];self.radius=d['radius'];self.has_ai=True;self.next_ai=self.calls+128
                append(self.out/'decisions.jsonl',dict(utc=now(),calls=self.calls,phase=self.phase,decision=d))
                self.checkpoint();return
            except InferenceInterrupted:raise PairStopped()
            except Exception as e:
                append(self.out/'inference_errors.jsonl',dict(utc=now(),error=str(e),calls=self.calls,last_genuine_policy_retained=self.has_ai))
                self.ai_retry_at=time.monotonic()+600
                if self.has_ai:self.next_ai=self.calls+128;return
                while time.monotonic()<self.ai_retry_at:
                    self.check();self.progress('waiting_for_ai_retry');time.sleep(5)

    def step(self):
        self.check();self.decide();forced=self.visit%4==3
        family=FAMILIES[(self.visit//4)%len(FAMILIES)] if forced else self.order[self.visit%len(self.order)]
        self.visit+=1;self.counts['visits_'+family]+=1;c=copy.deepcopy(self.current);r=self.best
        if family=='dispatch':
            days=day_order(self.st,r);k,cy=days[self.day_cursor%len(days)];self.day_cursor+=1
            self.joint(c,r,k,cy,self.radius,dict(family='dispatch',branch='incumbent'));return
        asset=None;d=None
        if forced:
            try:asset,d=broad_proposal(self.st,c,r,family,self.seed,self.visit)
            except ValueError as e:
                self.counts['structural_proposal_rejections']+=1
                append(self.out/'proposal_rejections.jsonl',dict(utc=now(),visit=self.visit,family=family,error=str(e),AC_calls=0))
        else:
            if self.menu is None or self.menu_plan!=c.plan_id or self.visit%8==0:
                self.menu=opportunities(self.st,c,r,self.seed,self.visit);self.menu_plan=c.plan_id
            possible=[d for d in self.menu if d['family']==family and (c.plan_id,json.dumps(d,sort_keys=True)) not in self.used]
            if possible:
                # Reproducible choice avoids always taking the first ranked site.
                rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.visit,44]));d=possible[int(rng.integers(len(possible)))]
                self.used.add((c.plan_id,json.dumps(d,sort_keys=True)))
                try:asset=apply(self.st,c,d,r)
                except ValueError as e:
                    self.counts['structural_proposal_rejections']+=1
                    append(self.out/'proposal_rejections.jsonl',dict(utc=now(),visit=self.visit,resource=d,error=str(e),AC_calls=0))
        if self.visit%224==0:self.used.clear()  # declared periodic native neighbourhood revisit
        if asset is None:
            self.counts['unavailable_'+family]+=1;days=day_order(self.st,r);k,cy=days[self.day_cursor%len(days)];self.day_cursor+=1
            self.joint(c,r,k,cy,self.radius,dict(family='dispatch',branch='fallback',requested_family=family));return
        begin=self.calls;ar=self.evaluate(asset,dict(family=family,branch='asset_initial',resource=d,forced=forced))
        control=c;cr=r;days=day_order(self.st,r);selected=[]
        for round_i in range(2):
            k,cy=days[self.day_cursor%len(days)];self.day_cursor+=1;selected.append((k,cy))
            control,cr=self.joint(control,cr,k,cy,self.radius,dict(family='dispatch',branch='paired_control',visit=self.visit,round=round_i))
            asset,ar=self.joint(asset,ar,k,cy,self.radius,dict(family='dispatch',branch='paired_asset',visit=self.visit,round=round_i))
        if self.calls-begin!=5:raise RuntimeError('Unbalanced resource comparison')
        pair=dict(visit=self.visit,phase=self.phase,resource=d,full_calls=5,redispatch_rounds_per_branch=2,
            same_days=True,days=selected,control=compact(cr),asset=compact(ar),asset_beats_control=rank(ar)<rank(cr))
        self.pairs.append(pair);append(self.out/'paired_controls.jsonl',pair)
        assert rank(self.best)<=min(rank(ar),rank(cr))

    def run(self):
        try:
            pop=np.load(D/'inputs/extremo/sim1.npy',allow_pickle=False);codec=ArrayCodec(self.strict)
            while self.init_index<len(pop):
                c=codec.decode(pop[self.init_index])[0];i=self.init_index;self.init_index+=1
                try:self.evaluate(c,dict(family='initialization',member=i))
                except PhaseChanged:pass
                self.checkpoint()
            if self.initial is None:self.initial=compact(self.best);save(self.out/'initial.json',self.initial)
            while True:
                try:self.step()
                except PhaseChanged:pass
                self.checkpoint()
                if self.calls-self.last_periodic>=256:
                    self.last_periodic=self.calls
                    if self.current.candidate_id!=self.last_audited:
                        v=self.fresh(self.current,self.st,'periodic')
                        if not audit_agrees(v,self.best):raise RuntimeError('Periodic audit mismatch')
                        append(self.out/'reproduction_deltas.jsonl',dict(utc=now(),label='periodic',saved=compact(self.best),fresh=compact(v)))
                        self.last_audited=self.current.candidate_id
                    self.checkpoint()
        except (PairStopped,SmokeComplete) as e:
            reason='smoke_complete' if isinstance(e,SmokeComplete) else 'pair_target' if (self.shared/'WINNER.json').exists() else 'user_stop'
            self.checkpoint()
            if self.current:
                v=self.fresh(self.current,self.strict,'final_strict')
                save(self.out/'final_candidate.json',self.current.to_dict());save(self.out/'summary.json',dict(
                    reason=reason,phase_at_stop=self.phase,initial=self.initial,final_strict=compact(v),calls=self.calls,
                    actual_AC_cases=self.ac_cases,audit_calls=self.audit_calls,audit_AC_cases=self.audit_ac,
                    finance=finance_checks(v) if v.get('metrics') else None,AI_wait_s=self.sup.wait_s if self.sup else 0,
                    AI_usage=self.sup.usage if self.sup else [],coverage=dict(self.counts),utc=now()))
            self.progress(reason)
        except Exception:
            self.checkpoint();save(self.out/'error.json',dict(utc=now(),traceback=traceback.format_exc(),calls=self.calls))
            self.progress('failed');raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('pair',choices=PAIRS);p.add_argument('method',choices=METHODS)
    p.add_argument('--section',default='runs');p.add_argument('--smoke-calls',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.section=='runs' and args.smoke_calls is not None:raise ValueError('No artificial production budget')
    verify_frozen();Search(args.pair,args.method,args.section,args.smoke_calls,args.resume).run()
