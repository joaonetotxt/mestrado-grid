"""Rolling plans and persistent independent cycle/cases alternation.

Physical kernel, acceptance, finance, risk and paired stopping are inherited unchanged.
"""
from analytic_search import *
from revision_policy import ReviewGate, pack_observation, unpack_observation, frontier, EXTRA_PROMPT
from review_state import BranchVault
from coverage_fix import SharedDispatchMode
from analytic_policy import PROMPT as ORIGINAL_PROMPT
from analytic_ai import Analyst
import allocation_ai as transport

transport.PROMPT = ORIGINAL_PROMPT[:ORIGINAL_PROMPT.index('rounds must be')] + '''rounds must be 1..12 and radius .03125..1.
Production has no total time or evaluation ceiling. An isolated test explicitly supplies its remaining budget.
''' + EXTRA_PROMPT.replace('this isolated diagnostic trial has only the explicit remaining evaluations given in its observation',
    'only isolated validation sections have the explicit remaining evaluations given in their observation')


class ReviewedSearch(AnalyticSearch):
    def __init__(self, *args, **kwargs):
        self.review_gate = ReviewGate()
        self.shared_modes = SharedDispatchMode()
        self.review_sequence = 0
        self.vault = None
        self.vault_state = None
        super().__init__(*args, **kwargs)
        self.vault = BranchVault(self.out/'suspended', self.vault_state)
        self.sup = Analyst(self.out/'ai_review') if self.method == 'HYBRID_AI' else None
        self.prune_work()
        save(self.out/'review_restore_verified.json', dict(utc=now(), calls=self.calls,
            candidate_id=self.current.candidate_id, strict_input_id=self.strict.input_id,
            gate=vars(self.review_gate), shared_dispatch_visits=self.shared_modes.visits,
            review_sequence=self.review_sequence, vault=self.vault.export(),
            active_step=self.active_step, pending=len(self.queue), active_work=len(self.kernel.work)))
        self.checkpoint()

    def restore_analytic(self, ext):
        super().restore_analytic(ext)
        state = ext.get('review_state')
        if state:
            self.review_gate.__dict__.update(state['gate'])
            self.shared_modes.visits = state['shared_dispatch_visits']
            self.review_sequence = state['sequence']
            self.vault_state = state['vault']
        elif self.migration_source:
            self.legacy_ai_requests = self.migration_source.get('legacy_AI_requests', self.legacy_ai_requests)

    def clock_value(self):
        # Cumulative active wall time, persisted by the inherited checkpoint.
        # No monotonic timestamp from a former process and no downtime charged.
        return self.previous_wall + time.monotonic() - self.clock

    def progress(self, state='running', **extra):
        k = self.kernel
        Search.progress(self, state, version='rolling_review_v2', analytic_visit=self.analytic_visit,
            pending_experiments=len(self.queue), portfolio_size=len(k.bank.entries) if k else 0,
            analytic_counters=dict(k.counts) if k else {}, legacy_AI_requests=self.legacy_ai_requests,
            review_sequence=self.review_sequence, shared_dispatch_visits=self.shared_modes.visits,
            suspended_experiments=self.vault.count if self.vault else 0, **extra)

    def checkpoint(self):
        k = self.kernel
        ext = None
        if k is not None:
            ext = dict(version=1, portfolio=k.bank.export(),
                coverage=[dict(stage=a, states=list(b), visits=n) for (a,b),n in k.coverage.visits.items()],
                feedback=k.feedback, counts=dict(k.counts), work=encode_work(k.work),
                pinned=[dict(candidate=c.to_dict(), record=r) for c,r,_ in k.pinned_sources.values()],
                queue=self.queue, visit=self.analytic_visit, active_step=self.active_step,
                AI_wait_s=self.ai_previous_wait+(self.sup.wait_s if self.sup else 0),
                AI_usage=self.ai_previous_usage+(self.sup.usage if self.sup else []),
                legacy_AI_requests=self.legacy_ai_requests,
                review_state=dict(gate=vars(self.review_gate), shared_dispatch_visits=self.shared_modes.visits,
                    sequence=self.review_sequence, vault=self.vault.export() if self.vault else self.vault_state))
        save_gz(self.out/'checkpoint.json.gz', dict(version=3, pair=self.pair, method=self.method,
            phase=self.phase, calls=self.calls, ac_cases=self.ac_cases, audit_calls=self.audit_calls,
            audit_ac=self.audit_ac, visit=self.visit, day_cursor=self.day_cursor, init_index=self.init_index,
            counts=dict(self.counts), best=self.best, current=self.current.to_dict() if self.current else None,
            initial=self.initial, recent=list(self.recent), pairs=list(self.pairs), order=self.order,
            radius=self.radius, next_ai=self.next_ai, has_ai=self.has_ai, used=list(self.used),
            last_audited=self.last_audited, last_periodic=self.last_periodic,
            wall=self.clock_value(), cpu=self.previous_cpu+time.process_time()-self.cpu,
            strict_input_id=self.strict.input_id, analytic_extension=ext,
            legacy_extension_preserved_in_parent=True))

    def suspend(self, key):
        w = self.kernel.work[key]
        self.vault.put(encode_work({key:w}), dict(experiment=key, call=self.calls,
            source=w['source'], control=compact(w['cr']), asset=compact(w['ar']) if w['ar'] else None,
            remaining_cycles=len(w['cycles']), completed_cycles=w['completed_cycles']))
        del self.kernel.work[key]

    def prune_work(self):
        if self.vault is None: return
        keep = {key for key,p in self.queue}
        if self.active_step: keep.add(self.active_step['work_id'])
        for key in list(self.kernel.work):
            if key not in keep: self.suspend(key)
        needed = {p['source'] for key,p in self.queue}
        if self.active_step: needed.add(self.active_step['proposal']['source'])
        self.kernel.pinned_sources = {cid:item for cid,item in self.kernel.pinned_sources.items() if cid in needed}

    def decide_analytic(self):
        if not self.sup: return
        reason = self.review_gate.reason(self.calls, self.clock_value(),
            self.best['metrics']['selection_objective'], self.queue)
        if not reason: return
        k = self.kernel
        while True:
            self.check()
            # Single-threaded immutable view while the real supervisor computes.
            sources = {self.current.candidate_id:(self.current,self.best)}
            sources.update({cid:(c,r) for cid,(c,r,_) in k.bank.entries.items()})
            for w in k.work.values():
                for ck,rk in [('control','cr'),('asset','ar')]:
                    if w[ck] is not None: sources[w[ck].candidate_id]=(w[ck],w[rk])
            dormant = self.vault.select()
            dormant_ids = set()
            for item in dormant:
                for w in decode_work(self.vault.load(item)).values():
                    for ck,rk in [('control','cr'),('asset','ar')]:
                        if w[ck] is not None:
                            cid=w[ck].candidate_id
                            if cid not in sources: sources[cid]=(w[ck],w[rk]); dormant_ids.add(cid)
            obs = observation(self.st,self.current,self.best,k.bank,k.feedback,k.coverage,
                None if self.smoke_calls is None else self.smoke_calls-self.calls)
            obs.update(review_reason=reason, unfinished_experiments=frontier(k),
                suspended_experiments=dormant,
                continuation_sources=[dict(id=cid,result=compact(r)) for cid,(c,r) in sources.items()],
                continuation_semantics='Only exact own-arm candidates; recent and rotating suspended branches. Old full work records remain losslessly on disk. Selecting a source begins a new explicit experiment from that evolved candidate.',
                actuator_semantics='Stage purchases persist. Topology exchanges propagate to later stages. Incumbent is bound to this exact observation, never silently transplanted.')
            for ar in obs['archives']:
                rr=sources[ar['id']][1]
                ar['case_violations']=[dict(stage=x['stage'],state=x['state_id'],violations={a:b for a,b in x.get('normalized_violations',{}).items() if b!=0}) for x in rr.get('cases',[]) if any(x.get('normalized_violations',{}).values())]
            packed=pack_observation(obs)
            assert unpack_observation(packed)==obs
            self.progress('waiting_for_ai')
            try:
                decision=self.sup.choose(packed,self.stopped,lambda:self.progress('waiting_for_ai'))
                answer=copy.deepcopy(decision)
                chosen=set()
                for p in answer['proposals']:
                    cid=self.current.candidate_id if p['source']=='incumbent' else p['source']
                    if cid not in sources: raise ValueError('Unknown own-arm source in reviewed plan')
                    c,r=sources[cid]
                    if cid in dormant_ids and cid not in chosen:
                        fresh=self.fresh(c,self.st,'suspended_source')
                        if not audit_agrees(fresh,r) and not archive_roundoff_only(fresh,r):
                            raise RuntimeError('Suspended source did not reproduce')
                    chosen.add(cid)
                    p['source']=cid
                before=len(self.queue)
                for key in list(k.work): self.suspend(key)
                k.pinned_sources.clear()
                for cid in chosen: k.pin_source(*sources[cid])
                self.review_sequence+=1
                self.queue=[(f'REVIEW_{self.review_sequence}_{i}',p) for i,p in enumerate(answer['proposals'])]
                self.has_ai=True
                self.review_gate.reviewed(self.calls,self.clock_value(),self.best['metrics']['selection_objective'])
                append(self.out/'reviews.jsonl',dict(utc=now(),reason=reason,call=self.calls,
                    candidate_id=self.current.candidate_id,pending_before=before,pending_after=len(self.queue),
                    sequence=self.review_sequence,source_ids=sorted(chosen),vault=self.vault.export(),
                    observation_bytes=len(json.dumps(obs)),packed_bytes=len(json.dumps(packed)),
                    bound_proposals=answer['proposals']))
                self.checkpoint()
                return
            except InferenceInterrupted: raise PairStopped()
            except (PairStopped,SmokeComplete): raise
            except Exception as exc:
                append(self.out/'inference_errors.jsonl',dict(utc=now(),call=self.calls,error=str(exc),synthetic_policy_used=False))
                retry=time.monotonic()+60
                while time.monotonic()<retry:
                    self.check();self.progress('waiting_for_ai_retry');time.sleep(5)

    def step(self):
        self.check()
        if self.active_step is None: self.decide_analytic()
        k=self.kernel;before=self.calls
        if self.active_step is not None:
            p,owner,experiment,work_id=(self.active_step[x] for x in ('proposal','owner','experiment','work_id'))
        else:
            self.analytic_visit+=1;self.visit+=1;v=self.analytic_visit;experiment=None
            if v%4==1:
                stage,cy=k.coverage.select(self.st,self.best)
                mode=self.shared_modes.next()
                p=proposal(self.st,'incumbent',[(stage,cy)],mode=mode)
                owner='shared_cycle_coverage'
            elif v%4==2: p=deterministic(k,v,self.seed,directed=False);owner='shared_resource_coverage'
            elif self.sup: experiment,p=self.queue[0];owner='AI_physical_proposal'
            else: p=deterministic(k,v,self.seed);owner='deterministic_physical_proposal'
            work_id=experiment or f'visit_{v}'
            self.active_step=dict(proposal=p,owner=owner,experiment=experiment,work_id=work_id)
            append(self.out/'proposals.jsonl',dict(call=self.calls,source=owner,proposal=p,shared_dispatch_visits=self.shared_modes.visits))
        try:
            done=k.execute(p,owner,work_id)
            if experiment is not None:
                self.queue.pop(0)
                if not done: self.queue.append((experiment,p))
        except (ValueError,KeyError,IndexError) as exc:
            k.counts['rejected_proposals']+=1
            append(self.out/'proposal_rejections.jsonl',dict(call=self.calls,proposal=p,error=str(exc)))
            if experiment is not None: self.queue.pop(0)
            if work_id in k.work: self.suspend(work_id)
        if self.calls==before:
            stage,cy=k.coverage.select(self.st,self.best)
            k.refine(copy.deepcopy(self.current),self.best,stage,cy,.25,'cases','fallback')
            if self.calls==before: k.evaluate(self.current,dict(role='explicit_noop_retest'))
        self.active_step=None
        self.prune_work()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('pair',choices=PAIRS);p.add_argument('method',choices=METHODS)
    p.add_argument('--section',default='runs');p.add_argument('--additional-calls',type=int)
    a=p.parse_args()
    if a.section=='runs' and a.additional_calls is not None: raise ValueError('No production budget')
    if a.section!='runs' and (a.additional_calls is None or a.additional_calls<1): raise ValueError('Bounded test required')
    verify_frozen()
    cp=read_gz(D/a.section/a.pair/a.method/'checkpoint.json.gz')
    ceiling=cp['calls']+a.additional_calls if a.additional_calls else None
    ReviewedSearch(a.pair,a.method,a.section,ceiling,True).run()
