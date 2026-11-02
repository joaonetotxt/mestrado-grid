"""Persistent production lifecycle around the unchanged isolated analytical kernel."""
import production_io
from run_search import Search, PairStopped, SmokeComplete
from campaign_support import *
from analytic_core import RepairPortfolio, CycleCoverage, observation
from complete_modes import CompleteKernel
from bound_analyst import BoundAnalyst
from run_trial import proposal, deterministic
from allocation_ai import InferenceInterrupted
from archive_reproduction import archive_roundoff_only
from collections import Counter
import types


def encode_work(work):
    result = copy.deepcopy(work)
    for row in result.values():
        for key in ('control', 'asset'):
            if row[key] is not None:
                row[key] = row[key].to_dict()
    return result


def decode_work(work):
    result = copy.deepcopy(work)
    for row in result.values():
        for key in ('control', 'asset'):
            if row[key] is not None:
                row[key] = Candidate.from_dict(row[key])
    return result


class AnalyticSearch(Search):
    def __init__(self, pair, method, section='runs', smoke_calls=None, resume=True):
        self.kernel = None
        self.queue = []
        self.analytic_visit = 0
        self.active_step = None
        self.loaded_analytic = None
        self.legacy_extension = None
        self.legacy_ai_requests = 0
        self.ai_previous_wait = 0.
        self.ai_previous_usage = []
        self.migration_source = None
        super().__init__(pair, method, section, smoke_calls, resume)
        if self.phase != 'strict':
            raise ValueError('Migration requires the already-transitioned original strict context')
        k = CompleteKernel.__new__(CompleteKernel)
        k.st, k.out, k.budget = self.st, self.out, float('inf')
        k.ev, k.ref = self.ev, self.ref
        k.ev.calls, k.ev.ac_cases = self.calls, self.ac_cases
        k.best, k.current = self.best, self.current
        k.bank = RepairPortfolio()
        k.coverage = CycleCoverage()
        k.feedback, k.counts, k.work, k.pinned_sources = [], Counter(), {}, {}
        k.started, k.cpu, k.seen = time.monotonic(), time.process_time(), {}
        k.evaluate = self.kernel_evaluate
        k.progress = self.progress
        self.kernel = k
        CompleteKernel.active_instance = k
        if self.loaded_analytic:
            self.restore_analytic(self.loaded_analytic)
        elif self.legacy_extension:
            k.bank = RepairPortfolio.restore(self.legacy_extension['portfolio'])
        if not k.bank.entries:
            k.bank.offer(self.current, self.best, dict(kind='own_checkpoint_import'))
        self.audit_sources()
        self.sup = BoundAnalyst(self.out/'ai_analytic') if method == 'HYBRID_AI' else None
        self.has_ai = bool(self.sup and self.queue)
        save(self.out/'analytic_restore_verified.json', dict(utc=now(), pid=os.getpid(),
             calls=self.calls, candidate_id=self.current.candidate_id,
             input_id=self.strict.input_id, result=compact(self.best), phase=self.phase,
             archive_entries=len(k.bank.entries), pending_experiments=len(self.queue),
             working_experiments=len(k.work), previous_analytic_version=bool(self.loaded_analytic),
             migration_source=self.migration_source, automatic_notifications=False))
        self.checkpoint()

    def restore(self):
        cp = read_gz(self.out/'checkpoint.json.gz')
        self.loaded_analytic = cp.get('analytic_extension')
        self.legacy_extension = cp.get('extension')
        if self.loaded_analytic:
            if self.loaded_analytic['version'] != 1:
                raise ValueError('Unknown analytic checkpoint')
        migration = self.out/'migration.json'
        if migration.exists():
            self.migration_source = read(migration)
            self.legacy_ai_requests = self.migration_source.get('legacy_AI_requests', 0)
        super().restore()

    def restore_analytic(self, ext):
        k = self.kernel
        k.bank = RepairPortfolio.restore(ext['portfolio'])
        k.coverage.visits = Counter({(x['stage'], tuple(x['states'])): x['visits'] for x in ext['coverage']})
        k.feedback, k.counts = ext['feedback'], Counter(ext['counts'])
        k.work = decode_work(ext['work'])
        for item in ext['pinned']:
            c = Candidate.from_dict(item['candidate'])
            k.pin_source(c, item['record'])
        self.queue = [(x[0], x[1]) for x in ext['queue']]
        self.analytic_visit = ext['visit']
        self.active_step = ext.get('active_step')
        self.ai_previous_wait = ext['AI_wait_s']
        self.ai_previous_usage = ext['AI_usage']
        self.legacy_ai_requests = ext['legacy_AI_requests']

    def audit_sources(self):
        """Audit every distinct restored working/archive candidate without cache."""
        sources = [(c, r) for c, r, _ in self.kernel.bank.entries.values()]
        sources += [(c, r) for c, r, _ in self.kernel.pinned_sources.values()]
        for work in self.kernel.work.values():
            sources += [(work['control'], work['cr'])]
            if work['asset'] is not None:
                sources.append((work['asset'], work['ar']))
        seen = {self.current.candidate_id}
        for c, record in sources:
            if c.candidate_id != record['candidate_id'] or record['signature'] != self.ev.signature:
                raise ValueError('Restored candidate/context identity mismatch')
            if c.candidate_id in seen:
                continue
            fresh = self.fresh(c, self.st, 'analytic_resume_source')
            if not audit_agrees(fresh, record):
                if not archive_roundoff_only(fresh, record):
                    raise RuntimeError('Restored alternative failed independent audit')
                append(self.out/'archive_roundoff.jsonl', dict(utc=now(), candidate_id=c.candidate_id,
                       saved=compact(record), fresh=compact(fresh), scope='infeasible restoration only'))
            seen.add(c.candidate_id)

    def progress(self, state='running', **extra):
        k = self.kernel
        super().progress(state, version='physical_analyst_v1',
             analytic_visit=self.analytic_visit, pending_experiments=len(self.queue),
             portfolio_size=len(k.bank.entries) if k else 0,
             analytic_counters=dict(k.counts) if k else {},
             legacy_AI_requests=self.legacy_ai_requests, **extra)

    def kernel_evaluate(self, c, event):
        r = super().evaluate(c, event)
        self.kernel.best, self.kernel.current = self.best, self.current
        self.kernel.bank.offer(c, r, event | dict(call=self.calls))
        return r

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
                  legacy_AI_requests=self.legacy_ai_requests)
        save_gz(self.out/'checkpoint.json.gz', dict(version=3, pair=self.pair, method=self.method,
            phase=self.phase, calls=self.calls, ac_cases=self.ac_cases, audit_calls=self.audit_calls,
            audit_ac=self.audit_ac, visit=self.visit, day_cursor=self.day_cursor, init_index=self.init_index,
            counts=dict(self.counts), best=self.best, current=self.current.to_dict() if self.current else None,
            initial=self.initial, recent=list(self.recent), pairs=list(self.pairs), order=self.order,
            radius=self.radius, next_ai=self.next_ai, has_ai=self.has_ai, used=list(self.used),
            last_audited=self.last_audited, last_periodic=self.last_periodic,
            wall=self.previous_wall+time.monotonic()-self.clock,
            cpu=self.previous_cpu+time.process_time()-self.cpu, strict_input_id=self.strict.input_id,
            analytic_extension=ext, legacy_extension_preserved_in_parent=True))

    def decide_analytic(self):
        if not self.sup or self.queue:
            return
        k = self.kernel
        # Completed plans' snapshots may be discarded; active work is always retained.
        if not k.work:
            k.pinned_sources.clear()
        while True:
            self.check()
            obs = observation(self.st, self.current, self.best, k.bank, k.feedback, k.coverage,
                              None if self.smoke_calls is None else self.smoke_calls-self.calls)
            self.progress('waiting_for_ai')
            try:
                decision = self.sup.choose(obs, self.stopped, lambda:self.progress('waiting_for_ai'))
                self.queue = [(f'AI_{self.sup.calls}_{i}', p) for i,p in enumerate(decision['proposals'])]
                self.has_ai = True
                self.checkpoint()
                return
            except InferenceInterrupted:
                raise PairStopped()
            except Exception as exc:
                append(self.out/'inference_errors.jsonl', dict(utc=now(), calls=self.calls, error=str(exc),
                       synthetic_policy_used=False))
                # No fabricated policy or silent switch to the non-AI treatment.
                retry = time.monotonic()+60
                while time.monotonic() < retry:
                    self.check(); self.progress('waiting_for_ai_retry'); time.sleep(5)

    def step(self):
        self.check()
        if self.active_step is None:
            self.decide_analytic()
        k = self.kernel
        before = self.calls
        if self.active_step is not None:
            p, owner, experiment, work_id = (self.active_step[x] for x in ('proposal','owner','experiment','work_id'))
        else:
            self.analytic_visit += 1; self.visit += 1
            v = self.analytic_visit
            experiment = None
            if v % 4 == 1:
                stage, cy = k.coverage.select(self.st, self.best)
                p = proposal(self.st, 'incumbent', [(stage,cy)], mode='cycle' if v%2 else 'cases')
                owner = 'shared_cycle_coverage'
            elif v % 4 == 2:
                p = deterministic(k, v, self.seed, directed=False); owner='shared_resource_coverage'
            elif self.sup:
                experiment, p = self.queue[0]; owner='AI_physical_proposal'
            else:
                p = deterministic(k, v, self.seed); owner='deterministic_physical_proposal'
            work_id = experiment or f'visit_{v}'
            self.active_step = dict(proposal=p,owner=owner,experiment=experiment,work_id=work_id)
            append(self.out/'proposals.jsonl', dict(call=self.calls, source=owner, proposal=p))
        try:
            done = k.execute(p, owner, work_id)
            if experiment is not None:
                self.queue.pop(0)
                if not done:
                    self.queue.append((experiment,p))
        except (ValueError, KeyError, IndexError) as exc:
            k.counts['rejected_proposals'] += 1
            append(self.out/'proposal_rejections.jsonl', dict(call=self.calls, proposal=p, error=str(exc)))
            if experiment is not None:
                self.queue.pop(0)
            k.work.pop(work_id, None)
        if self.calls == before:
            stage, cy = k.coverage.select(self.st, self.best)
            k.refine(copy.deepcopy(self.current), self.best, stage, cy, .25, 'cases', 'fallback')
            if self.calls == before:
                k.evaluate(self.current, dict(role='explicit_noop_retest'))
        self.active_step = None

    def run(self):
        try:
            # A migrated checkpoint that already meets the target must be audited now.
            if strict_success(self.best, self.phase):
                self.publish(self.current,self.best)
            return super().run()
        except PairStopped:
            # Covers the immediate target audit above; normal loop owns its own finalization.
            self.progress('pair_target' if (self.shared/'WINNER.json').exists() else 'user_stop')
        finally:
            self.checkpoint()


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('pair',choices=PAIRS); p.add_argument('method',choices=METHODS)
    p.add_argument('--section',default='runs'); p.add_argument('--additional-calls',type=int)
    a=p.parse_args()
    if a.section == 'runs' and a.additional_calls is not None:
        raise ValueError('Production has no call or time budget')
    if a.section != 'runs' and (a.additional_calls is None or a.additional_calls < 1):
        raise ValueError('Isolated integration requires a bounded budget')
    verify_frozen()
    cp=read_gz(D/a.section/a.pair/a.method/'checkpoint.json.gz')
    ceiling=cp['calls']+a.additional_calls if a.additional_calls else None
    AnalyticSearch(a.pair,a.method,a.section,ceiling,True).run()
