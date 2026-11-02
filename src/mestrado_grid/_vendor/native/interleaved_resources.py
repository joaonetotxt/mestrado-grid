"""Deterministic allocation policy; no evaluation/decoding changes.

Pair starts and scenario regimes use the same phase for each engine/beta.
This changes finite-budget exploration opportunities, not the native proposal
operators. No candidates, fitness or native memories cross streams.
"""
from resource_coverage import ResourceScheduler
from effective_scheduler import EffectiveScheduler
from meta_planning.engines import METHODS
from meta_planning import persistent_search as _persistent
_original_factory=_persistent.make_scheduler

VERSION='phased_resources_interleaved_v1_release'

def phase_for(method,beta,count):
    if method not in METHODS:raise ValueError('Unknown method')
    b=round(float(beta)*10)
    if not 0<=b<=10 or abs(b/10-float(beta))>1e-12:raise ValueError('Unknown beta class')
    return (b*len(METHODS)+METHODS.index(method))%count

class InterleavedResourceScheduler(ResourceScheduler):
    def bind_stream(self,method,beta):
        if hasattr(self,'allocation_version'):raise ValueError('Already configured; do not reset live policy state')
        self.allocation_version=VERSION;self.stream_method=method;self.stream_beta=float(beta)
        self.resource_phase=phase_for(method,beta,len(self.initial_order))
        self.interleaved_order=self.initial_order[self.resource_phase:]+self.initial_order[:self.resource_phase]
        completed={v['task'] for v in self.resource_visits}
        self.initial_pending=[t for t in self.interleaved_order if t not in completed]
        self.allocation_step=0;self.post_sweep_step=0;self.repeat_cursor=0;self.allocation_trace=[]
        return self

    def choose(self,reference,contexts):
        if not hasattr(self,'allocation_version'):raise ValueError('Explicit engine/beta identity required')
        step=self.allocation_step;self.allocation_step+=1
        if self.initial_pending:
            resource=step%3!=2
        else:
            resource=self.post_sweep_step%2==0;self.post_sweep_step+=1
        if not resource:
            self.allocation_trace.append(dict(step=step,kind='effective_or_original'))
            return EffectiveScheduler.choose(self,reference,contexts)
        if self.initial_pending:task=self.initial_pending.pop(0)
        else:
            task=self.interleaved_order[self.repeat_cursor%len(self.interleaved_order)];self.repeat_cursor+=1
        b=self.resource_block(reference,task);self.resource_cursor+=1
        b.metadata.update(allocation_version=VERSION,stream_phase=self.resource_phase,allocation_step=step)
        self.allocation_trace.append(dict(step=step,kind='resource',task=task))
        key=tuple(b.metadata['resource_context_key']);number=self.resource_contexts.get(key)
        if number is not None:
            context=contexts[number]
            if tuple(context.block.metadata['resource_context_key'])!=key:raise ValueError('Resource context identity mismatch')
            return context,None
        return None,b

    def report(self):
        out=super().report();out['protected_resources']['initial_opportunity_sweep_complete']=not self.initial_pending
        out['protected_resources']['order']=self.interleaved_order
        out['protected_resources']['maximum_resource_visits_between_same_family']=len(self.interleaved_order)
        out['protected_resources']['maximum_total_visits_per_recurring_family_cycle']=2*len(self.interleaved_order)
        out['allocation_policy']=dict(version=VERSION,phase=self.resource_phase,order=self.interleaved_order,
            initial_pending=list(self.initial_pending),steps=self.allocation_step,
            warmup_resource_fraction='2/3',after_warmup_resource_fraction='1/2',
            warmup_sequence=['resource','resource','effective_or_original'],
            original_scheduler_continues_its_saved_effective_step=True,
            trace=self.allocation_trace,scope='Early opportunities; no guarantee of accepted investment, FO improvement or exhaustive sites',
            unequal_first_families_declared=True,cross_stream_candidate_or_memory_transfer=False)
        return out

def migrate_scheduler_at_complete_boundary(scheduler,method,beta):
    """Preserve all existing fields/native contexts and add allocation policy.

    The caller must establish the complete-visit boundary. This does not read,
    write, pause or resume any production checkpoint.
    """
    if type(scheduler) is not ResourceScheduler:raise ValueError('Unsupported scheduler/version')
    scheduler.__class__=InterleavedResourceScheduler
    return scheduler.bind_stream(method,beta)

def install_for_isolated_run(method,beta):
    from meta_planning import persistent_search as p
    def factory(codec,seed,mass,options):
        if options.protected_resource_coverage:
            return InterleavedResourceScheduler(codec,seed,mass,options).bind_stream(method,beta)
        return _original_factory(codec,seed,mass,options)
    p.make_scheduler=factory
