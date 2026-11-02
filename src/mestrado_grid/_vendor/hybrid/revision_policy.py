"""Isolated planning policy. No physical acceptance, objective or risk changes."""
import copy,json,collections

def pack_observation(data):
    counts=collections.Counter()
    def count(x):
        if isinstance(x,list):
            key=json.dumps(x,separators=(',',':'),ensure_ascii=False)
            if len(key)>=80:counts[key]+=1
            for v in x:count(v)
        elif isinstance(x,dict):
            for v in x.values():count(v)
    count(data); tables={};ids={}
    def pack(x):
        if isinstance(x,list):
            key=json.dumps(x,separators=(',',':'),ensure_ascii=False)
            if counts[key]>1:
                if key not in ids:
                    name='t'+str(len(ids));ids[key]=name
                    tables[name]=[pack(v) for v in x]
                return {'table_ref':ids[key]}
            return [pack(v) for v in x]
        if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
        return x
    obs=pack(data)
    return dict(encoding='Every {table_ref: ID} means the exact array in shared_tables[ID]. References may be nested. No rows, locations, options or numbers were removed; original indices and order are preserved.',shared_tables=tables,observation=obs)

def unpack_observation(data):
    def unpack(x):
        if isinstance(x,dict):
            if set(x)=={'table_ref'}:return unpack(data['shared_tables'][x['table_ref']])
            return {k:unpack(v) for k,v in x.items()}
        if isinstance(x,list):return [unpack(v) for v in x]
        return x
    return unpack(data['observation'])

class ReviewGate:
    def __init__(self,max_calls=64,max_seconds=600,min_calls=16,relative_change=.05):
        self.max_calls=max_calls;self.max_seconds=max_seconds;self.min_calls=min_calls;self.relative_change=relative_change
        self.last_call=None;self.last_time=None;self.last_fo=None
    def reason(self,calls,seconds,fo,pending):
        if self.last_call is None:return 'initial_current_state_review'
        if not pending:return 'queue_finished'
        dc=calls-self.last_call
        if dc>=self.max_calls:return 'evaluation_interval'
        if seconds-self.last_time>=self.max_seconds:return 'wall_interval'
        if dc>=self.min_calls and abs(fo-self.last_fo)/max(1.,abs(self.last_fo))>=self.relative_change:return 'incumbent_changed'
        return None
    def reviewed(self,calls,seconds,fo):self.last_call=calls;self.last_time=seconds;self.last_fo=fo

def frontier(kernel):
    result=[]
    for key,w in kernel.work.items():
        def short(r):
            return dict(candidate_id=r['candidate_id'],status=r['status'],FO=r['metrics']['selection_objective'],
                        nonzero_violations={k:v for k,v in r['normalized_violations'].items() if v!=0})
        result.append(dict(experiment=key,control=short(w['cr']),asset=short(w['ar']) if w['ar'] else None,
                           completed=w['completed_cycles'],remaining=len(w['cycles']),actions=w['proposal']['actions']))
    return result

EXTRA_PROMPT='''This is a rolling review of an ongoing search. Production search has no fixed total evaluation or time ceiling; this isolated diagnostic trial has only the explicit remaining evaluations given in its observation.
You are recalled after at most 64 new full evaluations or 600 seconds, or after a 5% incumbent cost change with at least 16 evaluations. Do not wait for all old experiments to finish.
Choose short actionable experiments from the CURRENT incumbent or measured own-arm sources. Start with one round and one or two complete days when useful, and revise from evidence. You may still use all supported actions, locations, modes, 1..12 rounds, and up to eight proposals. A longer plan is not a commitment to execute it before review.
State-wise P/Q and full-cycle SOC/DR/EV are complementary; compare the current bottlenecks and select the appropriate mode. The original frozen evaluator checks all 96 states and entire daily cycles even for a local step. No reliability or economic relaxation.
Previous unfinished branches are preserved, not erased or declared useless. Work sources listed in continuation_sources can be selected by their candidate IDs to continue their actual evolved candidates. A proposal's source is bound to that exact candidate; never silently transplant it to a different network.
Use measured control-versus-resource gains and full costs. Do not credit dispatch improvement of the no-resource control to the resource. Propose large investments only as hypotheses, not as guaranteed gains. Archive case_violations identify the actual failed states for repair.
Repeated arrays may be losslessly represented by table_ref; expand using shared_tables. All catalogue options remain accessible. No other arm's results are supplied.'''
