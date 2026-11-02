"""Extract comparable charged-call milestones from the complete evidence log."""
from bootstrap import *
def milestones(run,summary):
 wanted=[8,72,264,520,1032,2056,4104];out=[];best=None
 for line in (run/'history.jsonl').open(encoding='utf-8'):
  r=json.loads(line)
  if r['accepted_global']:
   best={k:r.get(k) for k in ('candidate_id','plan_id','status','normalized_violation','normalized_violations','approved_cases','expected_cases','failed_cases','FO','VP30','CVaR_VP')}
  if r['evaluation'] in wanted:
   assert best is not None
   out.append(dict(evaluation=r['evaluation'],best=best.copy(),wall_s=r['search_elapsed_seconds'],cpu_s=r['process_cpu_seconds']))
 assert len(out)==len(wanted) and out[-1]['best']['candidate_id']==summary['final']['candidate_id']
 return out
