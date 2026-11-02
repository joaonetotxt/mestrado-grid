"""Run small physical integration tests using the installed distribution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from mestrado_grid.cli import METHODS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--with-ai',action='store_true',help='Include real authenticated Codex inference (uses service quota)')
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    args=parser.parse_args()
    methods=args.methods or list(METHODS if args.with_ai else METHODS[:-1])
    if 'HYBRID_AI' in methods and not args.with_ai:
        parser.error('Use --with-ai to include real inference.')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-X','utf8','-B','-m','mestrado_grid']
    subprocess.run(command+['prepare-demo','--output',str(root/'demo'),'--beta','0.5'],check=True)
    results=[]
    for method in methods:
        out=root/method
        completed=subprocess.run(command+['run','--method',method,'--study',str(root/'demo/study.json'),
            '--population',str(root/'demo/population.npy'),'--output',str(out),'--budget','88',
            '--regime','normal','--case-kind','synthetic','--starting-point','synthetic_original_tree'],
            capture_output=True,text=True,encoding='utf-8')
        if completed.returncode:
            print(completed.stderr,file=sys.stderr)
            raise RuntimeError('Integration test failed: '+method)
        r=json.loads((out/'result/summary.json').read_text(encoding='utf-8'))
        final,meter=r['final'],r['meter']
        assert r['completed'] and r['fresh_audit_matches'] and r['finance']['all_pass']
        assert final['status']=='feasible' and final['approved_cases']==final['expected_cases']==96
        assert final['normalized_violation']==0 and all(v==0 for v in final['normalized_violations'].values())
        assert meter['search_calls']==88 and meter['search_AC_cases']==8448
        assert meter['independent_audit_calls']==1 and meter['independent_audit_AC_cases']==96
        if method not in ('HYBRID','HYBRID_AI'):
            assert r['counters']['native_offspring']>0 and meter['LP_solves']==0
        else:
            assert meter['LP_solves']>0
        if method=='HYBRID_AI':
            assert r['AI']['accepted']>0 and r['AI']['tool_events']==0
        results.append(dict(method=method,status='passed',final=final,meter=meter,AI=r['AI']))
        print(method+': passed',flush=True)
    (root/'integration.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
