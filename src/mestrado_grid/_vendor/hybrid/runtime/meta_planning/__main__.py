"""Construct, search and independently audit direct AC planning candidates."""
import argparse
from .catalogue import analytic_study,load_leiria
from .codec import Codec
from .schema import Candidate,read,write_new
from .initialization import initialize
from .evaluator import Evaluator,save_evaluation,audit_saved
from .financial_policy import configure, POLICIES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--catalogue',choices=['analytic','leiria'],default='analytic')
    run.add_argument('--input',help='Optional contractual Leiria input JSON')
    run.add_argument('--candidate',help='Evaluate these saved physical controls instead of initializing')
    run.add_argument('--seed',type=int,default=0);run.add_argument('--mode',choices=['constructive','random','physical','resource'],default='constructive')
    run.add_argument('--output',required=True)
    run.add_argument('--financial-policy',choices=['castro-aligned','vp30',*POLICIES],
                     help='Default: Castro-aligned, unless explicitly set in the input')
    run.add_argument('--physical-config',help='JSON options for the separate physical extension')
    audit=commands.add_parser('audit');audit.add_argument('directory')
    search=commands.add_parser('search');search.add_argument('--config',required=True)
    search.add_argument('--legacy',action='store_true',help='Explicitly use the historical runner')
    search.add_argument('--plan-only',action='store_true',help='Check native configuration and budget without AC or output creation')
    search.add_argument('--study',help='Complete Study JSON for the canonical native route')
    search.add_argument('--policy');search.add_argument('--catalogue',choices=['analytic','leiria'],default='analytic')
    search.add_argument('--input');search.add_argument('--output',required=True)
    search.add_argument('--physical-config',help='JSON options for the separate physical extension')
    search.add_argument('--pause-after',type=int)
    search.add_argument('--financial-policy',choices=['castro-aligned','vp30',*POLICIES],
                        help='Default: Castro-aligned, unless explicitly set in the input')
    audit_search=commands.add_parser('audit-search');audit_search.add_argument('directory')
    args=parser.parse_args()
    if args.command=='search' and not args.legacy:
        import json
        from .execution import plan,run as native_run
        from .schema import Study
        if args.policy or args.pause_after is not None:
            parser.error('Custom policy and legacy pause/resume require --legacy; native physics uses the standard policy')
        if args.study and (args.input or args.physical_config):
            parser.error('--study is a complete input; do not also pass --input or --physical-config')
        if args.input and args.catalogue!='leiria':parser.error('--input requires --catalogue leiria')
        st=Study(**read(args.study)) if args.study else (analytic_study() if args.catalogue=='analytic' else load_leiria(args.input) if args.input else load_leiria())
        if args.physical_config:
            from .physical_catalogue import extend
            st=extend(st,**read(args.physical_config))
        st=configure(st,args.financial_policy)
        cfg=read(args.config)
        if args.plan_only:result=plan(st,cfg)
        else:
            summary=native_run(st,cfg,args.output)
            result=dict(completed=summary['completed'],evaluations=summary['evaluations'],best=summary['best'],financial_policy=summary['financial_policy'])
        print(json.dumps(result,ensure_ascii=False));return
    if args.command=='search' and args.legacy and (args.plan_only or args.study):
        parser.error('--plan-only/--study belong to the canonical native route')
    if args.command in ('search','audit-search'):
        import json
        from pathlib import Path
        from .engines import SearchConfig
        from .evaluator import Policy
        from .runner import run_search,unseal
        from .schema import Study
        if args.command=='audit-search':
            root=Path(args.directory)
            if not (root/'result.json').exists():raise ValueError('Search is not completed; use search to resume')
            protocol=unseal(root/'protocol.json');st=Study(**unseal(root/'study.json'))
            result=run_search(st,SearchConfig(**protocol['config']),root,Policy(**protocol['policy']))
        else:
            if args.input and args.catalogue!='leiria':raise ValueError('--input requires Leiria catalogue')
            st=analytic_study() if args.catalogue=='analytic' else load_leiria(args.input) if args.input else load_leiria()
            if args.physical_config:
                from .physical_catalogue import extend
                st=extend(st,**read(args.physical_config))
            st=configure(st,args.financial_policy)
            result=run_search(st,SearchConfig(**read(args.config)),args.output,Policy(**read(args.policy)) if args.policy else None,args.pause_after)
        print(json.dumps(dict(status=result['status'],evaluations=result['evaluations'],
            feasible=result.get('best_feasible') is not None,
            financial_policy=result['financial_policy'],
            selection_objective=(result.get('audited_metrics') or {}).get('selection_objective'),
            objective_VP=(result.get('audited_metrics') or {}).get('objective_VP'))))
        return
    if args.command=='audit':
        result=audit_saved(args.directory)
    else:
        from pathlib import Path
        if Path(args.output).exists():raise FileExistsError('Choose an unused output directory')
        if args.input and args.catalogue!='leiria':raise ValueError('--input is only valid for the Leiria adapter')
        st=analytic_study() if args.catalogue=='analytic' else load_leiria(args.input) if args.input else load_leiria()
        if args.physical_config:
            from .physical_catalogue import extend
            st=extend(st,**read(args.physical_config))
        st=configure(st,args.financial_policy)
        codec=Codec(st)
        if args.candidate:
            candidate=Candidate.from_dict(read(args.candidate));origin=dict(origin='explicit_physical_candidate',path=args.candidate)
        else:
            candidate,genome,origin=initialize(codec,args.seed,args.mode)
        result=Evaluator(st).evaluate(candidate)
        save_evaluation(args.output,st,candidate,result)
        write_new(Path(args.output)/'construction.json',dict(origin=origin,layout=codec.describe()))
    metrics=result.get('metrics')
    import json
    print(json.dumps(dict(status=result['status'],approved_cases=result.get('approved_cases',0),
                         financial_policy=result['financial_policy'],
                         selection_objective=metrics['selection_objective'] if metrics else None,
                         objective_VP=metrics['objective_VP'] if metrics else None,
                         wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
