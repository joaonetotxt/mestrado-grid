"""Paired finite comparisons through the canonical entry point; no auto-resume."""
import copy
from pathlib import Path
import numpy as np
from .execution import run,plan,resolve_options
from .prepared_search import fresh_population,write,read,digest
from .search_arrays import ArrayCodec
from .schema import Candidate
from . import economics,physical


def summarize(st,folder,result):
    c=Candidate.from_dict(read(Path(folder)/'best_candidate.json'));r=result['best']
    m=r.get('metrics');out=dict(status=r['status'],approved_cases=r['approved_cases'],
        candidate_id=c.candidate_id,plan_id=c.plan_id,evaluations=result['evaluations'])
    if r['status']=='feasible':
        w=economics.weights(st);weights=st.c['annual_hours']*(w/w.sum())[:,None]*st.probabilities[None,:]
        old=np.array([g['technology'] in ('PV','WT') for g in st.old],bool)
        new=np.array([g['technology'] in ('PV','WT') for g in st.new],bool)
        old_hi=physical.existing_bounds(st,c)[1] if c.physical is not None else st.existing_bounds()[1]
        new_hi=st.active_capacity(c)[:,None,:]*physical.availability(st)[None,:,:]
        curtailed=np.maximum(0,old_hi[:,:,old]-c.operation.existing_p[:,:,old]).sum(2)
        curtailed+=np.maximum(0,new_hi[:,:,new]-c.operation.new_p[:,:,new]).sum(2)
        out.update(FO=m['selection_objective'],VP30=m['objective_VP'],CVaR_total=m['finance']['CVaR_total'],
            cut_MWh_year=m['cut_equivalent_MWh'],renewable_curtailment_MWh_year=float(np.sum(weights*curtailed)),
            FO_gain=result['initial_best']['metrics']['selection_objective']-m['selection_objective'] if result['initial_best']['status']=='feasible' else None)
    out['coverage']=result['persistent_search']['effective_coverage']
    out['continuation_calls']=result['persistent_search']['continuation_calls']
    out['measurements']=len(result['persistent_search'].get('guidance',{}).get('measurements',[]))
    return out


def compare(study,config,methods,seeds,variants,output):
    """All arms share the same population per seed and exact call count.

    Variants are explicit PersistentOptions overrides, frozen before first run.
    Different guidance changes subsequent trajectories; this is paired empirical
    comparison, not an isolated decomposition of descendants' causal ancestry.
    """
    root=Path(output)
    if root.exists():raise FileExistsError('Comparison output exists; no automatic restart')
    if not methods or not seeds or not variants:raise ValueError('Empty comparison')
    if len(set(methods))!=len(methods) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate methods/seeds')
    import re
    if any(not re.fullmatch('[a-z][a-z0-9_]*',name) for name in variants):raise ValueError('Invalid variant name')
    configs=[]
    for seed in seeds:
        for method in methods:
            for name,override in variants.items():
                cfg=copy.deepcopy(config);cfg.update(seed=seed,method=method)
                cfg['persistent_options']=dict(resolve_options(cfg.get('persistent_options')),**override)
                plan(study,cfg)  # No AC or output before validating the whole matrix.
                configs.append((seed,method,name,cfg))
    root.mkdir(parents=True)
    write(root/'comparison_protocol.json',dict(config=config,methods=methods,seeds=seeds,variants=variants,
        study_id=study.input_id,stop='Exactly the declared matrix; no restart or extension'))
    starts={};rows=[];codec=ArrayCodec(study)
    for seed,method,name,cfg in configs:
        if seed not in starts:
            starts[seed]=(np.load(cfg['initial_population'],allow_pickle=False) if cfg.get('initial_population')
                else fresh_population(codec,seed,cfg['population'],cfg['initialization']))
            np.save(root/f'initial_{seed}.npy',starts[seed],allow_pickle=False)
        cfg.pop('initialization',None);cfg['initial_population']=str((root/f'initial_{seed}.npy').resolve())
        folder=root/f'{method}_{seed}_{name}'
        result=run(study,cfg,folder)
        assert result['initial_population_sha256']==digest(starts[seed])
        rows.append(dict(method=method,seed=seed,variant=name,**summarize(study,folder,result)))
        write(root/'comparison.json',dict(completed=False,rows=rows))
    write(root/'comparison.json',dict(completed=True,rows=rows))
    return rows
