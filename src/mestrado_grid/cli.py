"""Portable, finite-budget entry points. Each search owns a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

METHODS = ('PSO', 'HyDE', 'GA', 'ACO', 'Vortex', 'QPSO', 'SHADE', 'L-SHADE', 'HYBRID', 'HYBRID_AI')
VENDOR = Path(__file__).resolve().parent / '_vendor'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((VENDOR/'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        if digest(VENDOR/relative) != expected:
            raise ValueError('Scientific source integrity check failed: ' + relative)
    return len(manifest['files'])


def native_runtime():
    sys.path.insert(0, str(VENDOR/'native/runtime'))


def prepare_demo(output, beta, seed):
    """A standalone synthetic case with eight daily cycles and 96 AC cases."""
    import copy
    import numpy as np
    native_runtime()
    from meta_planning.catalogue import analytic_study
    from meta_planning.schema import Study, plain
    from meta_planning.physical_catalogue import extend
    from meta_planning.search_arrays import ArrayCodec
    from meta_planning.initialization import initialize
    base = analytic_study()
    d, el = copy.deepcopy(base.data), copy.deepcopy(base.electrical)
    d['config']['beta'] = beta
    d['config']['storage_scenario_cycles'] = [[s+8*j for s in cy] for j in range(4) for cy in base.cycles]
    d['config']['scenario_block_hours'] *= 4
    for name in ('operations', 'availability'):
        d[name] = [dict(row, scenario=row['scenario']+8*j) for j in range(4) for row in base.data[name]]
    d['scenarios'] = [dict(row, id=row['id']+8*j, probability=row['probability']/4) for j in range(4) for row in base.data['scenarios']]
    for g in d['extension']['generators']:
        old = g['availability']
        g['availability'] = {str(s): old[str(s % 8)] for s in range(32)}
    el['q_ratio'] = [row*4 for row in el['q_ratio']]
    st = extend(Study(d, el))
    codec = ArrayCodec(st)
    candidates = [initialize(codec, seed+i, 'mat_original')[0] for i in range(8)]
    population = np.asarray([codec.encode(c) for c in candidates])
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'study.json', plain(dict(data=st.data, electrical=st.electrical)))
    np.save(output/'population.npy', population, allow_pickle=False)
    dump(output/'provenance.json', dict(kind='synthetic_integration_case', buses=st.B, stages=st.K, states=st.S,
        expected_AC_cases=st.K*st.S, population=8, unique_candidates=len({c.candidate_id for c in candidates}),
        seed=seed, beta=beta, alpha=st.c['alpha'], study_id=st.input_id,
        study_sha256=digest(output/'study.json'), population_sha256=digest(output/'population.npy'),
        initialization='original synthetic tree and existing resources; physical initialization; no optimized historical vector',
        calibrated_network=False))
    print(json.dumps(dict(output=str(output), members=8, dimension=codec.dimension, study_id=st.input_id)))


def run(args):
    import numpy as np
    study_path, population_path, output = args.study.resolve(), args.population.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing runs are never overwritten: ' + str(output))
    if args.method == 'HYBRID_AI' and not shutil.which('codex'):
        raise ValueError('HYBRID_AI requires an installed, authenticated Codex CLI. No fallback is used.')
    if args.budget < 32:
        raise ValueError('Budget must be at least 32 complete evaluations, including eight initial members.')
    value = json.loads(study_path.read_text(encoding='utf-8-sig'))
    native_runtime()
    from meta_planning.schema import Study
    from meta_planning.search_arrays import ArrayCodec
    st = Study(**value)
    beta = st.c['beta']
    if st.K*st.S != 96 or st.c['alpha'] != .9 or not 0 <= beta <= 1 or abs(beta*10-round(beta*10)) > 1e-10:
        raise ValueError('This reference executor supports 96 AC cases, alpha=0.9, beta=0..1 in steps of 0.1.')
    codec = ArrayCodec(st)
    pop = np.load(population_path, allow_pickle=False)
    if pop.shape != (8, codec.dimension) or not np.isfinite(pop).all() or (pop < 0).any() or (pop > 1).any():
        raise ValueError('Population must contain eight finite, normalized genomes matching this study.')
    for x in pop:
        codec.decode(x)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'
    inputs.mkdir()
    beta_class = round(beta*10)
    shutil.copyfile(study_path, inputs/f'beta_{beta_class:02d}.json')
    shutil.copyfile(population_path, inputs/'sim1.npy')
    dump(output/'job.json', dict(job=[1,beta_class,args.seed,args.method], budget=args.budget,
        regime=args.regime, case_kind=args.case_kind, starting_point=args.starting_point,
        study_id=st.input_id, population_sha256=digest(population_path),
        method_memory='fresh', candidate_cache=False, case_cache=False, independent_final_audits=1))
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'mestrado_grid.worker', str(output)]
    with (output/'worker.stdout.log').open('w', encoding='utf-8') as stdout, (output/'worker.stderr.log').open('w', encoding='utf-8') as stderr:
        completed = subprocess.run(command, cwd=output, env=env, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    verify()
    if completed.returncode:
        print('Search failed; inspect ' + str(output/'worker.stderr.log'), file=sys.stderr)
        return completed.returncode
    summary = json.loads((output/'result/summary.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(method=args.method, completed=summary['completed'], final=summary['final'],
        meter=summary['meter'], output=str(output)), ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Planejamento de redes: dez métodos, avaliação AC e orçamento explícito.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('methods', help='List the ten implemented methods')
    sub.add_parser('verify', help='Verify installed scientific source hashes')
    demo = sub.add_parser('prepare-demo', help='Create an independent synthetic integration case')
    demo.add_argument('--output', required=True, type=Path)
    demo.add_argument('--beta', choices=[i/10 for i in range(11)], type=float, default=.5)
    demo.add_argument('--seed', type=int, default=96101)
    search = sub.add_parser('run', help='Run one method in a new directory, with a finite evaluation budget')
    search.add_argument('--method', choices=METHODS, required=True)
    search.add_argument('--study', type=Path, required=True)
    search.add_argument('--population', type=Path, required=True)
    search.add_argument('--output', type=Path, required=True)
    search.add_argument('--budget', type=int, required=True)
    search.add_argument('--seed', type=int, default=96101)
    search.add_argument('--regime', choices=('normal','extremo'), required=True, help='Label of scenarios already encoded in the study; does not change input data')
    search.add_argument('--case-kind', choices=('synthetic','research'), default='research')
    search.add_argument('--starting-point', default='user_supplied_population', help='Provenance of the supplied population')
    args = parser.parse_args(argv)
    try:
        count = verify()
        if args.command == 'methods':
            print('\n'.join(METHODS))
        elif args.command == 'verify':
            print(json.dumps(dict(verified_files=count)))
        elif args.command == 'prepare-demo':
            prepare_demo(args.output.resolve(), args.beta, args.seed)
        else:
            return run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    return 0
