"""Check the review package without running simulations or accessing the network."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
errors = []
EXCLUDED = {'__pycache__', '.git', '.pytest_cache', '.venv', 'venv', 'build', 'dist'}


def packaged(path):
    relative = path.relative_to(ROOT)
    if relative.as_posix() in {'PKG-INFO', 'setup.cfg'}:
        return False
    return not any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts)

def require(condition, message):
    if not condition:
        errors.append(message)

def load(relative):
    path = ROOT / relative
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        errors.append(f'{relative}: {exc}')
        return {}

def verify():
    markdowns = [p for p in ROOT.rglob('*.md') if packaged(p)]
    links = 0
    for path in markdowns:
        text = path.read_text(encoding='utf-8')
        require(not re.search(r'[A-Za-z]:[\\/]Users[\\/]', text),
                f'Caminho pessoal em {path.relative_to(ROOT)}')
        for target in re.findall(r'!?\[[^\]\n]*\]\(([^\n]+?)\)', text):
            target = target.strip().strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            local = unquote(target.split('#', 1)[0])
            if not local:
                continue
            dest = (path.parent / local).resolve()
            require(dest.is_relative_to(ROOT), f'Link fora do pacote: {target}')
            require(dest.exists(), f'Link ausente em {path.relative_to(ROOT)}: {target}')
            links += 1

    p = load('evidencias/protocolo_comparacao.json')
    d = load('evidencias/recorte_pareado.json')
    rows = d.get('rows', [])
    require(len(rows) == 10, 'Recorte deve conter dez métodos.')
    require({r.get('method') for r in rows} == set(p.get('methods', [])), 'Métodos inconsistentes.')
    for key in ['population_sha256', 'study_id']:
        require(len({r.get(key) for r in rows}) == 1 and all(r.get(key) for r in rows),
                f'{key} precisa coincidir entre métodos.')
    require(len({r.get('initial', {}).get('candidate_id') for r in rows}) == 1,
            'Identidade inicial difere entre métodos.')
    for r in rows:
        method = r.get('method')
        f, m = r.get('final', {}), r.get('meter', {})
        require(r.get('budget') == m.get('search_calls') == p.get('calls_per_trial') == 4104,
                f'{method}: orçamento incorreto.')
        require(f.get('status') == 'feasible' and f.get('approved_cases') == f.get('expected_cases') == 96,
                f'{method}: final não aprovado em 96/96.')
        violations = f.get('normalized_violations', {})
        require(bool(violations) and {'SAIDI','SAIFI','P','Q','voltage','SOC_balance'} <= set(violations)
                and f.get('normalized_violation') == 0 and all(v == 0 for v in violations.values()),
                f'{method}: violações ausentes ou diferentes de zero.')
        require(m.get('independent_audit_calls') == 1 and m.get('independent_audit_AC_cases') == 96,
                f'{method}: auditoria adicional incorreta.')
        require(m.get('candidate_cache') is False and m.get('case_cache') is False,
                f'{method}: cache contradiz protocolo.')
        require(all(re.fullmatch(r'[0-9a-f]{64}', str(f.get(k, ''))) for k in ['candidate_id','plan_id']),
                f'{method}: identificação inválida.')
        try:
            beta = d['beta']
            expected = f['PC1'] + (1-beta)*f['expected_operation'] + beta*f['CVaR_operation']
            require(math.isclose(f['FO'], expected, rel_tol=1e-12, abs_tol=1e-7),
                    f'{method}: composição da FO inconsistente.')
        except (KeyError, TypeError):
            errors.append(f'{method}: composição financeira ausente.')

    manifest = load('MANIFESTO_DOCUMENTACAO.json')
    declared = manifest.get('files', {})
    require(bool(declared), 'Manifesto vazio.')
    for relative, expected in declared.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'Arquivo do manifesto ausente: {relative}')
        else:
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'],
                    f'Hash divergente: {relative}')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'MANIFESTO_DOCUMENTACAO.json'
              and packaged(p)}
    require(actual == set(declared), 'Lista de arquivos difere do manifesto.')
    if errors:
        for e in errors:
            print('ERRO:', e)
        return 1
    print(f'OK: {len(markdowns)} documentos Markdown; {links} links locais; {len(declared)} arquivos com hash.')
    print('OK: dez finais auditados, partida comum, 4.104 chamadas e uma auditoria adicional por método.')
    print('Somente evidências salvas foram verificadas; nenhuma simulação ou avaliação AC foi executada.')
    return 0

if __name__ == '__main__':
    sys.exit(verify())
