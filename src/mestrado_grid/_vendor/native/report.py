"""Paired descriptive analysis. Never calls an evaluator or starts a search."""
from bootstrap import *
import html,statistics,datetime
import numpy as np
from experiment import METHOD_LIST,SEEDS,TOTAL_CALLS
from meta_planning.evaluator import rank
from progress_analysis import milestones
def esc(x):return html.escape(str(x))
def fmt(x):return '—' if x is None else f'{x:,.2f}'.replace(',','@').replace('.',',').replace('@','.')
def score(r):
 f=r['final']
 return rank(dict(**f,metrics=dict(selection_objective=f.get('FO')),search_selection='feasibility_violation_exact_cost_tie_v1'))
def make_report():
 rows=[read(p) for p in sorted((D/'runs').glob('*/*/summary.json'))]
 status=read(D/'benchmark_status.json') if (D/'benchmark_status.json').exists() else dict(state='not_started',completed=0,total=480,failed=0)
 lookup={(r['regime'],r['simulation'],r['beta'],r['seed'],r['method']):r for r in rows}
 cells=[];pairs=[];checks=[]
 for regime in ('normal','extremo'):
  for sim in (1,2):
   for beta in (0.,.5,1.):
    for method in METHOD_LIST:
     group=[lookup[k] for seed in SEEDS if (k:=(regime,sim,beta,seed,method)) in lookup]
     if not group:continue
     vals=[r['final']['FO'] for r in group if r['final']['status']=='feasible']
     viol=[r['final']['normalized_violation'] for r in group]
     deltas=[r['start']['FO']-r['final']['FO'] for r in group if r['start']['status']==r['final']['status']=='feasible']
     cells.append(dict(regime=regime,simulation=sim,beta=beta,method=method,n=len(group),feasible=len(vals),
      median_FO=float(np.median(vals)) if vals else None,q25_FO=float(np.quantile(vals,.25)) if vals else None,q75_FO=float(np.quantile(vals,.75)) if vals else None,
      median_violation=float(np.median(viol)),median_feasible_gain=float(np.median(deltas)) if deltas else None,
      full_family_opportunities=sum(r['resource_coverage']['initial_opportunity_sweep_complete'] for r in group),
      native_all_family_changes=sum(r['resource_coverage']['native_physical_family_coverage_complete'] for r in group)))
    for method in METHOD_LIST[:-1]:
     wins=ties=losses=0;data=[]
     for seed in SEEDS:
      a=lookup.get((regime,sim,beta,seed,'L-SHADE'));b=lookup.get((regime,sim,beta,seed,method))
      if a is None or b is None:continue
      assert a['calls']==b['calls']==TOTAL_CALLS
      for field in ('initial_population_sha256','source_population_sha256','study_sha256'):assert a[field]==b[field]
      assert a['allocation_policy']['order']==b['allocation_policy']['order']
      result=-1 if score(a)<score(b) else 1 if score(a)>score(b) else 0
      wins+=result==-1;ties+=result==0;losses+=result==1
      data.append(dict(seed=seed,lshade=a['final'],baseline=b['final']))
     if data:pairs.append(dict(regime=regime,simulation=sim,beta=beta,baseline=method,n=len(data),lshade_wins=wins,ties=ties,lshade_losses=losses,pairs=data))
 progress=[]
 for r in rows:
  assert r['fresh_audit_matches'] and r['finance']['all_pass'] and r['calls']==TOTAL_CALLS
  checks.append(dict(context=r['context'],fresh96=True,financial=True,full_families=r['resource_coverage']['initial_opportunity_sweep_complete']))
  progress.append(dict(context=r['context'],milestones=milestones(D/'runs'/r['context'],r)))
 save(D/'budget_progress.json',dict(checkpoints=[8,72,264,520,1032,2056,4104],trials=progress))
 analysis=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,completed_trials=len(rows),expected_trials=480,
  complete=len(rows)==480 and not status.get('failed',0),cells=cells,paired_comparisons=pairs,audits=checks,
  inference='Descriptive paired five-seed comparison. Beta classes, scenarios and starting populations are separate strata, not independent seed replicates. No global optimum or isolated causal attribution.')
 save(D/'analysis.json',analysis)
 body=[]
 for c in cells:
  body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [c['regime'],'MAT' if c['simulation']==1 else 'Heurística',c['beta'],c['method'],f"{c['n']}/5",f"{c['feasible']}/{c['n']}",fmt(c['median_FO']),fmt(c['median_feasible_gain']),str(c['median_violation']),f"{c['full_family_opportunities']}/{c['n']}",f"{c['native_all_family_changes']}/{c['n']}"])+'</tr>')
 pbody=['<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [p['regime'],p['simulation'],p['beta'],p['baseline'],p['n'],p['lshade_wins'],p['ties'],p['lshade_losses']])+'</tr>' for p in pairs]
 content='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>L-SHADE · comparação ampliada</title>
<style>body{font:16px system-ui;color:#18313b;max-width:1500px;margin:36px auto;padding:0 24px;background:#f4f7f8}h1{font-size:32px}section{background:white;padding:22px;margin:20px 0;border-radius:12px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e3eff0}small{color:#53646c}.badge{font-size:24px;color:#00766f}p{max-width:1100px;line-height:1.55}</style>
<h1>L-SHADE e sete motores atuais</h1>'''
 content+=f'<p class="badge">{len(rows)}/480 execuções concluídas e auditadas · {esc(status.get("state"))}</p>'
 content+='<section><h2>Protocolo</h2><p>Cinco sementes: 96101, 96109, 96121, 96137 e 96149. PSO, HyDE, GA, ACO, Vortex, QPSO, SHADE e L-SHADE. Partidas MAT e heurística, perfis normais e extremos; β = 0, 0,5 e 1; α = 0,9.</p><p><b>4.104 chamadas por execução:</b> oito de inicialização e 4.096 de busca. Cada projeção, reinicialização de diversidade e reavaliação de contexto consome orçamento, inclusive repetidos. Auditoria final de 96 casos adicional. A meta de 30 milhões é medida, mas não encerra antecipadamente este ensaio de orçamento igual.</p><p>FO Castro-aligned, CVaR, física e restrições oficiais preservados por hashes. Mesmo conjunto inicial de oito membros por comparação; cópias na partida heurística declaradas. Agenda integral de 15 famílias, com ordem inicial igual entre motores para a mesma semente/classe. Janelas P/Q, cobertura de recursos e reconfiguração mantidas. Oportunidade não garante instalação, mudança física nativa ou visita exaustiva de todos os locais e casos.</p><p>L-SHADE: população local inicial de oito, reduzida até quatro pelo progresso de chamadas do ensaio; memórias F/CR e arquivo de posições preservados, fitness reavaliado na mudança de contexto. Adaptação explícita ao executor por blocos, com mais gerações/visitas possíveis conforme a população diminui. Referência: <a href="https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf">Tanabe e Fukunaga (2014)</a>.</p></section>'
 content+='<section><h2>Resultados por contexto</h2><p>Custos são calculados apenas entre finais viáveis, sempre acompanhados da contagem de viabilidade. Redução mediana exige partida e final viáveis. Violações aparecem sem arredondamento para zero. “Famílias oferecidas” é distinta de mudança física nativa nas 15 famílias.</p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Perfis','Partida','β','Motor','Sementes','Viáveis','FO mediana (u.m.)','Redução mediana','Violação mediana','Famílias oferecidas','Mudança nativa completa'])+'</tr>'+''.join(body)+'</table></section>'
 content+='<section><h2>Comparações pareadas: L-SHADE versus cada motor</h2><p>Vitória usa viabilidade primeiro, menor violação normalizada e FO oficial no desempate exato. Cinco sementes permitem uma análise descritiva; não provam superioridade geral. Resultados parciais não representam todos os pares previstos.</p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['Perfis','Simulação','β','Comparador','Pares','Vitórias L-SHADE','Empates','Derrotas L-SHADE'])+'</tr>'+''.join(pbody)+'</table></section>'
 content+='<section><h2>Limites e rastreabilidade</h2><p>As partidas são fixas e iguais entre motores, não amostras de novas redes. Classes β e perfis não substituem sementes independentes. Concorrência de CPU afeta tempo; orçamento pareado usa chamadas reais. Recursos, inicialização e agenda direcionam a exploração e são compartilhados; origem imediata de uma melhora não isola causalidade. Nenhuma modificação foi aplicada às campanhas de produção. As verificações iniciais e smokes são separados do ranking.</p><p><a href="protocol.json">Protocolo</a> · <a href="analysis.json">Análise</a> · <a href="release_checks.json">Validações</a> · <a href="execution_hashes.json">Hashes da execução</a></p></section></html>'
 (D/'RESULTADOS.html').write_text(content,'utf-8')
 files=[D/'protocol.json',D/'execution_hashes.json',D/'analysis.json',D/'budget_progress.json',D/'RESULTADOS.html']+[p for p in (D/'runs').glob('*/*/*') if p.name in ('summary.json','validation.json.gz','best_genome.npy','best_candidate.json','history.jsonl','persistent_diagnostics.json','sessions.json','start_provenance.json')]
 save(D/'MANIFESTO.json',dict(complete=analysis['complete'],files={str(p.relative_to(D)):sha(p) for p in files if p.exists()}))
 return dict(completed=len(rows),expected=480)
if __name__=='__main__':print(make_report())
