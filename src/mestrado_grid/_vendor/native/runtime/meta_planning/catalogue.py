"""Read the audited Leiria electrical assumptions without importing its solvers."""
from pathlib import Path
import csv
import math
import hashlib
from .schema import Study, read

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT/'results/leiria180/contratos_PV_WT_20260920/execucao_02/referencias/b000/input.json'
DEFAULT_ELECTRICAL = ROOT/'results/leiria180/validacao_AC_T4_20260915'


def load_leiria(input_path=DEFAULT_INPUT, electrical_path=DEFAULT_ELECTRICAL):
    d = read(input_path); directory = Path(electrical_path)
    # Retain provenance without presenting inherited DC/linear-loss labels as
    # the physics executed by this direct AC evaluator. Source files are intact.
    c=d['config']
    c.setdefault('source_model_metadata',{k:c[k] for k in ('power_flow','assumptions','article_experiment_mip_gap') if k in c})
    c['power_flow']='balanced_AC_fixed_PQ_with_one_slack'
    c['assumptions']=[
        'Balanced nonlinear AC with Newton and radial-sweep cross-check; no economic redispatch.',
        'Voltages, terminal currents, losses and slack P/Q are calculated from the supplied controls.',
        'New DG and BESS have Q=0; conventional Q scales with served P; EV Q=0.',
        'Known representative T4 days with cyclic SOC and daily DR/EV energy conservation.',
        'Directional reliability paths and FOR costs remain proxies, not interruption simulation.',
        'Original source assumptions are retained separately in source_model_metadata.'
    ]
    c.pop('article_experiment_mip_gap',None)
    if d['buses'] != list(range(1,181)) or d['root'] != 1 or len(d['scenarios']) != 32:
        raise ValueError('This external electrical mapping is only valid for Leiria180/T4')
    def rows(name):
        with (directory/name).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))
    base = read(directory/'bases.json')
    load_rows=rows('Cargas_Q_P_originais.csv')
    ratio = {(int(r['Entrada_MAT']),int(r['Barra'])):float(r['Razao_Q_P']) for r in load_rows}
    expected_load_keys={(s,b) for s in range(1,17) for b in d['buses'] if b!=d['root']}
    if len(ratio)!=len(load_rows) or set(ratio)!=expected_load_keys:
        raise ValueError('Missing or duplicate load Q/P source rows')
    generator_rows=rows('Geradores_Q_originais.csv')
    generators_q={int(r['ID']):r for r in generator_rows}
    if len(generators_q)!=len(generator_rows):raise ValueError('Missing or duplicate generator Q source rows')
    for g in d['generators']:
        if g['id'] not in generators_q or int(generators_q[g['id']]['Barra'])!=g['bus']:
            raise ValueError('Generator Q source ID/bus mismatch')
    voltage_rows = rows('Tensoes_limites.csv')
    vs = {int(r['Barra']):r for r in voltage_rows}
    if len(vs)!=len(voltage_rows) or set(vs)!=set(d['buses']):
        raise ValueError('Missing or duplicate voltage bus limits')
    el = dict(Sbase_MVA=base['Sbase_MVA'],Ibase_A=base['Ibase_A'],vslack=base['Vslack_pu'],
              vmin=[float(vs[b]['Vmin_pu']) for b in d['buses']],vmax=[float(vs[b]['Vmax_pu']) for b in d['buses']],
              q_ratio=[[0. if b==1 else ratio[s//2+1,b] for s in range(32)] for b in d['buses']],
              generator_q={str(i):[float(r['Qmin_MVAr']),float(r['Qmax_MVAr'])] for i,r in generators_q.items()},
              branches=[dict(line=r['Ligacao'],option=int(r['Opcao']),r=float(r['R_opcao_pu']),x=float(r['X_branch_pu']),b=float(r['B_branch_pu'])) for r in rows('Opcoes_eletricas.csv')],
              assumptions=['Balanced AC; existing raw Q intervals, not manufacturer P-Q curves',
                           'New generation/BESS Q=0; EV Q=0; conventional Q scales with served P',
                           'Enlarged-option X is an unconfirmed catalogue assumption',
                           'SUB derating follows original state availability; no export',
                           'The 32 T4 states are known profiles, not surprise-event recourse'])
    source_files = [Path(input_path)] + [directory/name for name in ('bases.json','Cargas_Q_P_originais.csv','Tensoes_limites.csv','Geradores_Q_originais.csv','Opcoes_eletricas.csv')]
    el['source_hashes'] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    return Study(d,el)


def analytic_study():
    """Three buses, alternative branches, two BESS types, PV/WT/BIO and two days.

    A deliberately small integration fixture, not a calibrated Leiria substitute.
    Each cycle interleaves state IDs to detect accidental contiguous-day grouping.
    """
    S=8; years=[0,10,20]
    config=dict(investment_years=years,horizon_years=30,discount_rate=.0005,alpha=.9,beta=.5,annual_hours=8760,
                load_growth_by_stage=[1,1,1],generation_growth_by_stage=[1,1,1],SAIFI_limits=[2,2,2],SAIDI_limits=[10,10,10],
                storage_scenario_cycles=[[0,2,4,6],[1,3,5,7]],scenario_block_hours=[12,12,5,5,3,3,4,4],
                prices_per_MWh={'ESS':40},emissions_ton_per_MWh={'ESS':.2012},carbon_price_per_ton=51,
                unserved_energy_per_MWh=3000,loss_cost_per_MWh=120,EENS_cost_per_MWh=3000,
                substation_rated_MVA=10,substation_active_limit_MW=10)
    lines=[]; branches=[]
    for eid,i,j,existing in [('a',1,2,True),('b',2,3,True),('c',1,3,False)]:
        opts=[]
        for o in (0,1):
            opts.append(dict(id=o,capacity_MW=8+o,nominal_current_A=1000,capex=1000*(o+1),maintenance_per_year=10+o,failure_rate=[.1,.2],repair_hours=[1,2]))
            branches.append(dict(line=eid,option=o,r=.001/(o+1),x=.001,b=0.))
        lines.append(dict(id=eid,i=i,j=j,existing=existing,initial_closed=existing,options=opts))
    storage=[dict(id='ess'+str(j),bus=2,energy_MWh=j,power_MW=.2*j,capex=200000*j,maintenance_per_year=7000*j,eta_charge=.95,eta_discharge=.95,replacement_accounting='included_in_maintenance') for j in (1,2)]
    old=[dict(id=1,bus=1,technology='SUB',renewable=False,price=55,curtail_price=0),
         dict(id=2,bus=3,technology='BIO',renewable=False,price=45,curtail_price=0)]
    new=[dict(id=j,bus=3,technology=tech,max_MW=.15,capex_per_MW=1e6,fixed_per_MW_year=10000,variable_per_MWh=45 if tech=='BIO' else 0,curtail_per_MWh=0,emissions_ton_per_MWh=.02,
              availability={str(s):(.8 if tech!='PV' or s<4 else 0.) for s in range(S)}) for j,tech in enumerate(('PV','WT','BIO'),10)]
    operations=[dict(bus=b,scenario=s,time=0,load_MW=0. if b==1 else .3+.05*(s//2),ev_MW=.04 if b==3 else 0.) for b in (1,2,3) for s in range(S)]
    availability=[dict(generator=g['id'],scenario=s,time=0,minimum=0.,maximum=10 if g['id']==1 else .3,emissions=.6 if g['id']==1 else .7) for g in old for s in range(S)]
    d=dict(config=config,buses=[1,2,3],root=1,consumers={1:0,2:10,3:20},lines=lines,storage=storage,generators=old,availability=availability,operations=operations,
           scenarios=[dict(id=s,probability=config['scenario_block_hours'][s]/48) for s in range(S)],time_steps=[dict(id=0,hours=1)],investment_mode='multi',
           extension=dict(generators=new,flex=dict(load_shift_fraction=.1,shift_compensation_per_MWh=10,ev_enabled=True,ev_peak_multiplier=2,ev_compensation_per_MWh=5)))
    from .contracts import prepare
    d=prepare(d)
    el=dict(Sbase_MVA=1.,Ibase_A=1000/(math.sqrt(3)*30),vslack=1.04,vmin=[.95]*3,vmax=[1.05]*3,
            q_ratio=[[0.]*S,[.2]*S,[.2]*S],generator_q={'1':[-1.,3.],'2':[-.1,.2]},branches=branches,assumptions=['Analytical integration fixture'])
    return Study(d,el)
