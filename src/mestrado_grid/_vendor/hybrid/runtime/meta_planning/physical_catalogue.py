"""Traceable, explicitly synthetic 30 kV expansion scenarios.

The original catalogue is copied intact. Values below are study assumptions,
not certified Leiria equipment or commercial prices. Monetary units stay u.m.
"""
import copy
import csv
import math
import hashlib
from pathlib import Path
import numpy as np
from .schema import Study
from .physical import VERSION
from .contracts import number

SOURCES={
 'eland':'https://www.elandcables.com/media/cxebcdls/lxhio1av-cable-datasheet.pdf',
 'prysmian':'https://co.prysmian.com/sites/co.prysmian.com/files/media/documents/FT-ACSR.pdf',
 'mahmoud':'https://doi.org/10.3390/su142013299',
 'araujo':'https://doi.org/10.1007/s40313-026-01305-6',
 'eaton':'https://www.eaton.com/us/en-us/catalog/medium-voltage-power-distribution-control-systems/single-phase-overhead-voltage-regulator.html',
 'epri':'https://opendss.epri.com/PVinverterproperties.html',
 'sma':'https://manuals.sma.de/SHPxx20/en-US/5810005643.html'}


def source_profile(st,g,kind,ratio,cost_scale,night):
    tech=g.get('technology','BESS')
    # Primary resource and electrical connection are deliberately distinct. BIO
    # outages follow its original source; PV darkness does not imply disconnection.
    availability=[True]*st.S
    if tech=='BIO':
        availability=[(g['availability'][str(s)]>0 if kind=='new' else st.av[g['id'],s]['maximum']>0) for s in range(st.S)]
    return dict(id=g['id'],technology=tech,s_ratio=ratio,current_ratio=ratio,
                q_inj_ratio=.33 if tech!='BIO' else math.tan(math.acos(.85)),
                q_abs_ratio=.33 if tech!='BIO' else .25,power_factor=.85 if tech=='BIO' else 1.,
                electrical_available=availability,night_var_capable=bool(night and tech=='PV') or tech=='BESS',
                standby_MW_per_MW=.0005,q_loss_coefficient=.005,
                capex_per_MW=cost_scale*(15000+150000*(ratio-1)),maintenance_fraction=.02,life_years=15,
                aux_price_per_MWh=55.,owner='producer' if kind=='new' and tech in ('PV','WT') else 'operator',
                control='metaheuristic_Q; retrofit control rights assumed purchased, no legacy energy contract rewrite',
                provenance=dict(geometry='derived circle and nominal current; not a manufacturer D-curve',
                  technology='hypothesis: full converter PV/WT/BESS; synchronous BIO with pf wedge',
                  availability='hypothesis: connected except documented BIO primary outage; configurable separately',
                  costs='hypothesis 15000 u.m./MW enablement plus 150000*(S/P-1); O&M 2%, life 15',
                  losses='hypothesis standby 0.0005 MW/MW only during Q service plus 0.005 Q^2/S; incremental to original active efficiencies',
                  aux_tariff='hypothesis: producer reimburses incremental metered auxiliaries at 55 u.m./MWh; no Q fee',
                  envelope_source=SOURCES['epri'],night_source=SOURCES['sma']))


def add_conductors(st,d,el,scale,geometry_m=1.,resistance_temperature_C=75.):
    root=Path(__file__).resolve().parents[2]
    file=root/'expansion_planner/data/leiria180/interpreted_parameters/line_options_directed.csv'
    if st.B==180:
        with file.open(encoding='utf-8-sig',newline='') as f:
            source={(int(r['from_bus']),int(r['to_bus'])):r for r in csv.DictReader(f) if r['option_index']=='0'}
        el['physical']['provenance']['line_mapping']=dict(path=str(file.relative_to(root)),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    else:source={}
    for e in d['lines']:
        src=source.get((e['i'],e['j']))
        if st.B==180 and src is None:raise ValueError('Unmapped conductor family/length')
        length=float(src['distance_km_interpreted']) if src else 1.
        underground=not src or 'ACSR' not in src['reference_type']
        # Same-family replacements only. Temperature correction is explicit and
        # conservative 60 Hz overhead Rac is retained at 50 Hz, not relabeled.
        table=[(240,.125,.37,.24,435),(300,.100,.36,.26,493),(400,.0778,.34,.29,563),(500,.0605,.33,.31,642)] if underground else [(336.4*.506707479,.202,0,0,529),(477*.506707479,.143,0,0,659),(636*.506707479,.108,0,0,789)]
        for j,(area,res,ind,cap,current) in enumerate(table):
            option=len(e['options'])
            if underground:
                resistance=res*(1+.00403*(resistance_temperature_C-20))*1.03
                reactance=2*math.pi*50*ind*1e-3;b=2*math.pi*50*cap*1e-6
                name=f'LXHIO1AV_{int(area)}';origin=SOURCES['eland'];gmr=None
                price_km=scale*(90000+3*50000*(area/240)**.8)
            else:
                resistance=res;gmr=[.005,.007,.008][j]
                reactance=2*math.pi*50*2e-4*math.log(geometry_m/gmr);b=0.
                name=['Linnet','Hawk','Grosbeak'][j];origin=SOURCES['prysmian']
                price_km=scale*(25000+3*32000*(area/160)**.8)
            rating=math.sqrt(3)*30*current/1000
            cost=price_km*length
            # Candidate root cables historically contain a transformer cost.
            # Retain that independently documented lump sum in every alternative.
            transformer=float(src['transformer_purchase_separate']) if src else 0.
            cost+=transformer
            e['options'].append(dict(id=option,capacity_MW=rating,capacity_MVA=rating,nominal_current_A=current,capex=cost,
                maintenance_per_year=.02*(cost-transformer)+(float(src['transformer_maintenance_separate']) if src else 0.),
                failure_rate=copy.deepcopy(e['options'][0]['failure_rate']),repair_hours=copy.deepcopy(e['options'][0]['repair_hours']),
                physical_name=name,family='underground' if underground else 'overhead',section_mm2=area,length_km=length,
                voltage_kV=30.,frequency_Hz=50.,R_ohm=resistance*length,
                provenance=dict(source=origin,physical_table='Eland p6' if underground else 'Prysmian pp2-3',
                  units='R/X ohm/km; B siemens/km; length km; converted on 900 ohm base',
                  resistance='hypothesis Rac=Rdc20*(1+.00403*(T-20))*1.03' if underground else 'source Rac75 at 60Hz retained conservatively at 50Hz; installation validation pending',
                  temperature_C=resistance_temperature_C if underground else 75.,
                  geometry_m=geometry_m if not underground else None,GMR_m=gmr,
                  section='source mm2' if underground else 'derived aluminum area: source kcmil times 0.506707479, excludes steel',
                  reactance='source L converted at 50Hz' if underground else 'hypothesis GMR and equivalent spacing; X=2pi50*2e-4*ln(D/GMR)',
                  current='source buried rating, installation derating not confirmed' if underground else 'source ambient25 conductor75 solar1033.3 wind0.61; mechanical checks pending',
                  cost='synthetic supply power law plus fixed installation/km, full three-phase set; transformer separately retained',
                  maintenance='hypothesis 2% of new cable assembly; original transformer allowance retained',
                  reliability='hypothesis: original same-route directional failures and repairs retained; no presumed improvement',
                  shunt='source C at 50Hz' if underground else 'hypothesis negligible overhead capacitance')))
            el['branches'].append(dict(line=e['id'],option=option,r=resistance*length/900,x=reactance*length/900,b=b*length*900))


def extend(st,*,q_mode='original',sub_mode='SUB_EQUIPMENT',cost_scenario='central',cost_family='mahmoud_adapted',
           inverter_ratio=1.,night_var=False,enabled=None,capacitor_life=10,regulator_life=15,geometry_m=1.,temperature_C=75.):
    """Return a new Study; no caller data, old prices or active demand are mutated."""
    if type(night_var)!=bool:raise ValueError('night_var must be an explicit boolean')
    if not isinstance(q_mode,str):raise ValueError('q_mode must be an explicit string')
    if enabled is not None and not isinstance(enabled,dict):raise ValueError('enabled must be a mapping of boolean feature flags')
    for name,value in [('inverter_ratio',inverter_ratio),('geometry_m',geometry_m),('temperature_C',temperature_C)]:number(value,name)
    if 'physical' in st.electrical:raise ValueError('Already extended')
    if cost_scenario not in ('low','central','high'):raise ValueError('Unknown cost scenario')
    if cost_family not in ('mahmoud_adapted','araujo_adapted'):raise ValueError('Unknown cost family')
    if inverter_ratio not in (1.,1/.95,1.10):raise ValueError('Select a documented fixed inverter offer')
    if capacitor_life not in (10,20) or regulator_life not in (15,20):raise ValueError('Select a documented lifetime scenario')
    if not .1<=geometry_m<=5 or not 20<=temperature_C<=90:raise ValueError('Invalid conductor thermal/geometry hypothesis')
    d=copy.deepcopy(st.data);el=copy.deepcopy(st.electrical);scale={'low':.5,'central':1.,'high':2.}[cost_scenario]
    flags=dict(new_q=True,bess_q=True,retrofit=True,capacitors=True,regulators=True,conductors=True)
    if enabled:
        if not set(enabled)<=set(flags):raise ValueError('Unknown feature flag')
        flags.update(enabled)
    ratio=np.asarray(el['q_ratio'])
    if q_mode=='anchored':
        # First normal MAT profile per bus: a declared synthetic constant PF.
        # No fitting of the 89 tiny filler loads to consumers elsewhere.
        ratio=np.broadcast_to(ratio[:,0,None],ratio.shape).copy()
    elif q_mode.startswith('fp:'):
        fp=float(q_mode.split(':')[1])
        if fp not in (.90,.95,.98):raise ValueError('Unsupported load PF sensitivity')
        ratio=np.where(st.base_load[0]>0,math.tan(math.acos(fp)),0.)
    elif q_mode!='original':raise ValueError('Unknown load Q scenario')
    el['q_ratio']=ratio.tolist()
    p=dict(version=VERSION,enabled=flags,q_mode=q_mode,sub_mode=sub_mode,cost_scenario=cost_scenario,cost_family=cost_family,
           currency='constant model units; no FX conversion or commercial quote',replacement_policy='identical_at_lifetime',
           capacitor_sites=[b for b in st.buses if b!=d['root']],regulator_sites=[e['id'] for e in st.lines],
           capacitors=[],regulators=[],new_sources=[],old_sources=[],bess_sources=[],
           provenance=dict(sources=SOURCES,cost_sensitivity='low=0.5 central; high=2 central on new devices only',
                load_Q='original or synthetic fixed per-bus PF anchored to first profile, or illustrative .90/.95/.98',
                SUB='two explicitly alternative assumptions, no factual contingency inferred'),
           hypotheses=['30 kV balanced equivalents; apparatus insulation/installation not manufacturer-certified',
                'Inverter ratio is a fixed producer offer per scenario, never a free sizing gene outside operator costs',
                'No Q tariff; incremental auxiliary reimbursement is an explicit producer expense and operator credit',
                'Capacitors available in every modeled normal state; no capacitor outage events added',
                'Regulator failures use ideal automatic bypass; no customer interruption increment attributed',
                'Switching counts are representative-day cycles, not an annual event chronology',
                'Original R temperatures unknown: new conductor option is not asserted to dominate old equipment',
                'BIO site ceiling and original active connection/network checks preserved',
                'Replacement at exact lifetime renews only the identical device, preserving its controls and capacity'])
    el['physical']=p
    for amount,cheap in zip((.3,.6,.9,1.2),(1300,1600,1900,2800)):
        base=(1000+30000*amount) if cost_family=='mahmoud_adapted' else cheap
        for mode in ('fixed','switched','staged'):
            steps=round(amount/.3) if mode=='staged' else 1
            cost=scale*(base if mode=='fixed' else 1.3*base if mode=='switched' else 1.3*base+500*(steps-1))
            p['capacitors'].append(dict(id=f'CB_{mode}_{round(amount*1000)}',mode=mode,steps=steps,q_MVAr=amount,voltage_kV=30.,
                capex=cost,maintenance_per_year=.02*cost,life_years=capacitor_life,switch_cost=.1*scale,
                loss_coefficient=.0005,failure_rate=0.,repair_hours=0.,max_daily_switches=8*steps,
                electrical_available=[True]*st.S,
                provenance=dict(cost='Mahmoud Eq37 adapted per site + Araujo switching factor' if cost_family=='mahmoud_adapted' else 'Araujo dataset figures transposed to u.m./30kV study set',
                  loss='hypothesis 0.5 W/kvar as conductance; fixed/switched/staged distinct assemblies',
                  maintenance='hypothesis 2%/year; not source Eq37 charged twice',
                  switching='hypothesis .1 u.m./operation; staged additional 500 u.m./switch',
                  reliability='explicit no outage in these normal states; zero not a measured failure rate')))
    currents=(100,150,200,250,300,350,400) if cost_family=='mahmoud_adapted' else (300,400,500,600)
    prices=(38000,44800,50600,58100,64700,70300,77900) if cost_family=='mahmoud_adapted' else (11350,14500,17000,20000)
    for current,price in zip(currents,prices):
        cost=scale*price
        p['regulators'].append(dict(id=f'VR_{current}',voltage_kV=30.,current_from_A=current,current_to_A=current,capex=cost,
            maintenance_per_year=.02*cost,life_years=regulator_life,switch_cost=.1*scale,loss_coefficient=.0005,
            series_r_pu=.00001,series_x_pu=.00001,failure_rate=.05,repair_hours=4.,bypass_on_failure=True,max_daily_switches=128,
            electrical_available=[True]*st.S,
            provenance=dict(taps='Eaton +/-10%, 33 positions; equivalent 3-phase set, physical i->j orientation',
              cost='Mahmoud Eq38' if cost_family=='mahmoud_adapted' else 'Araujo dataset; adapted whole set not claimed original phase convention',
              losses='hypothesis 0.00001+j0.00001 pu series plus 0.5kW core at 1pu; independent of tap',
              reliability='hypothesis .05/year 4h; automatic bypass. Availability states normal, no outage event simulation',
              maintenance='hypothesis 2%/year')))
    p['new_sources']=[source_profile(st,g,'new',inverter_ratio,scale,night_var) for g in st.new]
    p['old_sources']=[source_profile(st,g,'old',inverter_ratio,scale,night_var) for g in st.old]
    p['bess_sources']=[source_profile(st,g,'bess',inverter_ratio,scale,True) for g in st.units]
    if flags['conductors']:add_conductors(st,d,el,scale,geometry_m,temperature_C)
    d['config']['power_flow']='balanced_AC_with_explicit_Q_shunts_and_taps'
    d['config']['assumptions']=[*p['hypotheses'],'Original active demand/events, prices, PPA45/45 and BESS base remain unchanged']
    el['assumptions']=copy.deepcopy(d['config']['assumptions'])
    return Study(d,el)
