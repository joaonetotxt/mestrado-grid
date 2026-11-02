"""Optional physical devices. Bounds/decoding only, never an economic dispatch.

Positive Q injects. Capacitors are admittances; real taps precede the physical
i->j line, independent of tree orientation or power direction. Lifetimes use
explicit identical replacement events, never annualized CAPEX plus purchase.
"""
import copy
import math
import numpy as np
from .contracts import number

VERSION='physical-ac-v1'


def shapes(st):
    p=st.electrical['physical'];C=len(p['capacitor_sites']);L=len(p['regulator_sites'])
    return dict(cap_choice=(C,),reg_choice=(L,),retro_choice=(st.G,),bess_choice=(st.U,),
                new_q=(st.K,st.S,st.N),bess_q=(st.K,st.S,st.U),
                cap_state=(st.K,st.S,C),tap=(st.K,st.S,L))


def empty(st):
    return dict(version=VERSION,**{n:np.zeros(s).tolist() for n,s in shapes(st).items()})


def choice_options(st,name):
    p=st.electrical['physical'];stages=range(1 if st.data['investment_mode']=='single' else st.K)
    opts=p['capacitors'] if name=='cap_choice' else p['regulators'] if name=='reg_choice' else [None]
    enabled=p['enabled'][{'cap_choice':'capacitors','reg_choice':'regulators','retro_choice':'retrofit','bess_choice':'bess_q'}[name]]
    return [None]+[(o,k) for o in range(len(opts)) for k in stages] if enabled else [None]


def installed(st,c,name,k):
    choices=choice_options(st,name)
    return [None if choices[int(i)] is None or choices[int(i)][1]>k else choices[int(i)][0]
            for i in c.physical[name]]


def validate_catalogue(st):
    p=st.electrical['physical']
    number(st.electrical['Sbase_MVA'],'Sbase_MVA',1e-12)
    if not math.isclose(st.electrical['Ibase_A'],1000/(math.sqrt(3)*30),rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('Physical extension requires a consistent 1 MVA / 30 kV base')
    if p['version']!=VERSION or p['sub_mode'] not in ('SUB_EQUIPMENT','SUB_P_ONLY'):
        raise ValueError('Unsupported physical catalogue')
    if p['replacement_policy']!='identical_at_lifetime':raise ValueError('Explicit replacement policy required')
    if set(p['enabled'])!={'new_q','bess_q','retrofit','capacitors','regulators','conductors'} or any(type(x)!=bool for x in p['enabled'].values()):
        raise ValueError('Explicit physical feature flags required')
    for name,known in [('capacitor_sites',st.buses),('regulator_sites',[e['id'] for e in st.lines])]:
        sites=p[name]
        identities={(type(x),x) for x in known}
        if len(set(sites))!=len(sites) or any((type(x),x) not in identities for x in sites):raise ValueError('Invalid '+name.replace('_',' '))
    for kind,rows,gens in [('new',p['new_sources'],st.new),('old',p['old_sources'],st.old),('bess',p['bess_sources'],st.units)]:
        if len(rows)!=len(gens):raise ValueError('Incomplete physical source mapping')
        for r,g in zip(rows,gens):
            if type(r['id'])!=type(g['id']) or r['id']!=g['id'] or r['technology']!=g.get('technology','BESS'):
                raise ValueError('Physical source identity/order must match the underlying catalogue')
            owner='producer' if kind=='new' and g['technology'] in ('PV','WT') else 'operator'
            if r['owner']!=owner:raise ValueError('Physical source owner contradicts the economic ledger')
            for n in ('s_ratio','q_inj_ratio','q_abs_ratio','current_ratio','standby_MW_per_MW','q_loss_coefficient','capex_per_MW','maintenance_fraction','aux_price_per_MWh'):
                number(r[n],n)
            if r['s_ratio']<1 or r['current_ratio']<1:raise ValueError('Converter must support original nominal P')
            if type(r['night_var_capable'])!=bool or type(r['life_years'])!=int or r['life_years']<=0:raise ValueError('Invalid device life/night capability')
            a=np.asarray(r['electrical_available'])
            if a.shape!=(st.S,) or any(type(x)!=bool for x in r['electrical_available']):raise ValueError('Electrical availability must be an explicit boolean per state')
            number(r['power_factor'],'power_factor',1e-12)
            if r['power_factor']>1:raise ValueError('Invalid source power factor')
    for kind in ('capacitors','regulators'):
        if len({r['id'] for r in p[kind]})!=len(p[kind]):raise ValueError('Duplicate physical option ID')
        for r in p[kind]:
            if len(r['electrical_available'])!=st.S or any(type(x)!=bool for x in r['electrical_available']):raise ValueError('Device availability must be explicit booleans')
            for n in ('capex','maintenance_per_year','switch_cost','loss_coefficient','failure_rate','repair_hours'):
                number(r[n],n)
            if r['repair_hours']>=8760 or type(r['life_years'])!=int or r['life_years']<=0:raise ValueError('Invalid device life/reliability')
            if r['voltage_kV']!=30:raise ValueError('Physical extension requires a declared 30 kV set')
            if kind=='capacitors':
                number(r['q_MVAr'],'capacitor Q',1e-12)
                if r['mode'] not in ('fixed','switched','staged'):raise ValueError('Invalid capacitor mode')
                if type(r['steps'])!=int or r['steps']<=0 or (r['mode']!='staged' and r['steps']!=1):raise ValueError('Invalid capacitor steps')
            else:
                for n in ('current_from_A','current_to_A'):number(r[n],n,1e-12)
                for n in ('series_r_pu','series_x_pu'):number(r[n],n)
                if r['bypass_on_failure'] is not True:raise ValueError('Only explicitly bypassed regulator failures supported')
            if type(r['max_daily_switches'])!=int or r['max_daily_switches']<0:raise ValueError('Invalid switching limit')
    if not p.get('provenance') or not p.get('hypotheses'):raise ValueError('Physical assumptions/provenance missing')


def validate_candidate(st,c):
    if 'physical' not in st.electrical:
        if c.physical is not None:raise ValueError('Physical candidate without catalogue')
        return
    if c.physical is None or set(c.physical)!={'version',*shapes(st)} or c.physical['version']!=VERSION:
        raise ValueError('Missing or incompatible physical controls')
    for name,shape in shapes(st).items():
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in np.asarray(c.physical[name],dtype=object).flat):
            raise ValueError('Physical controls must be numeric decisions, without booleans or strings')
        a=np.asarray(c.physical[name],float)
        if a.shape!=shape or not np.all(np.isfinite(a)):raise ValueError('Invalid physical array: '+name)
        if name.endswith('_choice'):
            if np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>=len(choice_options(st,name))):raise ValueError('Invalid installation choice')
        if name in ('cap_state','tap') and np.any(a!=np.floor(a)):raise ValueError('Discrete physical control required')


def sub_limits(st,k,s):
    av=st.av[st.sub['id'],s];growth=st.c['generation_growth_by_stage'][k]
    hi=av['maximum']*growth;lo=max(0.,av['minimum']*growth)
    mode=st.electrical.get('physical',{}).get('sub_mode','SUB_EQUIPMENT')
    # An active-import cap alone does not disconnect the interface or remove Q.
    # Zero equipment availability, in contrast, preserves the historical outage.
    qlo,qhi=st.electrical['generator_q'][str(st.sub['id'])] if mode=='SUB_P_ONLY' or hi>1e-9 else (0.,0.)
    rated=st.c['substation_rated_MVA']
    return lo,hi,qlo,qhi,rated if mode=='SUB_P_ONLY' else rated*hi/st.c['substation_active_limit_MW']


def envelope(r,nominal,active,state,primary_available=True):
    if nominal<=0 or not r['electrical_available'][state] or (not primary_available and not r['night_var_capable']):return 0.,0.
    if r['technology']=='BIO' and active<=1e-9:return 0.,0.
    circle=math.sqrt(max(0.,(r['s_ratio']*nominal)**2-active**2))
    lower=min(circle,r['q_abs_ratio']*nominal);upper=min(circle,r['q_inj_ratio']*nominal)
    if r['technology']=='BIO':
        wedge=abs(active)*math.tan(math.acos(r['power_factor']))
        lower=min(lower,wedge);upper=min(upper,wedge)
    return -lower,upper


def existing_connected(st,c,k):
    """Electrical connection of existing sources; retrofit applies from entry."""
    connected=np.ones((st.S,st.G),bool)
    if 'physical' in st.electrical:
        for j,o in enumerate(installed(st,c,'retro_choice',k)):
            if o is not None:connected[:,j]=st.electrical['physical']['old_sources'][j]['electrical_available']
    return connected


def existing_bounds(st,c):
    lo,hi,ql,qh=st.existing_bounds()
    for k in range(st.K):
        disconnected=~existing_connected(st,c,k)
        for a in (lo,hi,ql,qh):a[k][disconnected]=0.
    return lo,hi,ql,qh


def q_bounds(st,c):
    from .temporal import storage_dispatch
    p=st.electrical['physical'];active=st.active_capacity(c);ch,di=storage_dispatch(st,c)
    _,_,ol,oh=existing_bounds(st,c);nl=np.zeros((st.K,st.S,st.N));nh=nl.copy();bl=np.zeros((st.K,st.S,st.U));bh=bl.copy()
    storage=st.storage_active(c.plan)
    for k in range(st.K):
        ro=installed(st,c,'retro_choice',k);bo=installed(st,c,'bess_choice',k)
        for s in range(st.S):
            for j,(g,r) in enumerate(zip(st.old,p['old_sources'])):
                if ro[j] is not None:
                    nom=max(st.av[g['id'],ss]['maximum'] for ss in range(st.S))*st.c['generation_growth_by_stage'][k]
                    ol[k,s,j],oh[k,s,j]=envelope(r,nom,c.operation.existing_p[k,s,j],s,st.av[g['id'],s]['maximum']>0)
            if p['enabled']['new_q']:
                for j,(g,r) in enumerate(zip(st.new,p['new_sources'])):
                    nl[k,s,j],nh[k,s,j]=envelope(r,active[k,j],c.operation.new_p[k,s,j],s,g['availability'][str(s)]>0)
            for j,(u,r) in enumerate(zip(st.units,p['bess_sources'])):
                if storage[k,j] and bo[j] is not None:bl[k,s,j],bh[k,s,j]=envelope(r,u['power_MW'],di[k,j,s]-ch[k,j,s],s)
    return ol,oh,nl,nh,bl,bh


def availability(st):
    """Eligible active resource availability, excluding electrical outages."""
    a=np.array([[g['availability'][str(s)] for g in st.new] for s in range(st.S)]).reshape(st.S,st.N)
    if st.electrical.get('physical',{}).get('enabled',{}).get('new_q',False):
        a*=np.array([r['electrical_available'] for r in st.electrical['physical']['new_sources']]).T
    return a


def decode(st,c,blocks):
    p=st.electrical['physical'];value=empty(st)
    for name in ('cap_choice','reg_choice','retro_choice','bess_choice'):
        size=len(choice_options(st,name));value[name]=np.minimum((blocks['phys_'+name]*size).astype(int),size-1).tolist()
    # A BESS retrofit can be bought only at/after the underlying storage purchase.
    buys=dict(c.plan.storage_buys);opts=choice_options(st,'bess_choice')
    for j,u in enumerate(st.units):
        v=opts[value['bess_choice'][j]]
        if v is not None:
            value['bess_choice'][j]=0 if u['id'] not in buys else opts.index((0,max(v[1],buys[u['id']])))
    c.physical=value
    lo,hi,_,_=existing_bounds(st,c)
    c.operation.existing_p=lo+blocks['existing_p']*(hi-lo)
    # Equipment outage is distinct from resource availability. No energy appears
    # when the electrical interface is unavailable.
    if p['enabled']['new_q']:
        for j,r in enumerate(p['new_sources']):c.operation.new_p[:,:,j]*=np.asarray(r['electrical_available'])[None,:]
    ol,oh,nl,nh,bl,bh=q_bounds(st,c)
    c.operation.existing_q=ol+blocks['existing_q']*(oh-ol)
    for name,low,high in [('new_q',nl,nh),('bess_q',bl,bh)]:value[name]=(low+blocks['phys_'+name]*(high-low)).tolist()
    cs=np.zeros(shapes(st)['cap_state']);tap=np.zeros(shapes(st)['tap'])
    for k in range(st.K):
        for j,o in enumerate(installed(st,c,'cap_choice',k)):
            if o is None:continue
            r=p['capacitors'][o]
            cs[k,:,j]=1 if r['mode']=='fixed' else np.minimum((blocks['phys_cap_state'][k,:,j]*(r['steps']+1)).astype(int),r['steps'])
            cs[k,:,j]*=r['electrical_available']
        for j,o in enumerate(installed(st,c,'reg_choice',k)):
            if o is not None and p['regulator_sites'][j] in c.plan.trees[k]:tap[k,:,j]=(np.minimum((blocks['phys_tap'][k,:,j]*33).astype(int),32)-16)*np.asarray(p['regulators'][o]['electrical_available'])
    value['cap_state']=cs.tolist();value['tap']=tap.tolist()


def encode(st,c,fraction):
    out={};p=st.electrical['physical'];v=c.physical
    for n in ('cap_choice','reg_choice','retro_choice','bess_choice'):
        out['phys_'+n]=(np.asarray(v[n])+.5)/len(choice_options(st,n))
    ol,oh,nl,nh,bl,bh=q_bounds(st,c)
    out['existing_q']=fraction(c.operation.existing_q,ol,oh)
    for n,lo,hi in [('new_q',nl,nh),('bess_q',bl,bh)]:out['phys_'+n]=np.where(hi>lo,fraction(np.asarray(v[n]),lo,hi),.5)
    cs=np.zeros(shapes(st)['cap_state']);tap=np.full(shapes(st)['tap'],.5)
    for k in range(st.K):
        for j,o in enumerate(installed(st,c,'cap_choice',k)):
            if o is not None:
                r=p['capacitors'][o];cs[k,:,j]=0. if r['mode']=='fixed' else (np.asarray(v['cap_state'])[k,:,j]+.5)/(r['steps']+1)
        for j,o in enumerate(installed(st,c,'reg_choice',k)):
            if o is not None and p['regulator_sites'][j] in c.plan.trees[k]:tap[k,:,j]=(np.asarray(v['tap'])[k,:,j]+16.5)/33
    out['phys_cap_state']=cs;out['phys_tap']=tap
    return out


def structural_violations(st,c):
    p=st.electrical['physical'];v=c.physical;bad=0.;moves=0.;storage=st.storage_active(c.plan)
    buys=dict(c.plan.storage_buys);choices=choice_options(st,'bess_choice')
    for j,u in enumerate(st.units):
        choice=choices[int(v['bess_choice'][j])]
        if choice is not None and (u['id'] not in buys or choice[1]<buys[u['id']]):bad=max(bad,1.)
    for k in range(st.K):
        for j,o in enumerate(installed(st,c,'cap_choice',k)):
            z=np.asarray(v['cap_state'])[k,:,j]
            if o is None:bad=max(bad,float(np.max(abs(z),initial=0)))
            else:
                r=p['capacitors'][o];bad=max(bad,float(max(np.max(-z,initial=0),np.max(z-r['steps'],initial=0))))
                available=np.asarray(r['electrical_available'])
                bad=max(bad,float(np.max(abs(z[~available]),initial=0)))
                if r['mode']=='fixed':bad=max(bad,float(np.max(abs(z-available),initial=0)))
                for cy in st.cycles:moves=max(moves,float(np.sum(abs(z[cy]-np.roll(z[cy],1)))-r['max_daily_switches']))
        for j,o in enumerate(installed(st,c,'reg_choice',k)):
            z=np.asarray(v['tap'])[k,:,j];present=o is not None and p['regulator_sites'][j] in c.plan.trees[k]
            bad=max(bad,float(np.max(abs(z),initial=0)-(16 if present else 0)))
            if present:
                bad=max(bad,float(np.max(abs(z[~np.asarray(p['regulators'][o]['electrical_available'])]),initial=0)))
                for cy in st.cycles:moves=max(moves,float(np.sum(abs(z[cy]-np.roll(z[cy],1)))-p['regulators'][o]['max_daily_switches']))
    return dict(physical_installation=max(0.,bad),daily_switching=max(0.,moves))


def source_aux(r,nom,q):
    return 0. if abs(q)<=1e-12 else r['standby_MW_per_MW']*nom+r['q_loss_coefficient']*q*q/max(1e-12,r['s_ratio']*nom)


def case_devices(st,c,k,s,ix,edges,charge,discharge,V=None):
    """Network stamps before the flow; technical outputs after it. No control change."""
    p=st.electrical['physical'];v=c.physical;shunts=np.zeros(st.B,complex);inj=np.zeros(st.B,complex)
    active=st.active_capacity(c)[k];aux_operator=0.;aux_producer=0.;credit=0.;sources=[]
    ro=installed(st,c,'retro_choice',k);bo=installed(st,c,'bess_choice',k)
    for kind,gens,rows,qvals in [('new',st.new,p['new_sources'],v['new_q'][k][s]),('bess',st.units,p['bess_sources'],v['bess_q'][k][s]),('old',st.old,p['old_sources'],c.operation.existing_q[k,s])]:
        for j,(g,r,q) in enumerate(zip(gens,rows,qvals)):
            enabled=p['enabled']['new_q'] if kind=='new' else bo[j] is not None if kind=='bess' else ro[j] is not None
            if not enabled:continue
            nom=active[j] if kind=='new' else g['power_MW'] if kind=='bess' and st.storage_active(c.plan)[k,j] else 0. if kind=='bess' else max(st.av[g['id'],ss]['maximum'] for ss in range(st.S))*st.c['generation_growth_by_stage'][k]
            power=c.operation.new_p[k,s,j] if kind=='new' else discharge[k,j,s]-charge[k,j,s] if kind=='bess' else c.operation.existing_p[k,s,j]
            aux=source_aux(r,nom,q)
            inj[ix[g['bus']]]+=(1j*q if kind!='old' else 0)-aux
            if kind=='new' and g['technology'] in ('PV','WT'):aux_producer+=aux;credit+=aux*r['aux_price_per_MWh']
            else:aux_operator+=aux
            volt=abs(V[ix[g['bus']]]) if V is not None else 1.
            current=abs(complex(power,q))/max(volt,1e-12)*st.electrical['Ibase_A']
            sources.append(dict(kind=kind,id=g['id'],bus=g['bus'],P_gross_MW=float(power),Q_MVAr=float(q),aux_MW=float(aux),Snom_MVA=r['s_ratio']*nom,
                                current_A=current,current_limit_A=r['current_ratio']*nom*st.electrical['Ibase_A'],electrical_available=r['electrical_available'][s]))
    capacitors=[]
    for j,o in enumerate(installed(st,c,'cap_choice',k)):
        if o is None:continue
        r=p['capacitors'][o];z=v['cap_state'][k][s][j]/r['steps'];bus=p['capacitor_sites'][j]
        if not r['electrical_available'][s]:z=0.
        # G>=0 consumes active power, B>=0 injects capacitive vars.
        y=r['q_MVAr']*z*complex(r['loss_coefficient'],1);shunts[ix[bus]]+=y
        vv=abs(V[ix[bus]])**2 if V is not None else 1.
        capacitors.append(dict(bus=bus,option=o,steps=v['cap_state'][k][s][j],Q_MVAr=y.imag*vv,loss_MW=y.real*vv))
    regulators=[];byid={e['id']:e for e in edges}
    for j,o in enumerate(installed(st,c,'reg_choice',k)):
        eid=p['regulator_sites'][j]
        if o is None or eid not in byid:continue
        r=p['regulators'][o];e=byid[eid];n=v['tap'][k][s][j]
        if not r['electrical_available'][s]:continue  # explicitly bypassed failure
        e['tap']=1+.00625*n;e['r']+=r['series_r_pu'];e['x']+=r['series_x_pu']
        # Core loss as a positive conductance at the physical input terminal.
        shunts[e['i']]+=r['loss_coefficient']
        regulators.append(dict(line=eid,option=o,tap=n,ratio=e['tap'],current_from_A=r['current_from_A'],current_to_A=r['current_to_A'],
                               core_current_real=float((r['loss_coefficient']*V[e['i']]).real) if V is not None else 0.,
                               core_current_imag=float((r['loss_coefficient']*V[e['i']]).imag) if V is not None else 0.))
    return dict(shunts=shunts,injection=inj,sources=sources,capacitors=capacitors,regulators=regulators,
                auxiliary_operator_MW=aux_operator,auxiliary_producer_MW=aux_producer,auxiliary_reimbursement_per_hour=credit)


def technical_violations(st,c,k,s,devices,V,flows,bounds=None):
    ol,oh,nl,nh,bl,bh=q_bounds(st,c) if bounds is None else bounds;v=c.physical
    def ex(x,lo,hi):return max(0.,float(np.max(np.asarray(lo)-x,initial=0)),float(np.max(x-np.asarray(hi),initial=0)))
    q=max(ex(c.operation.existing_q[k,s],ol[k,s],oh[k,s]),ex(np.asarray(v['new_q'])[k,s],nl[k,s],nh[k,s]),ex(np.asarray(v['bess_q'])[k,s],bl[k,s],bh[k,s]))
    current=0.;power=0.
    for r in devices['sources']:
        current=max(current,(r['current_A']-r['current_limit_A'])/max(1.,r['current_limit_A']))
        power=max(power,(math.hypot(r['P_gross_MW'],r['Q_MVAr'])-r['Snom_MVA'])/max(1.,r['Snom_MVA']))
        if not r['electrical_available']:power=max(power,abs(r['P_gross_MW']),abs(r['Q_MVAr']))
    flow={e['id']:f for e,f in flows}
    for r in devices['regulators']:
        f=flow[r['line']];input_current=f['If']+complex(r['core_current_real'],r['core_current_imag'])
        # The output winding is at the sending end of the cable, AFTER the
        # ideal tap. The remote cable current It differs when line B is nonzero.
        output_current=f['If']/r['ratio']
        current=max(current,abs(input_current)*st.electrical['Ibase_A']/r['current_from_A']-1,abs(output_current)*st.electrical['Ibase_A']/r['current_to_A']-1)
    return dict(physical_Q=q,physical_current=max(0.,current),physical_S=max(0.,power))


def purchase_events(cost,year,life,end,discount,owner,kind,asset):
    return [dict(owner=owner,kind=kind,asset=asset,year=y,replacement=y!=year,nominal_cost=cost,VP=cost*(1+discount)**-y)
            for y in range(year,end,life)]


def ledger(st,c):
    from .economics import weights
    p=st.electrical['physical'];end=st.c['horizon_years'];rate=st.c['discount_rate'];w=weights(st);events=[];maint={'operator':{},'producer':{}}
    def add(cost,year,life,annual,owner,kind,asset,stop=end):
        events.extend(purchase_events(cost,year,life,min(end,stop),rate,owner,kind,asset))
        value=annual*sum((1+rate)**-y for y in range(year+1,min(end,stop)+1))
        maint[owner][kind]=maint[owner].get(kind,0.)+value
    for name,kind,rows,sites in [('cap_choice','capacitors',p['capacitors'],p['capacitor_sites']),('reg_choice','regulators',p['regulators'],p['regulator_sites'])]:
        choices=choice_options(st,name)
        for site,i in zip(sites,c.physical[name]):
            choice=choices[int(i)]
            if choice is not None:
                o,k=choice;r=rows[o];add(r['capex'],st.years[k],r['life_years'],r['maintenance_per_year'],'operator',kind,site)
    for name,kind,gens,rows in [('retro_choice','retrofit_existing',st.old,p['old_sources']),('bess_choice','retrofit_BESS',st.units,p['bess_sources'])]:
        choices=choice_options(st,name)
        for j,(g,r,i) in enumerate(zip(gens,rows,c.physical[name])):
            choice=choices[int(i)]
            if choice is not None:
                _,k=choice;nom=g['power_MW'] if name=='bess_choice' else max(st.av[g['id'],s]['maximum'] for s in range(st.S))*max(st.c['generation_growth_by_stage'])
                cost=nom*r['capex_per_MW'];add(cost,st.years[k],r['life_years'],cost*r['maintenance_fraction'],'operator',kind,g['id'])
    if p['enabled']['new_q']:
        for j,(g,r) in enumerate(zip(st.new,p['new_sources'])):
            for k,year in enumerate(st.years):
                cost=c.new_capacity[k,j]*r['capex_per_MW']
                if cost:
                    owner='producer' if g['technology'] in ('PV','WT') else 'operator';stop=year+g.get('contract',{}).get('duration_years',end)
                    add(cost,year,r['life_years'],cost*r['maintenance_fraction'],owner,'new_Q_interface',g['id'],stop)
    parts={};producer={}
    for owner,out in [('operator',parts),('producer',producer)]:
        for e in events:
            if e['owner']==owner:
                name=('replacement_' if e['replacement'] else 'CAPEX_')+e['kind'];out[name]=out.get(name,0.)+e['VP']
        out.update({'maintenance_'+n:x for n,x in maint[owner].items()})
    # Representative-day switching is weighted once by day occurrences.
    for name,control,kind,rows,sites in [('cap_choice','cap_state','capacitors',p['capacitors'],p['capacitor_sites']),('reg_choice','tap','regulators',p['regulators'],p['regulator_sites'])]:
        cost=0.
        for k in range(st.K):
            for j,o in enumerate(installed(st,c,name,k)):
                if o is None:continue
                z=np.asarray(c.physical[control])[k,:,j]
                for cy in st.cycles:cost+=w[k]*st.c['annual_hours']*st.probabilities[cy[0]]/st.hours[cy[0]]*sum(abs(z[cy]-np.roll(z[cy],1)))*rows[o]['switch_cost']
        parts['switching_'+kind]=float(cost)
    return dict(operator_parts_VP=parts,producer_parts_VP=producer,events=events,
                replacements='identical equipment at exact lifetime; no replacement at horizon endpoint',
                auxiliary_policy='producer repays metered incremental Q auxiliary energy at explicit tariff; ordinary PPA unchanged')
