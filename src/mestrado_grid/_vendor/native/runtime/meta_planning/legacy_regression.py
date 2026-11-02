"""OFFLINE ONLY: translate archived controls for regression, never initialization.

No model is constructed and no old result is injected into the direct search.
"""
import csv
import numpy as np
from .schema import Candidate,Plan,Operation,read


def read_vector(path):
    with open(path,encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    result={r['variable']:float(r['value']) for r in rows}
    if len(result)!=len(rows) or not all(np.isfinite(x) for x in result.values()):
        raise ValueError('Duplicate or nonfinite archived variables')
    return result


def from_archive(st,values,ac_directory=None,label='b000'):
    def get(family,*ix):return values[f'{family}[{",".join(map(str,ix))}]']
    plan=Plan([[e['id'] for e in st.lines if sum(get('arc',e['id'],o,di,k) for o in (0,1) for di in (0,1))>.5] for k in range(st.K)],
              [(e['id'],o,k) for e in st.lines for o in (0,1) for k in range(st.K) if get('buy',e['id'],o,k)>.5],
              [(u['id'],k) for u in st.units for k in range(st.K) if get('store_buy',u['id'],k)>.5]).canonical()
    capacity=np.array([[get('dg_buy',g['id'],k) for g in st.new] for k in range(st.K)]).reshape(st.K,st.N)
    ep=np.array([[[get('generation',g['id'],k,s,0) for g in st.old] for s in range(st.S)] for k in range(st.K)]).reshape(st.K,st.S,st.G)
    eq=np.zeros(ep.shape)
    npow=np.array([[[get('dg_gen',g['id'],k,s,0) for g in st.new] for s in range(st.S)] for k in range(st.K)]).reshape(st.K,st.S,st.N)
    soc=np.zeros((st.K,st.U,st.S))
    for k in range(st.K):
        for u,unit in enumerate(st.units):
            for day,cy in enumerate(st.cycles):
                soc[k,u,cy[0]]=get('storage_day_start',unit['id'],k,day)
                for previous,now in zip(cy[:-1],cy[1:]):soc[k,u,now]=get('SOC',unit['id'],k,previous,0)
    shifts={name:np.array([[[values.get(f'{name}[{b},{k},{s},0]',0.) for s in range(st.S)] for b in st.buses] for k in range(st.K)]) for name in ('dr_down','dr_up','ev_down','ev_up')}
    cut=np.array([[[get('shed',b,k,s,0) for s in range(st.S)] for b in st.buses] for k in range(st.K)])
    if ac_directory is not None:
        from pathlib import Path
        for k,year in enumerate(st.years):
            for s in range(st.S):
                record=read(Path(ac_directory)/f'{label}_a{year:02d}_e{s+1:02d}.json')
                gs={g['id']:g for g in record['generators']}
                for j,g in enumerate(st.old):
                    ep[k,s,j]=gs[f'existente_{g["id"]}']['P_corrigido'];eq[k,s,j]=gs[f'existente_{g["id"]}']['Q_corrigido']
                for j,g in enumerate(st.new):npow[k,s,j]=gs.get(f'novo_{g["id"]}',{}).get('P_corrigido',0.)
                cut[k,:,s]=record['total_cut']
    op=Operation(ep,eq,npow,soc,st.base_load-shifts['dr_down']+shifts['dr_up'],st.base_ev-shifts['ev_down']+shifts['ev_up'],cut)
    candidate=Candidate(plan,capacity,op);st.shape_check(candidate)
    # Audit the omitted final SOC boundary and independently stored P schedules:
    # a malformed historic vector cannot be silently made cyclic by conversion.
    from .temporal import storage_dispatch
    charge,discharge=storage_dispatch(st,candidate)
    max_error=0.
    for k in range(st.K):
        for u,unit in enumerate(st.units):
            for day,cy in enumerate(st.cycles):
                max_error=max(max_error,abs(get('SOC',unit['id'],k,cy[-1],0)-soc[k,u,cy[0]]))
            for s in range(st.S):
                max_error=max(max_error,abs(charge[k,u,s]-get('charge',unit['id'],k,s,0)),abs(discharge[k,u,s]-get('discharge',unit['id'],k,s,0)))
    if max_error>1e-6:raise ValueError('Historic storage schedule disagrees with its SOC boundaries')
    return candidate,dict(source='offline_historical_regression_only',storage_max_error=max_error)
