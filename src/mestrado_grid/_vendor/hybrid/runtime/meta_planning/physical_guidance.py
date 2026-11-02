"""Measured local AC responses; never a dispatch optimizer or feasibility test.

The operating point is supplied by the unchanged AC evaluator. Linear response
uses its exact Ybus/Jacobian, including taps and shunts. All predictions are
local, and every search candidate is evaluated with the original full AC.
"""
import copy
import numpy as np
from scipy.sparse.linalg import splu
from . import power_flow as pf, physical
from .evaluator import Evaluator
from .temporal import storage_dispatch
from .topology import line_schedule


def network_case(st,c,k,s):
    installed,_=line_schedule(st.data,c.plan)
    ordered,ix,edges=Evaluator(st)._edges(c,k,installed)
    charge,discharge=storage_dispatch(st,c)
    shunts=None
    if 'physical' in st.electrical:
        devices=physical.case_devices(st,c,k,s,ix,edges,charge,discharge)
        shunts=devices['shunts']
    return ordered,edges,shunts


def observables(st,V,edges):
    flows=pf.branch_flows(V,edges)
    # Signed branch P/Q, magnitudes for physical limits. Current conventions
    # exactly follow evaluator._case, including its transformer-side division.
    return dict(voltage=np.abs(V),
        branch_P=np.array([f['Sf'].real for f in flows]),
        branch_Q=np.array([f['Sf'].imag for f in flows]),
        branch_I=np.array([max(abs(f['If']/e.get('tap',1.)),abs(f['It']))*st.electrical['Ibase_A']/e['Imax_A'] for e,f in zip(edges,flows)]),
        branch_S=np.array([max(abs(f['Sf']),abs(f['St']))/e['Smax_MVA'] for e,f in zip(edges,flows)]))


def response(st,c,k,s,V):
    ordered,edges,shunts=network_case(st,c,k,s)
    V=np.asarray(V,complex);n=len(V);Y=pf.admittance(n,edges,shunts)
    J=pf.polar_jacobian(Y,V);factor=splu(J.tocsc())
    inverse=factor.solve(np.eye(2*(n-1)))
    dV=np.zeros((n,2*(n-1)),complex)
    dV[1:]=1j*V[1:,None]*inverse[:n-1]+(V[1:]/abs(V[1:]))[:,None]*inverse[n-1:]
    base=observables(st,V,edges);derivatives={key:[] for key in base}
    # Numerical differentiation only of algebraic branch observables, not AC
    # solves. A small voltage displacement checks both transformer directions.
    h=1e-5
    for j in range(dV.shape[1]):
        a=observables(st,V+h*dV[:,j],edges);b=observables(st,V-h*dV[:,j],edges)
        for key in base:derivatives[key].append((a[key]-b[key])/(2*h))
    derivatives={key:np.asarray(value).T for key,value in derivatives.items()}
    I=Y@V;dS=dV*np.conj(I[:,None])+V[:,None]*np.conj(Y@dV)
    derivatives['sub_P']=dS[0].real;derivatives['sub_Q']=dS[0].imag
    derivatives['loss']=dS.real.sum(axis=0)
    order=[st.bi[b] for b in ordered]
    return dict(candidate_id=c.candidate_id,plan_id=c.plan_id,stage=k,state=s,
        buses=ordered,edges=[e['id'] for e in edges],V=V,base=base,derivatives=derivatives,
        voltage_lower=np.array(st.electrical['vmin'])[order],voltage_upper=np.array(st.electrical['vmax'])[order],
        jacobian_residual=float(np.max(abs(J@inverse-np.eye(2*(n-1))))),
        scope='fixed topology, taps, shunts and supplied operating point; local injection response',
        extra_AC_solves=0)


def check_response(st,c,model,bus,reactive=False,step=1e-4):
    """Independent nonlinear NR+sweep probes for a measured injection column.

    These synthetic injection probes validate derivatives, not investable plans
    or feasible solutions. Both signs and all solves are explicitly counted.
    """
    k,s=model['stage'],model['state'];ordered,edges,shunts=network_case(st,c,k,s)
    V=model['V'];Y=pf.admittance(len(V),edges,shunts);S=V*np.conj(Y@V)
    i=ordered.index(bus);assert i>0
    col=i-1+(len(V)-1)*int(reactive);rows=[]
    for sign in (-1,1):
        trial=S.copy();trial[i]+=sign*step*(1j if reactive else 1.)
        nr=pf.newton(len(V),edges,trial,vslack=st.electrical['vslack'],start=V,shunts=shunts)
        sw=pf.sweep(len(V),edges,trial,vslack=st.electrical['vslack'],shunts=shunts)
        assert nr['converged'] and sw['converged']
        rows.append(observables(st,nr['V'],edges))
        assert np.max(abs(nr['V']-sw['V']))<1e-8
    errors={}
    for key in ('voltage','branch_P','branch_Q','branch_I','branch_S'):
        actual=(rows[1][key]-rows[0][key])/(2*step)
        predicted=model['derivatives'][key][:,col]
        errors[key]=dict(max_abs=float(np.max(abs(actual-predicted))),
            relative=float(np.linalg.norm(actual-predicted)/max(1e-9,np.linalg.norm(actual))))
    return dict(bus=bus,reactive=reactive,step=step,errors=errors,AC_cases=2,flow_solves=4)


def interaction_score(model,load_bus,resource_bus,q_ratio,reactive=False):
    """Electrical response similarity, independent of technology names/sites.

    A resource with the same response as restored load can compensate locally.
    This selects a grouping, never a setpoint or an acceptance decision.
    """
    buses=model['buses'];n=len(buses)-1
    if load_bus==buses[0] or resource_bus==buses[0]:return 0.
    a=buses.index(load_bus)-1;b=buses.index(resource_bus)-1
    pieces=[];others=[]
    for key,scale in (('voltage',.05),('branch_I',1.),('branch_S',1.)):
        d=model['derivatives'][key]
        pieces.append((d[:,a]+q_ratio*d[:,a+n])/scale)
        others.append(d[:,b+n if reactive else b]/scale)
    x=np.concatenate(pieces);y=np.concatenate(others)
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return 0. if den<1e-15 else float(x@y/den)
