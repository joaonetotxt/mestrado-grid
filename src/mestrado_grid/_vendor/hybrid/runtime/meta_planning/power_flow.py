# Numerical AC equations isolated from optimization/fluxo.py on 2026-09-21.
"""Balanced AC, fixed PQ buses and one slack. No optimal power flow.

Two independent algorithms: polar Newton using sparse Ybus/Jacobian, and
backward/forward current summation using the physical tree. Voltages and powers
are per unit; the application uses a 1 MVA base.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, bmat
from scipy.sparse.linalg import spsolve

def admittance(n, edges, shunts=None):
    rr=[];cc=[];vv=[]
    for e in edges:
        i,j=e['i'],e['j']; y=1/complex(e['r'],e['x']); sh=1j*e.get('b',0)/2
        a=e.get('tap',1.)
        rr.extend([i,i,j,j]);cc.extend([i,j,i,j]);vv.extend([a*a*(y+sh),-a*y,-a*y,y+sh])
    if shunts is not None:
        rr.extend(range(n));cc.extend(range(n));vv.extend(shunts)
    return coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr()

def polar_jacobian(Y,V):
    D=diags(V);I=Y@V;Dn=diags(V/np.abs(V));Di=diags(I)
    dvm=D@(Y@Dn).conjugate()+Di.conjugate()@Dn
    dva=1j*D@(Di-Y@D).conjugate()
    return bmat([[dva[1:,1:].real,dvm[1:,1:].real],
                 [dva[1:,1:].imag,dvm[1:,1:].imag]],format='csr')

def mismatch(Y,V,S):
    ds=V*np.conjugate(Y@V)-S
    return np.r_[ds[1:].real,ds[1:].imag]

def tree_voltage_start(n,edges,vslack=1.04):
    """Independent Newton guess following existing tap ratios along the tree.

    No injection, investment, tap decision or physical tolerance is changed.
    Shunts and load are omitted only in this initial voltage estimate.
    """
    parent,pe,order=tree_order(n,edges)
    V=np.empty(n,complex);V[0]=complex(vslack)
    for j in order[1:]:
        e=edges[pe[j]];a=e.get('tap',1.)
        if not np.isfinite(a) or a<=0:raise ValueError('Positive finite tap required')
        V[j]=V[parent[j]]*(a if parent[j]==e['i'] else 1/a)
    if not np.isfinite(V).all():raise ValueError('Nonfinite transformer start')
    return V

def newton(n,edges,S,vslack=1.04,tol=2e-9,maxiter=50,start=None,shunts=None):
    Y=admittance(n,edges,shunts);V=tree_voltage_start(n,edges,vslack) if start is None else np.array(start,complex)
    V[0]=vslack; hist=[]
    for it in range(maxiter+1):
        f=mismatch(Y,V,S);err=float(np.max(abs(f),initial=0.));hist.append(err)
        if err<=tol: return dict(converged=True,V=V,iterations=it,residual=err,history=hist)
        if it==maxiter:break
        dx=spsolve(polar_jacobian(Y,V),-f)
        if not np.all(np.isfinite(dx)):break
        accepted=False
        for step in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            vm=abs(V).copy();va=np.angle(V).copy();va[1:]+=step*dx[:n-1];vm[1:]+=step*dx[n-1:]
            if min(vm)<.05:continue
            trial=vm*np.exp(1j*va)
            if np.max(abs(mismatch(Y,trial,S)))<err:
                V=trial;accepted=True;break
        if not accepted:break
    return dict(converged=False,V=V,iterations=it,residual=err,history=hist)

def tree_order(n,edges):
    assert len(edges)==n-1, 'not a tree: edge count'
    adj=[[] for _ in range(n)]
    for ix,e in enumerate(edges):
        adj[e['i']].append((e['j'],ix));adj[e['j']].append((e['i'],ix))
    parent=np.full(n,-1,int);parent_edge=np.full(n,-1,int);parent[0]=0;order=[0]
    for b in order:
        for j,ix in adj[b]:
            if j==parent[b]:continue
            assert parent[j]<0, 'cycle or duplicate edge'
            parent[j]=b;parent_edge[j]=ix;order.append(j)
    assert len(order)==n, 'disconnected'
    return parent,parent_edge,order

def sweep(n,edges,S,vslack=1.04,tol=2e-12,maxiter=1000,shunts=None):
    parent,pe,order=tree_order(n,edges);V=np.full(n,complex(vslack));sh=np.zeros(n,complex)
    err=float('inf')
    if shunts is not None:sh+=shunts
    for e in edges:
        sh[e['i']]+=e.get('tap',1.)**2*1j*e.get('b',0)/2;sh[e['j']]+=1j*e.get('b',0)/2
    for it in range(1,maxiter+1):
        if min(abs(V))<.05:break
        currents=np.conj(-S/V)+sh*V
        for j in reversed(order[1:]):
            e=edges[pe[j]];a=e.get('tap',1.);t=a if parent[j]==e['i'] else 1/a
            currents[parent[j]]+=t*currents[j]
        new=np.empty(n,complex);new[0]=vslack
        for j in order[1:]:
            e=edges[pe[j]];a=e.get('tap',1.);forward=parent[j]==e['i']
            t=a if forward else 1/a;z=complex(e['r'],e['x'])/(1 if forward else a*a)
            new[j]=t*new[parent[j]]-z*currents[j]
        err=float(max(abs(new-V)));V=new
        if not np.all(np.isfinite(V)):break
        if err<tol:return dict(converged=True,V=V,iterations=it,voltage_change=err)
    return dict(converged=False,V=V,iterations=it,voltage_change=err)

def branch_flows(V,edges):
    out=[]
    for e in edges:
        vf,vt=V[e['i']],V[e['j']];a=e.get('tap',1.);series=(a*vf-vt)/complex(e['r'],e['x'])
        If=a*(series+1j*e.get('b',0)/2*a*vf);It=-series+1j*e.get('b',0)/2*vt
        Sf=vf*np.conj(If);St=vt*np.conj(It)
        out.append(dict(If=If,It=It,Sf=Sf,St=St,series=series))
    return out
