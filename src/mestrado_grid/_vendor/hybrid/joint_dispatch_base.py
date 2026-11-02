"""Isolated auxiliary daily P/Q, DR, EV and cyclic SOC proposals.

Linear models only propose controls. Original nonlinear AC and original
financial/temporal/reliability rules accept the complete portfolio.
No simultaneous charging/discharging is admitted to the actual candidate.
"""
from common import *
from collections import Counter
from scipy import sparse
from meta_planning.physical_guidance import response

class JointRefiner:
    def __init__(self,st,flex=True):
        self.st=st;self.ev=Evaluator(st);self.flex=flex;self.counts=Counter();self.history=[]
    def evaluate(self,c):
        self.counts['full_evaluations']+=1;self.counts['AC_cases']+=self.st.K*self.st.S
        return self.ev.evaluate(c)
    def propose(self,c,k,cy,rows,radius=.5):
        st=self.st;op=c.operation;n=st.B-1;L=len(cy)
        models=[response(st,c,k,s,np.array(rows[k,s]['voltage_real'])+1j*np.array(rows[k,s]['voltage_imag'])) for s in cy]
        bybus={b:i-1 for i,b in enumerate(models[0]['buses'])};lo,hi,*_=physical.existing_bounds(st,c)
        ql,qh,nl,nh,bl,bh=physical.q_bounds(st,c);cl,ch,el,eh=demand_bounds(st)
        active=st.active_capacity(c)[k];av=physical.availability(st);charge,discharge=storage_dispatch(st,c)
        names=[];bounds=[];cost=[];maps=[[] for _ in cy];offset=np.zeros((L,2*n));eq={};ubextra=[]
        def add(name,s,j,bus,p,q,low,high,direct=0.):
            if high-low<1e-12:return None
            ix=len(names);names.append((name,s,j));bounds.append((float(low),float(high)));cost.append(float(direct))
            if bus is not None:
                a=cy.index(s);b=bybus[bus];assert b>=0
                if p:maps[a].append((b,ix,p))
                if q:maps[a].append((b+n,ix,q))
            return ix
        def eql(key,ix,v):
            if ix is not None:eq.setdefault(key,{})[ix]=v
        def weighted(s,x):return self.scenario_weights[s]*x
        for s in cy:
            a=cy.index(s)
            for b,bus in enumerate(st.buses):
                if bus==st.data['root']:continue
                load=op.conventional[k,b,s];cut=op.cut[k,b,s];ratio=st.electrical['q_ratio'][b][s]
                ii=add('restore',s,b,bus,-1,-ratio,-min(load-cut,radius*max(load,.01)),min(cut,radius*max(load,.01)),weighted(s,-st.c['unserved_energy_per_MWh']))
                if self.flex:
                    for name,base,lower,upper,price,pq in [('conventional',st.base_load,cl,ch,st.data['extension']['flex']['shift_compensation_per_MWh'],ratio),('ev',st.base_ev,el,eh,st.data['extension']['flex']['ev_compensation_per_MWh'],0.)]:
                        val=getattr(op,name)[k,b,s];lower_value=lower[k,b,s]
                        if name=='conventional':lower_value=max(lower_value,cut)
                        z=add(name,s,b,bus,-1,-pq,lower_value-val,upper[k,b,s]-val)
                        if z is None:continue
                        eql((name,b),z,st.hours[s])
                        # Epigraph of compensation max(base - demand, 0).
                        e=add('compensation',s,b,None,0,0,0,max(base[k,b,s]-lower_value,0)+1e-9,weighted(s,price))
                        if e is not None:ubextra.append(({z:-1.,e:-1.},val-base[k,b,s]))
                        if ii is not None and name=='conventional':ubextra.append(({ii:-1.,z:-1.},load-cut))
            for j,g in enumerate(st.old):
                p=op.existing_p[k,s,j];span=radius*max(hi[k,s,j],.01)
                direct=g['price']+st.c['carbon_price_per_ton']*st.av[g['id'],s]['emissions']-(g['curtail_price'] if g['renewable'] else 0)
                add('existing_p',s,j,g['bus'],1,0,max(lo[k,s,j]-p,-span),min(hi[k,s,j]-p,span),weighted(s,direct))
                q=op.existing_q[k,s,j];add('existing_q',s,j,g['bus'],0,1,max(ql[k,s,j]-q,-span),min(qh[k,s,j]-q,span))
            for j,g in enumerate(st.new):
                mx=active[j]*av[s,j];p=op.new_p[k,s,j];span=radius*max(mx,.01)
                direct=g['variable_per_MWh']+st.c['carbon_price_per_ton']*g['emissions_ton_per_MWh']-(g['curtail_per_MWh'] if g['technology'] in ('WT','PV') else 0)
                add('new_p',s,j,g['bus'],1,0,max(-p,-span),min(mx-p,span),weighted(s,direct))
                q=c.physical['new_q'][k][s][j];add('new_q',s,j,g['bus'],0,1,max(nl[k,s,j]-q,-span),min(nh[k,s,j]-q,span))
        storage=[]
        for j,u in enumerate(st.units):
            if not st.storage_active(c.plan)[k,j]:continue
            idx={}
            for s in cy:
                a=cy.index(s);b=bybus[u['bus']]
                # Absolute new charge/discharge, referenced to current injections.
                offset[a,b]+=charge[k,j,s]-discharge[k,j,s]
                ci=add('charge',s,j,u['bus'],-1,0,0,u['power_MW'])
                di=add('discharge',s,j,u['bus'],1,0,0,u['power_MW'],weighted(s,st.c['prices_per_MWh']['ESS']+st.c['carbon_price_per_ton']*st.c['emissions_ton_per_MWh']['ESS']))
                ei=add('soc_abs',s,j,None,0,0,0,u['energy_MWh']);idx[s]=(ci,di,ei)
                q=c.physical['bess_q'][k][s][j];add('bess_q',s,j,u['bus'],0,1,bl[k,s,j]-q,bh[k,s,j]-q)
            for a,s in enumerate(cy):
                ci,di,ei=idx[s];nxt=idx[cy[(a+1)%L]][2]
                eq['soc',j,s]={nxt:1.,ei:-1.,ci:-u['eta_charge']*st.hours[s],di:st.hours[s]/u['eta_discharge']}
            storage.append((j,idx))
        N=len(names);ineq=[];rhs=[]
        for a,s in enumerate(cy):
            rr,cc,vv=zip(*maps[a]);A=sparse.csr_matrix((vv,(rr,cc)),shape=(2*n,N));m=models[a];dd=m['derivatives'];base=m['base'];row=rows[k,s]
            def limits(der,low,high,val):
                der=np.atleast_2d(der);G=sparse.csr_matrix(der)@A;value=np.atleast_1d(val)+der@offset[a]
                ineq.extend((G,-G));rhs.extend((np.broadcast_to(high,value.shape)-value,value-np.broadcast_to(low,value.shape)))
            limits(dd['voltage'],m['voltage_lower'],m['voltage_upper'],base['voltage'])
            limits(dd['branch_I'],0.,.999,base['branch_I']);limits(dd['branch_S'],0.,.999,base['branch_S'])
            pl,pu,qlow,qup,sm=physical.sub_limits(st,k,s);sp,sq=row['sub_P_MW'],row['sub_Q_MVAr']
            limits(dd['sub_P'],pl,pu,sp);limits(dd['sub_Q'],qlow,qup,sq)
            limits((sp*dd['sub_P']+sq*dd['sub_Q'])/max(np.hypot(sp,sq),1e-9),0.,sm*.999,np.hypot(sp,sq))
            price=st.sub['price']+st.c['carbon_price_per_ton']*st.av[st.sub['id'],s]['emissions']
            grad=weighted(s,price*dd['sub_P']+st.c['loss_cost_per_MWh']*dd['loss'])
            cost=np.asarray(cost)+np.asarray(A.T@grad).ravel()
        def matrix(dicts):
            rr=[];cc=[];vv=[]
            for i,d in enumerate(dicts):
                for j,v in d.items():rr.append(i);cc.append(j);vv.append(v)
            return sparse.csr_matrix((vv,(rr,cc)),shape=(len(dicts),N))
        if ubextra:ineq.append(matrix([d for d,b in ubextra]));rhs.append(np.array([b for d,b in ubextra]))
        Aub=sparse.vstack(ineq,format='csr');bub=np.concatenate(rhs);Aeq=matrix(list(eq.values()));beq=np.zeros(len(eq))
        def solve(bd):
            self.counts['LP_solves']+=1
            return ca.linprog(np.asarray(cost)/3000,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bd,method='highs')
        res=solve(bounds)
        if not res.success:return None,dict(success=False,lp_status=res.status)
        # Relaxation only chooses a charge/discharge direction. Re-solve the
        # entire P/Q+SOC problem with mutually exclusive directions enforced.
        if storage:
            locked=list(bounds)
            for j,idx in storage:
                directions={s:res.x[idx[s][1]]-res.x[idx[s][0]] for s in cy}
                if not any(v>1e-8 for v in directions.values()) or not any(v<-1e-8 for v in directions.values()):
                    order=sorted(cy,key=lambda s:directions[s]);charging=set(order[:L//2])
                else:charging={s for s in cy if directions[s]<=0}
                for s,(ci,di,ei) in idx.items():locked[di if s in charging else ci]=(0.,0.)
            res=solve(locked)
            if not res.success:return None,dict(success=False,lp_status=res.status,direction_locked=True)
            assert all(min(res.x[ci],res.x[di])<1e-8 for j,idx in storage for ci,di,ei in idx.values())
        return (names,res.x),dict(success=True,variables=N,storage_units=len(storage),direction_locked=bool(storage))
    def apply(self,c,k,cy,step,f):
        st=self.st;out=copy.deepcopy(c);names,values=step
        for (name,s,j),value in zip(names,values):
            if name=='restore':out.operation.cut[k,j,s]-=f*value
            elif name in ('existing_p','existing_q','new_p'):getattr(out.operation,name)[k,s,j]+=f*value
            elif name in ('conventional','ev'):getattr(out.operation,name)[k,j,s]+=f*value
            elif name in ('new_q','bess_q'):out.physical[name][k][s][j]+=f*value
            elif name=='soc_abs':out.operation.soc[k,j,s]=(1-f)*c.operation.soc[k,j,s]+f*value
        # Only numerical-bound projection; daily energy is conserved algebraically.
        normalize(st,out);out.operation.cut=np.clip(out.operation.cut,0.,out.operation.conventional)
        lo,hi,*_=physical.existing_bounds(st,out);out.operation.existing_p=np.clip(out.operation.existing_p,lo,hi)
        out.operation.new_p=np.clip(out.operation.new_p,0.,st.active_capacity(out)[:,None,:]*physical.availability(st)[None,:,:])
        ol,oh,nl,nh,bl,bh=physical.q_bounds(st,out)
        out.operation.existing_q=np.clip(out.operation.existing_q,ol,oh)
        out.physical['new_q']=np.clip(out.physical['new_q'],nl,nh).tolist();out.physical['bess_q']=np.clip(out.physical['bess_q'],bl,bh).tolist()
        return out
    def refine(self,c,rounds=1,callback=None):
        st=self.st;c=copy.deepcopy(c);br=self.evaluate(c);best=copy.deepcopy(c)
        if br['failed_cases'] or not all(x['passed'] for x in br['reliability']) or any(v>self.ev.policy.temporal_tolerance for v in br['temporal_violations'].values()):return c,br
        rows={(r['stage'],r['state_id']):r for r in br['cases']}
        for it in range(rounds):
            for k in range(st.K):
                for cy in st.cycles:
                    step,meta=self.propose(c,k,cy,rows);accepted=False;fraction=0
                    if step:
                        oldcost=sum(st.probabilities[s]*sum(rows[k,s]['hourly_costs'].values()) for s in cy)
                        oldv=sum(sum(rows[k,s]['normalized_violations'].values()) for s in cy)
                        for fraction in (1.,.5,.25,.125,.0625):
                            trial=self.apply(c,k,cy,step,fraction);ins,_=line_schedule(st.data,trial.plan);top=self.ev._edges(trial,k,ins)
                            ch,di=storage_dispatch(st,trial);bd=ca.operating_bounds(st,trial)
                            rr=[self.ev._case(trial,k,s,top,ch,di,bd) for s in cy];self.counts['AC_cases']+=len(rr)
                            self.counts['local_day_trials']+=1
                            if all(r['converged'] for r in rr):
                                cost=sum(st.probabilities[r['state_id']]*sum(r['hourly_costs'].values()) for r in rr)
                                violation=sum(sum(r['normalized_violations'].values()) for r in rr)
                                if (all(r['passed'] for r in rr) and (oldv>0 or cost<oldcost-1e-7)) or (oldv>0 and violation<oldv-1e-8):
                                    c=trial;rows.update({(k,r['state_id']):r for r in rr});accepted=True;break
                    self.history.append(dict(iteration=it,stage=k,cycle=cy,accepted=accepted,fraction=fraction,**meta))
                    if callback:callback(self,k,cy)
            result=self.evaluate(c)
            if rank(result)<rank(br):best=copy.deepcopy(c);br=result
            else:c=copy.deepcopy(best);rows={(r['stage'],r['state_id']):r for r in br['cases']}
        return best,br

