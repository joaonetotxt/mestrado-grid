"""AC simulation of supplied controls, with explicit violations and no redispatch."""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy
import hashlib
import time
import warnings
from importlib.metadata import version
import numpy as np
from . import power_flow as pf
from . import economics
from .schema import VERSION,Candidate,fingerprint,plain,read,write_new
from .temporal import violations,storage_dispatch
from .topology import line_schedule,reliability


@dataclass(frozen=True)
class Policy:
    newton_iterations: int=50
    sweep_iterations: int=1000
    constraint_tolerance: float=1e-6
    temporal_tolerance: float=1e-8
    balance_tolerance_MVA: float=1e-7
    cross_tolerance_pu: float=1e-8

    def __post_init__(self):
        if any(type(x)!=int or x<=0 for x in (self.newton_iterations,self.sweep_iterations)):
            raise ValueError('Positive integer flow budgets required')
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)) or not np.isfinite(x) or x<=0 for x in (self.constraint_tolerance,self.temporal_tolerance,self.balance_tolerance_MVA,self.cross_tolerance_pu)):
            raise ValueError('Positive finite tolerances required')


def source_hash():
    return fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')})


def rank(result):
    """Comparable tuples: feasible, measured violations, then numerical/structural failure."""
    experimental=result.get('construction_selection')
    physical_score=None
    if experimental is not None:
        if experimental.get('policy')!='native_initial_energy_v1':raise ValueError('Unknown construction selection')
        values=[experimental[k] for k in ('load_cut_MWh_year','connected_curtailment_MWh_year')]
        if any(isinstance(v,bool) or not np.isfinite(v) or v<0 for v in values):raise ValueError('Invalid construction energies')
        physical_score=float(sum(values))
    if result['status']=='feasible':
        metrics=result['metrics'];financial_policy=None
        if 'financial_policy' in metrics:
            from .financial_policy import resolve, VP30
            financial_policy=resolve({'selection_financial_policy':metrics['financial_policy']})
            if 'selection_objective' not in metrics:
                raise ValueError('Declared financial policy requires selection_objective')
        # Unnamed legacy/synthetic records retain their historical VP fitness.
        value=float(metrics['selection_objective'] if 'selection_objective' in metrics else metrics['objective_VP'])
        if not np.isfinite(value):raise ValueError('Nonfinite feasible objective')
        if financial_policy is not None and financial_policy==VP30 and value!=float(metrics['objective_VP']):
            raise ValueError('VP30 selected fitness differs from objective_VP')
        return (0,physical_score,value) if experimental is not None else (0,value)
    if result['status']=='violated':
        value=float(result['normalized_violation'])
        if not np.isfinite(value) or value<=0:raise ValueError('Invalid measured violation')
        if result.get('search_selection')=='feasibility_violation_exact_cost_tie_v1':
            if experimental is not None:raise ValueError('Incompatible selection profiles')
            metrics=result.get('metrics') or {}
            cost=metrics.get('selection_objective')
            if cost is None or not np.isfinite(cost) or result.get('failed_cases')!=0:
                raise ValueError('Cost tiebreak requires complete converged evaluation')
            return (1,value,float(cost))
        # No financial ranking of infeasible candidates. Energy only breaks an
        # equal measured-violation score during the explicitly separate phase.
        return (1,value,physical_score) if experimental is not None else (1,value)
    if result['status']=='inconclusive':
        # No cost is inferred from partial AC results. Among numerical failures,
        # retain a common direction: fewer failed cases, then measured violations.
        if 'failed_cases' not in result:return (2,float('inf'),float('inf'))
        failed=result['failed_cases'];count=result['expected_cases'];value=result['normalized_violation']
        if type(failed)!=int or type(count)!=int or count<=0 or not 0<=failed<=count or not np.isfinite(value) or value<0:
            raise ValueError('Invalid partial AC diagnostics')
        return (2,failed/count,float(value))
    if result['status']=='invalid_structure':return (3,float('inf'))
    raise ValueError('Unknown evaluation status')


class Evaluator:
    def __init__(self,study,policy=None):
        self.study=study;self.policy=policy or Policy();self.code=source_hash()
        self.versions={name:version(name) for name in ('numpy','scipy')}
        self.signature=fingerprint(dict(input=study.input_id,policy=asdict(self.policy),code=self.code,version=VERSION,dependencies=self.versions))

    def _edges(self,candidate,k,installed):
        st=self.study;ordered=[st.data['root']]+[b for b in st.buses if b!=st.data['root']]
        ix={b:i for i,b in enumerate(ordered)};byid={e['id']:e for e in st.lines};edges=[]
        for name in candidate.plan.trees[k]:
            e=byid[name];option=installed[name,k];o=e['options'][option];raw=st.eopts[name,option]
            edges.append(dict(id=name,option=option,i=ix[e['i']],j=ix[e['j']],r=raw['r'],x=raw['x'],b=raw['b'],
                Imax_A=o['nominal_current_A'],Smax_MVA=o['capacity_MW'],
                FOR=[f*r/(8760-r) for f,r in zip(o['failure_rate'],o['repair_hours'])]))
        return ordered,ix,edges

    def _case(self,candidate,k,s,topology,charge,discharge,bounds):
        st=self.study;op=candidate.operation;pol=self.policy;ordered,ix,edges=topology
        order=[st.bi[b] for b in ordered]
        ratio=np.array(st.electrical['q_ratio'])[order,s]
        conv=op.conventional[k,order,s];ev=op.ev[k,order,s];cut=op.cut[k,order,s]
        injection=-(conv+ev-cut).astype(complex)-1j*ratio*(conv-cut)
        for j,g in enumerate(st.old):injection[ix[g['bus']]]+=op.existing_p[k,s,j]+1j*op.existing_q[k,s,j]
        for j,g in enumerate(st.new):injection[ix[g['bus']]]+=op.new_p[k,s,j]
        for j,u in enumerate(st.units):injection[ix[u['bus']]]+=discharge[k,j,s]-charge[k,j,s]
        extended='physical' in st.electrical;shunts=None;devices=None
        if extended:
            from . import physical
            edges=copy.deepcopy(edges)
            devices=physical.case_devices(st,candidate,k,s,ix,edges,charge,discharge)
            injection+=devices['injection'];shunts=devices['shunts']
        row=dict(stage=k,year=st.years[k],state_id=s,converged=False,passed=False)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                nr=pf.newton(st.B,edges,injection,vslack=st.electrical['vslack'],maxiter=pol.newton_iterations,shunts=shunts)
                sw=pf.sweep(st.B,edges,injection,vslack=st.electrical['vslack'],maxiter=pol.sweep_iterations,shunts=shunts)
            row['newton_iterations']=nr['iterations'];row['sweep_iterations']=sw['iterations']
            row['warnings']=[str(w.message) for w in caught]
            if not nr['converged'] or not sw['converged']:
                row['failure']='flow_nonconvergence';return row
            V=nr['V'];calc=V*np.conj(pf.admittance(st.B,edges,shunts)@V)
            # Root local load/resources are supported; balance generation is NOT
            # blindly equated to network injection when the root has local demand.
            sub=calc[0]-injection[0];balanced=injection.copy();balanced[0]+=sub
            residual=float(np.max(abs(calc-balanced)));cross=float(np.max(abs(V-sw['V'])))
            flows=pf.branch_flows(V,edges)
            if not all(np.all(np.isfinite(a)) for a in (injection,V,sw['V'],calc,sub,residual,cross)) or not all(np.isfinite(v) for f in flows for v in f.values()):
                row['failure']='nonfinite_flow';return row
        except (ValueError,ArithmeticError,RuntimeError,np.linalg.LinAlgError) as exc:
            row['failure']='flow_exception';row['error']=str(exc);return row
        lo,hi,qlo,qhi=bounds[:4]
        if extended:qlo,qhi=bounds[4][:2]
        def excess(x,lower,upper):
            return float(max(0.,np.max(np.asarray(lower)-x,initial=0.),np.max(x-np.asarray(upper),initial=0.)))
        sub_av=st.av[st.sub['id'],s];sub_hi=sub_av['maximum']*st.c['generation_growth_by_stage'][k]
        sub_lo=max(0.,sub_av['minimum']*st.c['generation_growth_by_stage'][k])
        sub_qlo,sub_qhi=st.electrical['generator_q'][str(st.sub['id'])] if sub_hi>1e-9 else (0.,0.)
        sub_s=st.c['substation_rated_MVA']*sub_hi/st.c['substation_active_limit_MW']
        if extended:sub_lo,sub_hi,sub_qlo,sub_qhi,sub_s=physical.sub_limits(st,k,s)
        active=st.active_capacity(candidate)[k]
        new_hi=active*np.array([g['availability'][str(s)] for g in st.new])
        if extended:new_hi=active*physical.availability(st)[s]
        raw=dict(voltage=excess(abs(V),np.array(st.electrical['vmin'])[order],np.array(st.electrical['vmax'])[order]),
                 P=max(excess(op.existing_p[k,s],lo[k,s],hi[k,s]),excess(op.new_p[k,s],0.,new_hi),excess(np.array([sub.real]),sub_lo,sub_hi)),
                 Q=max(excess(op.existing_q[k,s],qlo[k,s],qhi[k,s]),excess(np.array([sub.imag]),sub_qlo,sub_qhi)),
                 branch_I=max([0.]+[max(abs(f['If']/e.get('tap',1.)),abs(f['It']))*st.electrical['Ibase_A']/e['Imax_A']-1 for e,f in zip(edges,flows)]),
                 branch_S=max([0.]+[max(abs(f['Sf']),abs(f['St']))/e['Smax_MVA']-1 for e,f in zip(edges,flows)]),
                 SUB_S=max(0.,(abs(sub)-sub_s)/max(1.,sub_s)),balance_MVA=residual,cross_pu=cross)
        if extended:
            devices=physical.case_devices(st,candidate,k,s,ix,copy.deepcopy(topology[2]),charge,discharge,V)
            raw.update(physical.technical_violations(st,candidate,k,s,devices,V,list(zip(edges,flows)),bounds[4]))
        normalized={name:max(0.,val-(pol.balance_tolerance_MVA if name=='balance_MVA' else pol.cross_tolerance_pu if name=='cross_pu' else pol.constraint_tolerance)) /
                    (max(1.,sub_hi) if name=='P' else max(1.,sub_qhi-sub_qlo) if name=='Q' else .05 if name=='voltage' else 1.) for name,val in raw.items()}
        loss=float(sum((f['Sf']+f['St']).real for f in flows))
        if extended:loss+=float(np.sum(shunts.real*abs(V)**2))
        for_power=float(sum(abs((f['Sf'].real-f['St'].real)/2)*e['FOR'][int(f['Sf'].real-f['St'].real<0)] for e,f in zip(edges,flows)))
        row.update(converged=True,passed=all(v==0. for v in normalized.values()),raw_violations=raw,normalized_violations=normalized,
                   sub_P_MW=float(sub.real),sub_Q_MVAr=float(sub.imag),loss_MW=loss,FOR_proxy_MW=for_power,
                   bus_order=ordered,voltage_real=V.real.tolist(),voltage_imag=V.imag.tolist(),
                   branches=[dict(id=e['id'],option=e['option'],P_from_MW=float(f['Sf'].real),Q_from_MVAr=float(f['Sf'].imag),P_to_MW=float(f['St'].real),Q_to_MVAr=float(f['St'].imag),
                                  I_from_A=float(abs(f['If'])*st.electrical['Ibase_A']),I_to_A=float(abs(f['It'])*st.electrical['Ibase_A'])) for e,f in zip(edges,flows)])
        row['hourly_costs']=economics.hourly(st,candidate,k,s,float(sub.real),loss,for_power,charge,discharge)
        if extended:
            row['physical']={key:value for key,value in devices.items() if key not in ('shunts','injection')}
            row['shunt_losses_MW']=float(np.sum(shunts.real*abs(V)**2))
            row['hourly_costs']['producer_Q_auxiliary_reimbursement']=-devices['auxiliary_reimbursement_per_hour']
        if not all(np.isfinite(v) for v in [*raw.values(),*normalized.values(),loss,for_power,*row['hourly_costs'].values()]):
            return dict(stage=k,year=st.years[k],state_id=s,converged=False,passed=False,failure='nonfinite_physics_or_cost')
        return row

    def evaluate(self,candidate):
        started=time.perf_counter();st=self.study
        st.check_unchanged()
        if source_hash()!=self.code or fingerprint(dict(version=VERSION,data=st.data,electrical=st.electrical))!=st.input_id:
            raise ValueError('Code or input changed during evaluator lifetime')
        candidate=copy.deepcopy(candidate)
        from .financial_policy import resolve
        result=dict(status='invalid_structure',signature=self.signature,policy=asdict(self.policy),financial_policy=resolve(st.c),candidate_id=None,metrics=None,cases=[],normalized_violation=None)
        try:
            st.shape_check(candidate);result['candidate_id']=candidate.candidate_id;result['plan_id']=candidate.plan_id
            installed,_=line_schedule(st.data,candidate.plan)
            capacity=candidate.new_capacity;caps=np.array([g['max_MW'] for g in st.new])
            if np.any(capacity < 0) or np.any(capacity.sum(axis=0)>caps+1e-8) or (st.data['investment_mode']=='single' and np.any(capacity[1:]!=0)):
                raise ValueError('Invalid purchase capacities')
            temporal=violations(st,candidate);rel=reliability(st.data,candidate.plan)
            if 'physical' in st.electrical:
                from .physical import structural_violations
                temporal.update(structural_violations(st,candidate))
        except (ValueError,KeyError,TypeError) as exc:
            result.update(error=str(exc),wall_seconds=time.perf_counter()-started);return result
        norm={name:max(0.,v-self.policy.temporal_tolerance) for name,v in temporal.items()}
        for metric in ('SAIFI','SAIDI'):
            norm[metric]=max(max(0.,r[metric]-st.c[metric+'_limits'][-1 if st.data['investment_mode']=='single' else k]-1e-8)/max(1.,st.c[metric+'_limits'][-1 if st.data['investment_mode']=='single' else k]) for k,r in enumerate(rel))
        charge,discharge=storage_dispatch(st,candidate);bounds=st.existing_bounds()
        if 'physical' in st.electrical:
            from .physical import q_bounds,existing_bounds
            bounds=(*existing_bounds(st,candidate),q_bounds(st,candidate))
        cases=[]
        for k in range(st.K):
            topology=self._edges(candidate,k,installed)
            for s in range(st.S):cases.append(self._case(candidate,k,s,topology,charge,discharge,bounds))
        for case in cases:
            for name,value in case.get('normalized_violations',{}).items():norm[name]=max(norm.get(name,0.),value)
        converged=all(row['converged'] for row in cases)
        metrics=None
        if converged:
            # Invalid delivered energy must not be invoiced by the strict PPA
            # ledger; retain measured violations rather than fabricate payments.
            try:
                metrics=economics.portfolio(st,candidate,cases)
                fingerprint(metrics)  # Reject NaN/Inf anywhere in the published ledger.
            except ValueError as exc:result['accounting_error']=str(exc)
            if 'accounting_error' in result:metrics=None
        status='inconclusive' if not converged else 'violated' if any(v>0 for v in norm.values()) else 'feasible' if metrics is not None else 'inconclusive'
        result.update(status=status,metrics=metrics,cases=cases,temporal_violations=temporal,reliability=rel,
                      normalized_violations=norm,normalized_violation=float(sum(norm.values())),wall_seconds=time.perf_counter()-started,
                      approved_cases=sum(r['passed'] for r in cases),expected_cases=st.K*st.S,
                      failed_cases=sum(not r['converged'] for r in cases),
                      physics='balanced_AC_with_explicit_Q_shunts_and_taps' if 'physical' in st.electrical else 'balanced_AC_fixed_PQ_with_one_slack',dependencies=self.versions,
                      interpretation='Feasible means supplied controls satisfy this catalogue; no optimality certificate or economic repair')
        return result


def save_evaluation(directory,study,candidate,result,compact=False,shared_study=None):
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Use a new result directory')
    if result['candidate_id']!=candidate.candidate_id:raise ValueError('Candidate/result identity mismatch')
    if compact:
        from .persistence import save_compact_evaluation
        return save_compact_evaluation(directory,study,candidate,result,shared_study)
    write_new(directory/'candidate.json',candidate.to_dict())
    write_new(directory/'study.json',dict(data=study.data,electrical=study.electrical))
    write_new(directory/'result.json',result)
    write_new(directory/'manifest.json',{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('candidate.json','study.json','result.json')})


def _compare_reopened(old,new,path='result'):
    """Compare the complete deterministic result, including detailed ledgers.

    Absolute tolerance is 1e-5 for VP currency fields, 1e-8 for hourly costs,
    1e-10 otherwise. No relative tolerance scales with billion-unit costs.
    Timing is excluded by the caller; it is not an electrical/financial output.
    """
    if isinstance(old,dict) and isinstance(new,dict):
        if set(old)!=set(new):raise ValueError('Reopened fields mismatch: '+path)
        for name in old:_compare_reopened(old[name],new[name],path+'.'+name)
    elif isinstance(old,list) and isinstance(new,list):
        if len(old)!=len(new):raise ValueError('Reopened length mismatch: '+path)
        for i,(a,b) in enumerate(zip(old,new)):_compare_reopened(a,b,f'{path}[{i}]')
    elif isinstance(old,(int,float)) and not isinstance(old,bool) and isinstance(new,(int,float)) and not isinstance(new,bool):
        tolerance=1e-5 if '_VP' in path else 1e-8 if '.hourly_costs.' in path else 1e-10
        if not np.isfinite(old) or not np.isfinite(new) or abs(old-new)>tolerance:
            raise ValueError('Reopened numeric mismatch: '+path)
    elif type(old)!=type(new) or old!=new:
        raise ValueError('Reopened value mismatch: '+path)


def audit_saved(directory):
    from .schema import Study
    directory=Path(directory)
    from .persistence import load_evaluation
    snapshot,vector,old=load_evaluation(directory)
    st=Study(**snapshot);candidate=Candidate.from_dict(vector)
    new=Evaluator(st,Policy(**old['policy'])).evaluate(candidate)
    if old['signature']!=new['signature'] or old['candidate_id']!=new['candidate_id'] or old['status']!=new['status']:
        raise ValueError('Reopened execution differs in policy, identity or status')
    if (old['metrics'] is None)!=(new['metrics'] is None):raise ValueError('Reopened accounting availability mismatch')
    if old['normalized_violation']!=new['normalized_violation'] or old['approved_cases']!=new['approved_cases']:
        raise ValueError('Reopened violation mismatch')
    if old['metrics'] is not None:
        for key in ('objective_VP','expected_VP','CVaR_operation_VP','cut_equivalent_MWh'):
            if abs(old['metrics'][key]-new['metrics'][key])>1e-5:raise ValueError('Reopened metric mismatch: '+key)
    _compare_reopened(plain({k:v for k,v in old.items() if k!='wall_seconds'}),
                      plain({k:v for k,v in new.items() if k!='wall_seconds'}))
    return new
