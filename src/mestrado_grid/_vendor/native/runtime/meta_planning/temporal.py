"""Algebraic cyclic SOC and bounded energy redistribution. No economic search."""
import numpy as np


def demand_bounds(st):
    f=st.data['extension'].get('flex',{})
    dr=float(f.get('load_shift_fraction',0.))
    if not 0<=dr<=1:
        raise ValueError('Invalid DR fraction')
    lo=(1-dr)*st.base_load
    hi=st.base_load+dr*st.base_load.max(axis=2,keepdims=True)
    if f.get('ev_enabled',False):
        peak=float(f['ev_peak_multiplier'])
        if not np.isfinite(peak) or peak<1:
            raise ValueError('EV multiplier must accommodate the original profile')
        elo=np.zeros(st.base_ev.shape)
        ehi=np.broadcast_to(peak*st.base_ev.max(axis=2,keepdims=True),st.base_ev.shape).copy()
    else:
        elo=st.base_ev.copy();ehi=st.base_ev.copy()
    return lo,hi,elo,ehi


def balance_profile(proposed, hours, lower, upper, target):
    """Redistribute residual energy in proportion to remaining bound headroom.

    Feasible profiles are unchanged. This is a deterministic conservation repair,
    with no prices, ranking or optimization solver. Leading axes are batched.
    """
    h=np.asarray(hours,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    x=np.clip(np.asarray(proposed,float),lo,hi)
    target=np.asarray(target,float)
    if np.any(target<np.sum(lo*h,axis=-1)-1e-9) or np.any(target>np.sum(hi*h,axis=-1)+1e-9):
        raise ValueError('Demand bounds cannot accommodate required daily energy')
    residual=target-np.sum(x*h,axis=-1)
    residual=np.where(abs(residual)<=1e-12*np.maximum(1,abs(target)),0.,residual)
    room=np.where((residual>=0)[...,None],hi-x,x-lo)
    available=np.sum(room*h,axis=-1)
    fraction=np.divide(abs(residual),available,out=np.zeros(residual.shape),where=available>0)
    x+=np.sign(residual)[...,None]*np.minimum(1.,fraction)[...,None]*room
    if np.max(abs(np.sum(x*h,axis=-1)-target),initial=0)>1e-8:
        raise ValueError('Energy conservation repair did not close')
    return x


def storage_dispatch(st,candidate):
    soc=candidate.operation.soc
    delta=np.zeros(soc.shape)
    for cy in st.cycles:
        delta[:,:,cy]=soc[:,:,cy[1:]+cy[:1]]-soc[:,:,cy]
    ec=np.array([u['eta_charge'] for u in st.units])[None,:,None]
    ed=np.array([u['eta_discharge'] for u in st.units])[None,:,None]
    charge=np.maximum(delta,0)/(ec*st.hours[None,None,:])
    discharge=np.maximum(-delta,0)*ed/st.hours[None,None,:]
    return charge,discharge


def normalize(st,candidate):
    op=candidate.operation
    active=st.storage_active(candidate.plan)
    inactive_count=int(np.count_nonzero(op.soc*(~active)[:,:,None]))
    op.soc*=active[:,:,None]
    contracted=0
    for u,unit in enumerate(st.units):
        cap=unit['energy_MWh'];power=unit['power_MW']
        op.soc[:,u,:]=np.clip(op.soc[:,u,:],0.,cap)
        for cy in st.cycles:
            levels=op.soc[:,u,cy]
            delta=np.roll(levels,-1,axis=-1)-levels
            allowed=np.where(delta>=0,power*unit['eta_charge']*st.hours[cy],power*st.hours[cy]/unit['eta_discharge'])
            ratios=np.divide(allowed,abs(delta),out=np.full(delta.shape,np.inf),where=abs(delta)>0)
            scale=np.minimum(1.,ratios.min(axis=-1))
            scale=np.where(scale>1-1e-12,1.,scale)
            contracted+=int(np.count_nonzero(scale<1))
            mean=levels.mean(axis=-1,keepdims=True)
            op.soc[:,u,cy]=mean+scale[:,None]*(levels-mean)
    clo,chi,elo,ehi=demand_bounds(st)
    changes={}
    for name,base,lo,hi in [('conventional',st.base_load,clo,chi),('ev',st.base_ev,elo,ehi)]:
        array=getattr(op,name);before=array.copy()
        for cy in st.cycles:
            array[:,:,cy]=balance_profile(array[:,:,cy],st.hours[cy],lo[:,:,cy],hi[:,:,cy],np.sum(base[:,:,cy]*st.hours[cy],axis=-1))
        changes[name+'_repaired_entries']=int(np.count_nonzero(abs(array-before)>1e-10))
    return dict(inactive_SOC_entries_zeroed=inactive_count,SOC_cycles_contracted=contracted,**changes)


def violations(st,candidate):
    """Normalized maxima by family, measured on the submitted physical controls."""
    st.shape_check(candidate);op=candidate.operation
    def bound(x,lo,hi,scale):
        return float(max(0.,np.max((lo-x)/scale,initial=0.),np.max((x-hi)/scale,initial=0.)))
    clo,chi,elo,ehi=demand_bounds(st)
    scale=np.maximum(1.,st.base_load.max(axis=2,keepdims=True))
    v=dict(DR_bounds=bound(op.conventional,clo,chi,scale),EV_bounds=bound(op.ev,elo,ehi,np.maximum(1.,ehi)),
           conventional_cut=bound(op.cut,0.,op.conventional,scale),DR_energy=0.,EV_energy=0.)
    for name,base,actual in [('DR_energy',st.base_load,op.conventional),('EV_energy',st.base_ev,op.ev)]:
        for cy in st.cycles:
            target=np.sum(base[:,:,cy]*st.hours[cy],axis=-1)
            error=abs(np.sum(actual[:,:,cy]*st.hours[cy],axis=-1)-target)/np.maximum(1.,target)
            v[name]=max(v[name],float(np.max(error,initial=0.)))
    cap=np.array([u['energy_MWh'] for u in st.units])[None,:,None]
    power=np.array([u['power_MW'] for u in st.units])[None,:,None]
    active=st.storage_active(candidate.plan)[:,:,None]
    v['SOC_bounds']=bound(op.soc,0.,cap*active,np.maximum(cap,1e-12))
    charge,discharge=storage_dispatch(st,candidate)
    v['BESS_power']=max(bound(charge,0.,power*active,np.maximum(power,1e-12)),bound(discharge,0.,power*active,np.maximum(power,1e-12)))
    # Closure is structural: the last boundary is the first boundary of that
    # same day. Derived dispatch must still conserve internal energy with losses.
    v['SOC_balance']=0.
    for cy in st.cycles:
        ec=np.array([u['eta_charge'] for u in st.units])[None,:]
        ed=np.array([u['eta_discharge'] for u in st.units])[None,:]
        z=np.sum(charge[:,:,cy]*st.hours[cy],axis=-1)*ec-np.sum(discharge[:,:,cy]*st.hours[cy],axis=-1)/ed
        v['SOC_balance']=max(v['SOC_balance'],float(np.max(abs(z)/np.maximum(cap[:,:,0],1e-12),initial=0.)))
    return v
