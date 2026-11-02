"""Physics-informed native coordinates and separately budgeted topology branches.

This is a declared guided adaptation of the native engines. The guide measures
responses and selects/parameterizes contexts; it never optimizes setpoints.
"""
import time
from collections import Counter
import numpy as np
from .guided_blocks import GuidedScheduler,WindowBlock
from .native_blocks import Field,endpoint_value,endpoint_key
from .physical_guidance import response,interaction_score


class LinkedBlock(WindowBlock):
    """One native scalar controls a signed, locally balanced P/Q/service ray.

    Remaining coordinates are independent. Both directions and no-op are
    reachable. Other visits use independent coordinates for the same controls.
    """
    def __init__(self,reference,ids,direction,fields,metadata,mass,radius):
        self.link_ids=np.asarray(ids,int);self.direction=np.asarray(direction,float)
        assert len(ids)>1 and np.isfinite(self.direction).all() and np.any(self.direction)
        super().__init__(reference,[Field('measured_joint_ray',self.link_ids)]+fields,
                         dict(metadata,linked_coordinates=self.link_ids.tolist(),linked_direction=self.direction.tolist()),mass,radius)

    def limit(self,sign):
        d=sign*self.direction;a=self.reference[self.link_ids]
        room=np.where(d>0,1-a,a)
        return float(np.min(np.minimum(room,self.radius)/np.maximum(abs(d),1e-300)))

    def decode(self,z):
        out=super().decode(z);step=float(2*endpoint_value(np.asarray(z)[0],self.mass)-1)
        out[self.link_ids]=self.reference[self.link_ids]+step*self.limit(1 if step>=0 else -1)*self.direction
        return np.clip(out,0,1)

    def project(self,full):
        out=super().project(full);delta=np.asarray(full)[self.link_ids]-self.reference[self.link_ids]
        t=float(delta@self.direction/(self.direction@self.direction));lim=self.limit(1 if t>=0 else -1)
        step=0. if lim==0 else np.clip(t/lim,-1,1)
        out[0]=endpoint_key((1+step)/2,self.mass);return out


class PhysicalScheduler(GuidedScheduler):
    def __init__(self,codec,seed,mass,options):
        super().__init__(codec,seed,mass,options)
        self.reference=None;self.record=None;self.models={};self.measurements=[]
        self.pending=[];self.queued_plans=set();self.polished_plans=set();self.deferred=[]
        self.remaining_budget=None;self.promotions=[]

    def update(self,record):
        super().update(record);self.record=record

    def set_reference(self,reference,remaining):
        self.reference=np.asarray(reference);self.remaining_budget=remaining

    def model(self,reference,case):
        if not self.options.measured_guidance:return None
        c=self.codec.decode(reference)[0];key=(c.candidate_id,*case)
        if key in self.models:return self.models[key]
        row=self.feedback.get(case,{})
        if self.record is None or self.record.get('candidate_id')!=c.candidate_id:return None
        if 'voltage_real' not in row:return None
        started=time.monotonic();m=response(self.codec.study,c,*case,np.asarray(row['voltage_real'])+1j*np.asarray(row['voltage_imag']))
        self.measurements.append(dict(candidate_id=c.candidate_id,case=case,seconds=time.monotonic()-started,
            injection_columns=2*(self.codec.study.B-1),jacobian_residual=m['jacobian_residual'],extra_AC_solves=0))
        self.models={key:m} # Rebuild after a changed operating point; no stale sensitivity.
        return m

    def service(self,reference,case,reason,radius_override=None):
        # All service contexts include P AND Q from their first visit.
        count=self.case_visits[case];self.case_visits[case]+=1
        radius=self.options.radii[count%len(self.options.radii)] if radius_override is None else radius_override
        st=self.codec.study;fields=[];selected={};mapping={}
        k,s=case;cutids=self.base.ids('cut',[case]);width=min((1,2,4)[count%3],len(cutids))
        if not width:return super().service(reference,case,reason)
        cuts=self.subset((case,'measured_load'),cutids,width)
        c=None;m=None
        if not self.coverage_preflight:
            c=self.codec.decode(reference)[0];m=self.model(reference,case)
        cutmap=self.base.maps['cut'][k,:,s]
        # Only adaptive slots rank current opportunities. Protected uniform
        # slots keep their seed decks, and remainder fields retain all loads.
        if c is not None and reason!='uniform_coverage':
            loads=[j for j,gene in enumerate(cutmap) if gene in cutids]
            merit={j:float(c.operation.cut[k,j,s]) for j in loads}
            order=sorted(loads,key=lambda j:merit[j],reverse=True)
            cuts=np.asarray([cutmap[j] for j in order[:width]],int)
        load_index=int(np.flatnonzero(cutmap==cuts[0])[0]);load_bus=st.buses[load_index]
        qr=float(np.asarray(st.electrical['q_ratio'])[load_index,s])
        fields.append(Field('cut:individual',cuts,'direct'));selected['cut']=cuts
        remainder=np.setdiff1d(cutids,cuts)
        if len(remainder):fields.append(Field('cut:group',remainder))
        headroom={};slopes={}
        if c is not None:
            from .physical import existing_bounds,availability
            _,upper,_,_=existing_bounds(st,c)
            upper_new=st.active_capacity(c)[:,None,:]*availability(st)[None,:,:]
            for family,cap,actual in [('existing_p',upper,c.operation.existing_p),('new_p',upper_new,c.operation.new_p)]:
                for j,gene in enumerate(self.base.maps[family][k,s]):
                    if gene>=0:headroom[int(gene)]=max(0.,float(cap[k,s,j]-actual[k,s,j]))
        for f in self.base.joint_fields([case]):
            family=f.name.split(':')[0]
            if family=='cut' or not len(f.indices):continue
            ids=f.indices
            # Seed decks ensure alternatives remain visited; every second visit
            # can use response similarity to select a companion among all sites.
            chosen=self.subset((case,f.name),ids,min(2,len(ids)))
            if m is not None and count%2==0 and family in ('existing_p','existing_q','new_p','phys_new_q','phys_bess_q'):
                catalogue=st.old if family.startswith('existing') else [{'bus':bus} for bus in self.codec.storage_sites] if family=='phys_bess_q' else st.new
                amap=self.base.maps[family][k,s]
                scores=[]
                for j,g in enumerate(catalogue):
                    gene=int(amap[j])
                    if gene in ids:
                        score=interaction_score(m,load_bus,g['bus'],qr,reactive=family.endswith('q'))
                        # An inactive plant cannot balance restored load. It
                        # remains selectable through uniform/remainder fields
                        # and investments, but is not a measured P companion.
                        if family in ('existing_p','new_p') and headroom.get(gene,0.)<=1e-9:continue
                        scores.append((score,gene))
                        mapping[gene]=(family,j,g['bus'])
                if scores:
                    best=max(scores)[1];chosen=np.unique(np.r_[chosen[:1],best])
            fields.append(Field(f.name+':individual',chosen,'direct'));selected[f.name]=chosen
            remainder=np.setdiff1d(ids,chosen)
            if len(remainder):fields.append(Field(f.name+':group',remainder))
        for family in ('phys_tap','phys_cap_state'):
            ids=self.base.ids(family,[case])
            if len(ids):fields.append(Field(family,self.subset((case,family),ids,min(2,len(ids))),'direct'))
        meta=dict(scope='service_joint',scheduler='physical',allocation=reason,case_groups=[[case]],
            service_face='pq_cut',case_visit=count,node_width=width,load_bus=load_bus,
            measurement_anchor=None if m is None else m['candidate_id'],
            pairing='measured response plus protected seed-deck companions',
            economic_signal=self.feedback.get(case,{}).get('economic_opportunity',0.),
            selected_load_cut_MW=None if c is None else float(c.operation.cut[k,load_index,s]),
            grouped_remainder=True,
            selected_P_headroom_MW={str(gene):headroom.get(int(gene),0.) for f in fields if f.kind=='direct' for gene in f.indices if int(gene) in headroom})
        # Alternate linked and independent coordinates. The ray is an operator,
        # not a calculated dispatch solution; its sign/size come from Engine.
        if m is not None and count%2==0:
            candidates=[]
            for gene,(family,j,bus) in mapping.items():
                if family not in ('existing_p','new_p'):continue
                if not any(gene in f.indices for f in fields):continue
                trial=reference.copy();h=min(1e-5,1-reference[gene])
                if h<=1e-10:continue
                trial[gene]+=h;t=self.codec.decode(trial)[0]
                slope=(getattr(t.operation,family)[k,s,j]-getattr(c.operation,family)[k,s,j])/h
                if slope>1e-10:candidates.append((interaction_score(m,load_bus,bus,qr),gene,slope))
            cutgene=int(cuts[0]);trial=reference.copy();h=min(1e-5,reference[cutgene])
            if candidates and h>1e-10:
                trial[cutgene]-=h;t=self.codec.decode(trial)[0]
                cut_slope=(c.operation.cut[k,load_index,s]-t.operation.cut[k,load_index,s])/h
                _,pgene,p_slope=max(candidates)
                if cut_slope>1e-10:
                    ids=[cutgene,pgene];direction=[-1.,cut_slope/p_slope]
                    rest=[Field(f.name,np.array([i for i in f.indices if i not in ids]),f.kind,f.endpoint) for f in fields]
                    return LinkedBlock(reference,ids,direction,rest,meta,self.base.mass,radius)
        return WindowBlock(reference,fields,meta,self.base.mass,radius)

    def observe_candidate(self,context,full,record,origin):
        if context.block.metadata['scope']!='topology' or record['status']=='invalid_structure':return
        pid=record.get('plan_id')
        if not pid or pid in self.queued_plans:return
        anchor=self.codec.decode(context.block.reference)[0].plan_id
        if pid==anchor:return
        self.queued_plans.add(pid)
        self.pending.append(dict(reference=np.asarray(full).copy(),plan_id=pid,parent_origin=origin,
            parent_candidate_id=record['candidate_id'],stage=context.block.stage))

    def choose(self,reference,contexts):
        if self.pending and not self.coverage_preflight:
            item=self.pending.pop(0);fields=[];k=item['stage']
            # Keep every case of the modified stage available to dispatch;
            # investments/tree are fixed only inside this temporary branch.
            for s in range(self.codec.study.S):
                for f in self.base.joint_fields([(k,s)]):
                    fields.append(Field(f.name+':state'+str(s),f.indices,f.kind,f.endpoint))
            self.polished_plans.add(item['plan_id'])
            meta={key:value for key,value in item.items() if key!='reference'}
            return None,WindowBlock(item['reference'],fields,dict(scope='topology_dispatch',**meta),self.base.mass,.02)
        old,block=super().choose(reference,contexts)
        if self.coverage_preflight:return old,block
        if old is not None and old.block.metadata['scope'] in ('topology','topology_dispatch'):
            case=divmod(self.base.case_order[self.uniform%len(self.base.case_order)],self.codec.study.S);self.uniform+=1
            return None,self.service(reference,case,'uniform_coverage')
        if old is not None and old.block.metadata['scope']=='service_joint':
            d=self.statistics.get(old.number,{})
            case=tuple(old.block.metadata['case_groups'][0][0])
            if d.get('rejections',0)/max(1,d.get('native_calls',0))>.4:
                radius=max(.001,old.block.radius/2)
                self.promotions.append(dict(parent_context=old.number,reason='measured_rejection',radius=radius,case=case))
                return None,self.service(reference,case,'rejection_adaptation',radius)
            if d.get('gain',0)>0 and getattr(old,'visits',0)>=2 and d.get('rejections',0)/max(1,d.get('native_calls',0))<.1 and old.block.radius<1:
                radius=min(1.,old.block.radius*2)
                self.promotions.append(dict(parent_context=old.number,reason='native_success',radius=radius,case=case))
                return None,self.service(reference,case,'success_adaptation',radius)
            self.case_visits[case]+=1
        if block is not None and block.metadata['scope']=='topology' and self.remaining_budget is not None:
            block.metadata['scheduler']='physical'
            # One outer native batch plus initialization can yield at most 2N
            # distinct plans. Reserve a complete native dispatch visit for each.
            n=getattr(self,'population',8)
            reserve=2*n*(1+n*(1+self.options.topology_dispatch_generations))
            if self.remaining_budget<reserve:
                self.deferred.append(dict(reason='insufficient_budget_for_all_topology_dispatch_branches',remaining=self.remaining_budget,reserve=reserve))
                case=divmod(self.base.case_order[self.uniform%len(self.base.case_order)],self.codec.study.S);self.uniform+=1
                return None,self.service(reference,case,'uniform_coverage')
        return old,block

    def report(self):
        out=super().report();out.update(measurements=self.measurements,promotions=self.promotions,
            topology_queued=sorted(self.queued_plans),topology_polished=sorted(self.polished_plans),
            topology_pending=len(self.pending),deferred_topology=self.deferred,
            note='Measured grouping and linked native coordinates are declared physical guidance, not a separate optimizer. All final acceptance uses unchanged rank.')
        return out
