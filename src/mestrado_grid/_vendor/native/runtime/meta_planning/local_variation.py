"""Declared sparse contextual variation; all preferences are explicit sampling rules.

No evaluator, prices, loads, constraint magnitudes or target shedding are read.
The native method supplies direction; a uniform subset limits simultaneous changes.
"""
import numpy as np
from .categorical import reset_one
from .exploration import activate_one
from .schema import fingerprint


def propose(native,reference,categorical,exploration,step,fraction,category_rate,rng):
    ids=np.concatenate([np.arange(r['start'],r['start']+len(r['sizes'])) for r in exploration['continuous']])
    order=rng.permutation(ids);count=max(1,int(np.ceil(fraction*len(ids))));selected=order[:count]
    out=reference.copy();out[selected]=reference[selected]+np.clip(native[selected]-reference[selected],-step,step)
    # One symmetric perturbation also releases initially constant directions.
    i=int(order[0]);before=float(out[i]);out[i]=np.clip(reference[i]+step*rng.uniform(-1,1),0,1)
    row=next(r for r in exploration['continuous'] if r['start']<=i<r['start']+len(r['sizes']))
    continuous=dict(family=row['name'],index=i,size=0,before=before,target=float(out[i]))
    category=None;activation=None
    if rng.random()<category_rate:
        category=reset_one(out,categorical,np.ones(len(out),bool),rng)
        activation=activate_one(out,category,exploration['links'],np.ones(len(out),bool),rng)
        if activation is not None:
            j=activation['index'];out[j]=np.clip(reference[j]+step*(2*activation['target']-1),0,1)
            activation['target']=float(out[j])
    overwritten={continuous['index']}|{e['index'] for e in (category,activation) if e is not None}
    native_ids=[int(j) for j in selected if int(j) not in overwritten]
    native_count=int(np.count_nonzero(abs(out[native_ids]-reference[native_ids])>1e-12))
    changed=int(np.count_nonzero(abs(out-reference)>1e-12))
    event=dict(reference_genome_id=fingerprint(reference),native_genome_id=fingerprint(native),
        selected='all_continuous' if count==len(ids) else selected.tolist(),selected_count=count,step=step,fraction=fraction,category_rate=category_rate,
        changed_keys=changed,native_changed_keys=native_count,common_changed_keys=changed-native_count)
    return out,category,continuous,activation,event


def validate_event(event,spec,dimension,step,fraction,rate):
    if event is None:return
    if set(event)!={'reference_genome_id','native_genome_id','selected','selected_count','step','fraction','category_rate','changed_keys','native_changed_keys','common_changed_keys'}:
        raise ValueError('Invalid local variation event')
    for key in ('reference_genome_id','native_genome_id'):
        v=event[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('Invalid proposal hash')
    ids={r['start']+j for r in spec['continuous'] for j in range(len(r['sizes']))}
    selected=event['selected'];count=max(1,int(np.ceil(fraction*len(ids))))
    if type(event['selected_count'])!=int or event['selected_count']!=count:raise ValueError('Invalid sparse selection count')
    if count==len(ids):
        if selected!='all_continuous':raise ValueError('Invalid full coordinate subset')
    elif not isinstance(selected,list) or len(selected)!=count or any(type(i)!=int or i not in ids for i in selected) or len(set(selected))!=len(selected):
        raise ValueError('Invalid sparse coordinate subset')
    if any(isinstance(event[k],bool) or event[k]!=v for k,v in [('step',step),('fraction',fraction),('category_rate',rate)]):raise ValueError('Local parameters differ from configuration')
    if type(event['changed_keys'])!=int or not 0<=event['changed_keys']<=min(dimension,count+2):raise ValueError('Invalid local change count')
    if any(type(event[n])!=int or event[n]<0 for n in ('native_changed_keys','common_changed_keys')) or event['native_changed_keys']+event['common_changed_keys']!=event['changed_keys'] or event['native_changed_keys']>count-1:
        raise ValueError('Invalid native/common change accounting')
