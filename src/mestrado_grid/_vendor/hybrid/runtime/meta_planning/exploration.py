"""Full mixed-domain mutation and conditional control activation, price-blind.

Categorical variation alone does not release constant-zero HyDE dimensions or
PSO velocities erased while an optional controller is inactive. Every mixed
child also receives one uniformly selected continuous-family/key reset. A Q
retrofit transition receives one linked Q key from its proposed entry onwards.
These are search proposals; no solution is repaired economically or by AC.
"""
import copy
import numpy as np
from .categorical import validate_spec,validate_events


def validate_exploration(spec,categorical,dimension):
    if not isinstance(spec,dict) or set(spec)!={'continuous','links'}:raise ValueError('Invalid mixed exploration specification')
    continuous=validate_spec(spec['continuous'],dimension)
    cat={row['start']+j:n for row in categorical for j,n in enumerate(row['sizes'])}
    controlled={row['start']+j:row['name'] for row in continuous for j in range(len(row['sizes']))}
    if set(cat)&set(controlled) or any(n!=0 for r in continuous for n in r['sizes']):raise ValueError('Mixed domains overlap or are not continuous')
    seen=set()
    for link in spec['links']:
        if set(link)!={'choice_index','stages','family','by_stage'}:raise ValueError('Invalid activation link')
        i=link['choice_index'];stages=link['stages'];by_stage=link['by_stage']
        if type(i)!=int or i not in cat or i in seen or not isinstance(stages,list) or len(stages)!=cat[i] or not isinstance(by_stage,list) or not by_stage:
            raise ValueError('Invalid activation domain')
        seen.add(i)
        if stages[0] is not None or any(type(k)!=int or not 0<=k<len(by_stage) for k in stages[1:]):raise ValueError('Invalid activation entry stages')
        allids=[]
        for values in by_stage:
            if not isinstance(values,list):raise ValueError('Invalid activation controls')
            for j in values:
                if type(j)!=int or controlled.get(j)!=link['family']:raise ValueError('Activation control/domain mismatch')
                allids.append(j)
        if len(set(allids))!=len(allids):raise ValueError('Duplicate activation control')
    return copy.deepcopy(spec)


def activate_one(x,event,links,active,rng):
    if event is None or event['target']==0:return None
    link=next((r for r in links if r['choice_index']==event['index']),None)
    if link is None:return None
    entry=link['stages'][event['target']]
    ids=[i for group in link['by_stage'][entry:] for i in group if active[i]]
    if not ids:return None
    i=int(rng.choice(ids));before=float(x[i]);x[i]=float(rng.random())
    return dict(family=link['family'],index=i,size=0,before=before,target=float(x[i]))


def validate_mixed_events(continuous_events,activation_events,categorical_events,spec,population):
    validate_events(continuous_events,spec['continuous'],population)
    validate_events(activation_events,spec['continuous'],population)
    links={r['choice_index']:r for r in spec['links']}
    for cat,event in zip(categorical_events,activation_events):
        if event is None:continue
        if cat is None or not cat['target'] or cat['index'] not in links:raise ValueError('Activation has no installation transition')
        link=links[cat['index']];entry=link['stages'][cat['target']]
        if event['index'] not in [i for group in link['by_stage'][entry:] for i in group]:raise ValueError('Activation precedes entry or belongs to another asset')
