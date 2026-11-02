"""Objective-blind categorical variation shared by explicitly mixed variants.

One active class, one coordinate, one uniformly selected different category per
non-elite child. Absence and every purchase date/type are symmetric targets.
No electrical trial, cost, historical solution or forced investment is used.
"""
import copy
import numpy as np


def validate_spec(spec,dimension):
    seen=set();names=set()
    for row in spec:
        if set(row)!={'name','start','sizes'} or not isinstance(row['name'],str) or row['name'] in names:
            raise ValueError('Invalid categorical domain')
        start=row['start'];sizes=row['sizes'];names.add(row['name'])
        if type(start)!=int or start<0 or not isinstance(sizes,list) or not sizes or start+len(sizes)>dimension:
            raise ValueError('Invalid categorical range')
        if any(type(n)!=int or n<0 for n in sizes):raise ValueError('Invalid category count')
        indices=set(range(start,start+len(sizes)))
        if seen&indices:raise ValueError('Overlapping categorical ranges')
        seen|=indices
    return copy.deepcopy(spec)


def category(value,size):
    return min(int(value*size),size-1) if size else float(value)


def reset_one(x,spec,active,rng):
    eligible=[]
    for row in spec:
        sizes=np.asarray(row['sizes']);ids=np.flatnonzero(active[row['start']:row['start']+len(sizes)]&(sizes!=1))
        if len(ids):eligible.append((row,ids))
    if not eligible:return None
    row,ids=eligible[int(rng.integers(len(eligible)))];local=int(rng.choice(ids));index=row['start']+local
    size=row['sizes'][local];before=category(x[index],size)
    if size:
        target=int(rng.integers(size-1));target+=int(target>=before)
        x[index]=(target+.5)/size
    else:
        x[index]=float(rng.random());target=float(x[index])
    return dict(family=row['name'],index=index,size=size,before=before,target=target)


def validate_events(events,spec,population):
    if not isinstance(events,list) or len(events)!=population:raise ValueError('Invalid categorical events')
    byname={r['name']:r for r in spec}
    for event in events:
        if event is None:continue
        if set(event)!={'family','index','size','before','target'} or event['family'] not in byname:
            raise ValueError('Invalid categorical event')
        row=byname[event['family']];i=event['index'];size=event['size']
        if type(i)!=int or not row['start']<=i<row['start']+len(row['sizes']) or type(size)!=int or row['sizes'][i-row['start']]!=size:
            raise ValueError('Categorical event/domain mismatch')
        for key in ('before','target'):
            v=event[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not (0<=v<size if size else 0<=v<=1):
                raise ValueError('Invalid categorical target')
            if size and type(v)!=int:raise ValueError('Noninteger category')
        if size and event['before']==event['target']:raise ValueError('Categorical reset did not change category')


def transitions(spec,reference,proposed,evaluated):
    """Counts relative to the incumbent before this evaluation, not acceptance.

    cap_state counts normalized keys (bins depend on the selected equipment).
    Inactive controls and chronology repairs can remove a proposed transition.
    """
    if reference is None:return {}
    out={}
    for row in spec:
        start=row['start'];size=np.asarray(row['sizes']);stop=start+len(size)
        def values(x):
            a=np.asarray(x[start:stop]);return np.where(size>0,np.minimum(np.floor(a*size),size-1),a)
        old=values(reference);raw=values(proposed);actual=values(evaluated)
        out[row['name']]=dict(proposed=int(np.count_nonzero(raw!=old)),evaluated=int(np.count_nonzero(actual!=old)),
            added=int(np.count_nonzero((old==0)&(actual>0))) if row['name'].endswith('choice') else None,
            removed=int(np.count_nonzero((old>0)&(actual==0))) if row['name'].endswith('choice') else None)
    return out
