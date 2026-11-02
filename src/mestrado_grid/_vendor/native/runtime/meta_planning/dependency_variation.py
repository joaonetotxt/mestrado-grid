"""Declared dependency neighbourhoods; no prices, AC calls or dispatch repair.

Groups collect all stages/states at a bus, or investment/taps/tree keys of an
edge. These are overlapping physical *dependencies*, represented as a disjoint
coordinate partition; AC/network coupling between groups is not eliminated.
"""
import copy
import numpy as np
from .schema import fingerprint
from .categorical import reset_one
from .exploration import activate_one


def partition(codec):
    st=codec.study;p=codec.plan_codec
    names=['bus:'+str(b) for b in st.buses]+['edge:'+str(e['id']) for e in p.lines]
    buses={b:i for i,b in enumerate(st.buses)}
    edges={e['id']:len(buses)+i for i,e in enumerate(p.lines)}
    labels=np.full(codec.dimension,-1,dtype=int)
    def assign(name,full):
        b=codec.blocks[name];labels[b.start:b.stop]=np.asarray(full).ravel()[codec.indices[name]]
    a=np.empty(p.dimension,int);a[:len(p.lines)]=[edges[e['id']] for e in p.lines]
    a[len(p.lines):p.tree_offset]=[buses[b] for b in p.sites]
    a[p.tree_offset:]=np.tile([edges[e['id']] for e in p.lines],st.K);assign('plan',a)
    for name in ('existing_p','existing_q','new_capacity','new_p','phys_new_q'):
        if name not in codec.blocks:continue
        gens=st.old if name.startswith('existing') else st.new
        assign(name,np.broadcast_to([buses[g['bus']] for g in gens],codec.blocks[name].shape))
    for name in ('conventional','ev','cut'):
        assign(name,np.broadcast_to(np.array([buses[b] for b in st.buses])[None,:,None],codec.blocks[name].shape))
    sites=codec.storage_sites if hasattr(codec,'storage_sites') else [u['bus'] for u in st.units]
    assign('soc',np.broadcast_to(np.array([buses[b] for b in sites])[None,:,None],codec.blocks['soc'].shape))
    if 'physical' in st.electrical:
        ph=st.electrical['physical']
        for name,loc in [('phys_bess_q',[buses[b] for b in sites]),
                         ('phys_cap_state',[buses[b] for b in ph['capacitor_sites']]),
                         ('phys_tap',[edges[e] for e in ph['regulator_sites']]),
                         ('phys_cap_choice',[buses[b] for b in ph['capacitor_sites']]),
                         ('phys_reg_choice',[edges[e] for e in ph['regulator_sites']]),
                         ('phys_retro_choice',[buses[g['bus']] for g in st.old]),
                         ('phys_bess_choice',[buses[u['bus']] for u in st.units])]:
            assign(name,np.broadcast_to(loc,codec.blocks[name].shape))
    if np.any(labels<0):raise ValueError('Unassigned dependency coordinate')
    # Remove empty buses/edges, retaining deterministic order.
    used=np.unique(labels);mapping={int(k):i for i,k in enumerate(used)}
    return dict(names=[names[i] for i in used],labels=[mapping[int(i)] for i in labels])


def validate_partition(spec,dimension):
    if not isinstance(spec,dict) or set(spec)!={'names','labels'}:raise ValueError('Invalid dependency partition')
    names=spec['names'];labels=spec['labels']
    if not isinstance(names,list) or not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):raise ValueError('Invalid dependency names')
    if not isinstance(labels,list) or len(labels)!=dimension or any(type(i)!=int or not 0<=i<len(names) for i in labels) or set(labels)!=set(range(len(names))):raise ValueError('Invalid dependency labels')
    return copy.deepcopy(spec)


def propose(native,reference,categorical,exploration,spec,config,rng):
    labels=np.asarray(spec['labels']);delta=native-reference
    support=np.flatnonzero(abs(delta)>1e-12)
    group=int(rng.integers(len(spec['names'])))
    if config.update_scheme=='dependency':
        # A nonzero native direction must not vanish solely due to an empty mask.
        active_groups=np.unique(labels[support])
        if len(active_groups) and group not in active_groups:group=int(rng.choice(active_groups))
        selected=np.flatnonzero(labels==group)
    else:
        count=max(1,int(np.ceil(config.subset_fraction*len(reference))))
        selected=rng.choice(len(reference),count,replace=False)
        if len(support) and not np.intersect1d(selected,support).size:selected[0]=rng.choice(support)
        group=None
    step=float(rng.choice(config.step_levels)) if config.step_levels else config.block_step
    # One scalar preserves direction ratios; categorical coordinates retain
    # their native target key instead of clipping away category transitions.
    categories={r['start']+j for r in categorical for j in range(len(r['sizes']))}
    cont=np.array([int(i) for i in selected if int(i) not in categories],int)
    cat=np.array([int(i) for i in selected if int(i) in categories],int)
    out=reference.copy();magnitude=float(np.max(abs(delta[cont]))) if len(cont) else 0.
    scale=min(1.,step/magnitude) if magnitude else 1.
    out[cont]=reference[cont]+scale*delta[cont];out[cat]=native[cat]
    before_common=out.copy();category=continuous=activation=None
    # Release a constant direction without overwriting native variation.
    ids=np.concatenate([np.arange(r['start'],r['start']+len(r['sizes'])) for r in exploration['continuous']])
    idle=ids[abs(out[ids]-reference[ids])<=1e-12]
    if len(idle):
        i=int(rng.choice(idle));row=next(r for r in exploration['continuous'] if r['start']<=i<r['start']+len(r['sizes']))
        out[i]=np.clip(reference[i]+rng.uniform(-step,step),0,1)
        continuous=dict(family=row['name'],index=i,size=0,before=float(before_common[i]),target=float(out[i]))
    if rng.random()<config.categorical_rate:
        free=abs(out-reference)<=1e-12
        category=reset_one(out,categorical,free,rng)
        activation=activate_one(out,category,exploration['links'],free,rng)
    event=dict(reference_genome_id=fingerprint(reference),native_genome_id=fingerprint(native),
        proposal_genome_id=fingerprint(out),group=group,selected=selected.tolist(),step=step,scale=scale,
        source=config.direction_source,available_direction_keys=int(len(support)),
        retained_direction_keys=int(np.count_nonzero(abs(before_common-reference)>1e-12)),
        common_changed_keys=int(np.count_nonzero(abs(out-before_common)>1e-12)))
    return out,category,continuous,activation,event


def validate_event(event,spec,config,dimension):
    fields={'reference_genome_id','native_genome_id','proposal_genome_id','group','selected','step','scale','source','available_direction_keys','retained_direction_keys','common_changed_keys'}
    if not isinstance(event,dict) or set(event)!=fields:raise ValueError('Invalid dependency event')
    for k in ('reference_genome_id','native_genome_id','proposal_genome_id'):
        x=event[k]
        if not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x):raise ValueError('Invalid dependency hash')
    selected=event['selected']
    if not isinstance(selected,list) or not selected or any(type(i)!=int or not 0<=i<dimension for i in selected) or len(set(selected))!=len(selected):raise ValueError('Invalid dependency subset')
    if config.update_scheme=='dependency':
        g=event['group']
        if type(g)!=int or not 0<=g<len(spec['names']) or selected!=np.flatnonzero(np.asarray(spec['labels'])==g).tolist():raise ValueError('Invalid dependency mask')
    elif event['group'] is not None or len(selected)!=max(1,int(np.ceil(config.subset_fraction*dimension))):raise ValueError('Invalid support mask')
    if isinstance(event['step'],bool) or event['step'] not in (config.step_levels or [config.block_step]) or isinstance(event['scale'],bool) or not isinstance(event['scale'],(int,float)) or not 0<event['scale']<=1 or event['source']!=config.direction_source:raise ValueError('Invalid dependency scale/source')
    for k,maximum in [('available_direction_keys',dimension),('retained_direction_keys',len(selected)),('common_changed_keys',3)]:
        if type(event[k])!=int or not 0<=event[k]<=maximum:raise ValueError('Invalid dependency count')
    if event['available_direction_keys'] and not event['retained_direction_keys']:raise ValueError('Native direction lost by mask')


def restart_due(engine,history):
    c=engine.config
    if not c.restart_patience or engine.generation==0 or engine.cursor or engine.done:return None
    if engine.generation-engine.last_restart_generation<c.restart_patience:return None
    completed=[h for h in history if h['generation']>0]
    recent=completed[-c.restart_patience*engine.n:]
    stagnant=len(recent)==c.restart_patience*engine.n and not any(h['incumbent_improved'] for h in recent)
    diversity=float(np.mean(np.std(engine.population,axis=0)))
    if not stagnant and diversity>c.diversity_floor:return None
    return dict(reason='stagnation' if stagnant else 'diversity',diversity=diversity)


def restart_seed(config,number,index):
    return int(np.random.SeedSequence([config.seed,84219,number,index]).generate_state(1)[0])


def validate_restart(event,config,generation,index):
    if not isinstance(event,dict) or set(event)!={'number','generation','index','seed','reason','diversity','proposed_genome_id'} or not config.restart_patience:raise ValueError('Invalid restart event')
    if type(event['number'])!=int or not 1<=event['number']<=generation or type(event['generation'])!=int or event['generation']!=generation or type(event['index'])!=int or event['index']!=index:raise ValueError('Invalid restart identity')
    if type(event['seed'])!=int or event['seed']!=restart_seed(config,event['number'],index):raise ValueError('Invalid restart seed')
    if event['reason'] not in ('stagnation','diversity') or isinstance(event['diversity'],bool) or not isinstance(event['diversity'],(float,int)) or not np.isfinite(event['diversity']) or not 0<=event['diversity']<=.5:raise ValueError('Invalid restart trigger')
    h=event['proposed_genome_id']
    if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid restart hash')


def prepare_restart(engine,history,codec):
    trigger=restart_due(engine,history)
    if trigger is None:return
    from .initialization import initialize
    from .engines import key
    order=sorted(range(engine.n),key=lambda i:key(engine.records[i]))
    protected={order[0]}
    if engine.config.method=='GA' or engine.config.direction_source=='common':protected.add(0)
    count=min(engine.n-len(protected),max(1,int(np.ceil(engine.config.restart_fraction*engine.n))))
    indices=[int(i) for i in reversed(order) if i not in protected][:count]
    positions=[];events=[]
    for i in indices:
        number=engine.restart_count+1;seed=restart_seed(engine.config,number,i)
        mode=engine.config.initialization
        if mode=='mixed':mode='random'
        x=initialize(codec,seed,mode)[1];positions.append(x)
        events.append(dict(number=number,generation=engine.generation,index=i,seed=seed,
                           **trigger,proposed_genome_id=fingerprint(x)))
    engine.install_restart(indices,positions,events)


def parent_trace(history,cache,config):
    """Reconstruct survivor identities without running a proposal or AC again."""
    from .engines import key
    n=config.population;parents=[None]*n;trace=[];batch=[]
    for j,h in enumerate(history):
        trace.append(parents[j%n]);batch.append(h)
        if j%n!=n-1:continue
        children=[x['candidate_id'] for x in batch]
        if j<n or config.direction_source=='common' or config.method not in ('ACO','HyDE'):parents=children.copy()
        elif config.method=='HyDE':
            parents=[new if key(cache[new])<key(cache[old]) else old for old,new in zip(parents,children)]
        else:parents=sorted(parents+children,key=lambda cid:key(cache[cid]))[:n]
        for i,x in enumerate(batch):
            if x.get('restart_event') is not None:parents[i]=x['candidate_id']
        batch=[]
    return trace
