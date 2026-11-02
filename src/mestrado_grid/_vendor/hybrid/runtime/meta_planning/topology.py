# Pure topology routines isolated from legacy solver modules.
import copy
import itertools
from dataclasses import asdict
import numpy as np
from .schema import Plan as Candidate, fingerprint

def orient_tree(data, edges):
    lines = {e['id']: e for e in data['lines']}
    if len(edges) != len(data['buses'])-1 or len(set(edges)) != len(edges):
        raise ValueError('A tree must have exactly n-1 distinct edges.')
    adjacency = {b: [] for b in data['buses']}
    for e in edges:
        if e not in lines:
            raise ValueError('Unknown line: '+str(e))
        line = lines[e]
        adjacency[line['i']].append((line['j'], e, 0))
        adjacency[line['j']].append((line['i'], e, 1))
    seen = {data['root']}; queue = [data['root']]; orientation = {}
    for b in queue:
        for child,e,direction in adjacency[b]:
            if child not in seen:
                seen.add(child); queue.append(child); orientation[e] = direction
    if len(seen) != len(data['buses']):
        raise ValueError('Disconnected or cyclic candidate.')
    return orientation

def line_schedule(data, candidate):
    lines = {e['id']: e for e in data['lines']}
    count = len(data['config']['investment_years'])
    if len(candidate.trees) != count:
        raise ValueError('Wrong number of stage trees.')
    buys = [tuple(x) for x in candidate.line_buys]
    if len(set(buys)) != len(buys):
        raise ValueError('Duplicate line purchase.')
    for e,o,k in buys:
        if any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_)) for x in (o,k)) or e not in lines or o not in range(len(lines[e]['options'])) or k not in range(count):
            raise ValueError('Unknown line option or purchase stage.')
        if data['investment_mode'] == 'single' and k != 0:
            raise ValueError('Late purchase in single-stage investment mode.')
        if lines[e]['existing'] and o == 0:
            raise ValueError('Original sunk cable option has no replacement purchase price.')
    installed = {}; retired = {}
    for e, line in lines.items():
        previous = 0 if line['existing'] else None
        for k in range(count):
            now = [o for ee,o,kk in buys if ee == e and kk == k]
            if len(now) > 1:
                raise ValueError('Two purchases on the same edge and stage.')
            retired[e,k] = 0
            if now:
                o = now[0]
                # IDs identify catalogue entries; they have no physical order.
                # A different installed option is a paid replacement, even if
                # it reduces rating. AC feasibility remains the evaluator's job.
                if o == previous:
                    raise ValueError('Purchase repeats the currently installed option.')
                if previous is not None:
                    retired[e,k] = 1
                previous = o
            installed[e,k] = previous
    for k,tree in enumerate(candidate.trees):
        orient_tree(data, tree)
        if any(installed[e,k] is None for e in tree):
            raise ValueError('Tree uses a line before installation.')
    storage = {u['id']: u for u in data['storage']}; sites = set()
    for u,k in candidate.storage_buys:
        if not isinstance(k,(int,np.integer)) or isinstance(k,(bool,np.bool_)) or u not in storage or k not in range(count):
            raise ValueError('Unknown storage option or purchase stage.')
        if storage[u]['bus'] in sites:
            raise ValueError('More than one storage purchase at a bus.')
        if data['investment_mode'] == 'single' and k != 0:
            raise ValueError('Late storage in single-stage investment mode.')
        sites.add(storage[u]['bus'])
    return installed, retired

def repair_trees(data, candidate, preferred):
    """Priority spanning-tree repair; buy missing selected circuits on first use.

    Existing purchases remain sunk decisions. Repair guarantees topology and
    installation chronology only; electrical feasibility is measured separately by the AC evaluator.
    """
    result = copy.deepcopy(candidate)
    lines = {e['id']: e for e in data['lines']}
    trees = []
    for k,priority in enumerate(preferred):
        available = {e for e,line in lines.items() if line['existing']}
        available.update(e for e,o,kk in result.line_buys if kk <= k)
        order = list(dict.fromkeys(list(priority)+sorted(available)+sorted(lines)))
        parent = {b:b for b in data['buses']}
        def root(b):
            while parent[b] != b:
                parent[b] = parent[parent[b]]; b = parent[b]
            return b
        selected = []
        for e in order:
            if e not in lines:
                continue
            line = lines[e]; a,b = root(line['i']),root(line['j'])
            if a == b:
                continue
            parent[a] = b; selected.append(e)
            if e not in available:
                future = [(o,kk) for ee,o,kk in result.line_buys if ee == e and kk > k]
                if future:
                    o,kk = min(future, key=lambda item:item[1])
                    result.line_buys.remove(next(x for x in result.line_buys if tuple(x) == (e,o,kk)))
                else:
                    o = 0
                purchase_stage = 0 if data['investment_mode'] == 'single' else k
                result.line_buys.append((e,o,purchase_stage)); available.add(e)
            if len(selected) == len(data['buses'])-1:
                break
        orient_tree(data, selected); trees.append(selected)
    result.trees = trees
    line_schedule(data, result)
    return result

def canonical(plan):
    return Candidate([sorted(t) for t in plan.trees],sorted(map(tuple,plan.line_buys)),sorted(map(tuple,plan.storage_buys)))

def plan_id(plan):return fingerprint(asdict(canonical(plan)))

def reliability(data,plan):
    installed,_=line_schedule(data,plan);lines={e['id']:e for e in data['lines']};out=[]
    total=sum(data['consumers'].values())
    for k,tree in enumerate(plan.trees):
        direction=orient_tree(data,tree);paths={data['root']:(0.,0.)};pending=set(tree)
        while pending:
            progress=False
            for e in sorted(pending):
                line=lines[e];di=direction[e];i,j=(line['i'],line['j']) if di==0 else (line['j'],line['i'])
                if i not in paths:continue
                opt=line['options'][installed[e,k]];f=opt['failure_rate'][di];r=opt['repair_hours'][di]
                paths[j]=(paths[i][0]+f,paths[i][1]+f*r);pending.remove(e);progress=True
            if not progress:raise ValueError('Disconnected reliability paths')
        fi=sum(data['consumers'][b]*paths[b][0] for b in paths)/total
        di=sum(data['consumers'][b]*paths[b][1] for b in paths)/total
        target=-1 if data['investment_mode']=='single' else k
        out.append(dict(stage=k,SAIFI=fi,SAIDI=di,paths=paths,passed=fi<=data['config']['SAIFI_limits'][target]+1e-8 and di<=data['config']['SAIDI_limits'][target]+1e-8))
    return out

class PlanCodec:
    """Every allowed line schedule, site choice and spanning tree is representable.

    Option IDs are nominal categories, never an upgrade ordering.
    Every distinct purchasable-option replacement is eligible, including return to
    an earlier purchased type; existing option 0 is sunk equipment, not a free
    purchasable replacement. Each purchase is charged in full. Identical purchases
    are excluded as redundant. AC limits decide electrical feasibility.
    Categories describe complete schedules; no independent rounding of arcs.
    Disconnected installed graphs are repaired by advancing/adding purchases.
    All repair-induced decisions are included in the canonical ID and accounting.
    """
    def __init__(self,data):
        self.data=copy.deepcopy(data);self.K=len(data['config']['investment_years'])
        self.lines=sorted(data['lines'],key=lambda e:e['id'])
        stages=range(1 if data['investment_mode']=='single' else self.K)
        self.line_choices=[]
        for e in self.lines:
            choices=[()]
            options=range(1 if e['existing'] else 0,len(e['options']))
            for count in range(1,len(stages)+1):
                for os in itertools.product(options,repeat=count):
                    previous=0 if e['existing'] else None
                    valid=True
                    for o in os:
                        if o==previous:
                            valid=False;break
                        previous=o
                    if not valid:continue
                    for ks in itertools.combinations(stages,count):
                        choices.append(tuple((e['id'],o,k) for o,k in zip(os,ks)))
            self.line_choices.append(choices)
        self.sites=sorted({u['bus'] for u in data['storage']})
        self.storage_choices=[ [()] + [((u['id'],k),) for u in sorted(data['storage'],key=lambda u:u['id']) if u['bus']==b for k in stages] for b in self.sites]
        self.categories=[len(x) for x in self.line_choices+self.storage_choices]+[len(self.lines)]*(self.K*len(self.lines))
        self.dimension=len(self.categories)
        self.tree_offset=len(self.lines)+len(self.sites)

    def decode(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(self.dimension,) or not np.all(np.isfinite(x)):raise ValueError('Invalid genome shape or nonfinite gene')
        if np.any(x<0) or np.any(x>1):raise ValueError('Genome outside [0,1]')
        choices=self.line_choices+self.storage_choices;items=[]
        for value,options in zip(x,choices):items.append(options[min(int(value*len(options)),len(options)-1)])
        buys=[p for group in items[:len(self.lines)] for p in group]
        stores=[p for group in items[len(self.lines):] for p in group]
        keys=x[self.tree_offset:].reshape(self.K,len(self.lines))
        # Prefer installed edges, then missing edges only if necessary to connect.
        priorities=[]
        for k in range(self.K):
            available={e['id'] for e in self.lines if e['existing']}|{e for e,o,j in buys if j<=k}
            order=sorted(range(len(self.lines)),key=lambda j:(self.lines[j]['id'] not in available,keys[k,j],self.lines[j]['id']))
            priorities.append([self.lines[j]['id'] for j in order])
        before=copy.deepcopy(buys)
        plan=canonical(repair_trees(self.data,Candidate([[] for _ in range(self.K)],buys,stores),priorities))
        return plan,dict(repaired_line_schedule=sorted(before)!=plan.line_buys)

    def encode(self,plan):
        plan=canonical(plan);line_schedule(self.data,plan);x=[]
        for e,options in zip(self.lines,self.line_choices):
            schedule=tuple(sorted((p for p in plan.line_buys if p[0]==e['id']),key=lambda p:p[2]));x.append((options.index(schedule)+.5)/len(options))
        byid={u['id']:u['bus'] for u in self.data['storage']}
        for b,options in zip(self.sites,self.storage_choices):
            schedule=tuple(p for p in plan.storage_buys if byid[p[0]]==b);x.append((options.index(schedule)+.5)/len(options))
        for tree in plan.trees:
            # Selected tree edges precede all others; any acyclic tree is recovered.
            order=sorted(range(len(self.lines)),key=lambda j:(self.lines[j]['id'] not in tree,self.lines[j]['id']))
            rank={j:i for i,j in enumerate(order)}
            x.extend((rank[j]+.5)/len(self.lines) for j in range(len(self.lines)))
        return np.array(x,dtype=float)
