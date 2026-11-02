"""Isolated bounded ask/tell engines. Never patches the frozen engine registry.

HyDE-DF follows the 2019 author's core source (references/HyDE_DF_author.m),
including exponential DF, per-coordinate epsilon, rotating permutations,
strict jDE success, and clipping (BRM=1); no competition restart extension.
Author Fernando Lezama/GECAD; Storn, Price, Neumaier, Van Zandt base DE.
Original permission notice is preserved with the reference source.
L-SHADE follows Tanabe/Fukunaga 2014 with the common constrained comparator.
GVNS uses a finite signed-coordinate VND on the shared decoded search space.
These are declared constraint/block adaptations, not paper benchmark replicas.
"""
import copy,math
import numpy as np
from meta_planning.evaluator import rank
from meta_planning.advanced_engines import success_weights

def midpoint(v,parent):return np.where(v<0,parent/2,np.where(v>1,(parent+1)/2,v))
def decay(progress):
 a=max(0.,1.-float(progress))
 return 0. if a<=0 else math.exp(1.-1./(a*a))

class Base:
 def __init__(self,initial,seed,budget):
  self.population=np.asarray(initial,float).copy()
  if self.population.ndim!=2 or len(self.population)<4 or self.population.shape[1]<1 or not np.isfinite(self.population).all() or np.any((self.population<0)|(self.population>1)):raise ValueError('Invalid initial population')
  if type(budget)!=int or budget<len(self.population):raise ValueError('Budget must include initial population')
  self.n0=len(self.population);self.d=self.population.shape[1];self.budget=budget
  self.rng=np.random.default_rng(np.random.SeedSequence([seed,41821]))
  self.evaluations=0;self.generation=0;self.cursor=0;self.pending=self.population.copy()
  self.received=[];self.records=None;self.best=None;self.awaiting=False;self.trace=[]
 @property
 def done(self):return self.evaluations>=self.budget
 def ask(self):
  if self.done:raise StopIteration('Evaluation budget exhausted')
  if self.awaiting:raise RuntimeError('tell required before next ask')
  self.awaiting=True;return self.pending[self.cursor].copy()
 def tell(self,actual,record):
  if not self.awaiting:raise RuntimeError('ask required before tell')
  x=np.asarray(actual,float)
  if x.shape!=(self.d,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Invalid evaluated keys')
  rank(record);self.awaiting=False;self.pending[self.cursor]=x;self.received.append(copy.deepcopy(record))
  if self.best is None or rank(record)<rank(self.best['record']):self.best=dict(x=x.copy(),record=copy.deepcopy(record))
  self.evaluations+=1;self.cursor+=1
  if self.cursor==len(self.pending):
   if self.generation==0:self.population=self.pending.copy();self.records=self.received
   else:self.accept()
   self.generation+=1;self.cursor=0;self.received=[]
   self.trace.append(dict(evaluations=self.evaluations,generation=self.generation,population=len(self.population)))
   if not self.done:self.pending=self.propose()

class LSHADE(Base):
 def __init__(self,initial,seed,budget,memory=6,p=.11,archive_rate=2.6):
  super().__init__(initial,seed,budget)
  if memory<1 or not 0<p<=1 or archive_rate<0:raise ValueError('Invalid L-SHADE parameters')
  self.mf=np.full(memory,.5);self.mcr=np.full(memory,.5);self.mi=0
  self.archive=[];self.p=p;self.archive_rate=archive_rate;self.updates=0;self.weight_modes=[]
 def propose(self):
  pop=self.population;n,d=pop.shape;rng=self.rng
  order=sorted(range(n),key=lambda j:rank(self.records[j]));top=min(n,max(2,int(np.floor(n*self.p+.5))))
  union=np.vstack([pop,np.asarray(self.archive).reshape(-1,d)])
  out=np.empty_like(pop);self.tf=np.empty(n);self.tc=np.empty(n)
  for i in range(n):
   m=int(rng.integers(len(self.mf)));cr=0. if self.mcr[m]<0 else float(np.clip(rng.normal(self.mcr[m],.1),0,1))
   f=self.mf[m]+.1*rng.standard_cauchy()
   while not np.isfinite(f) or f<=0:f=self.mf[m]+.1*rng.standard_cauchy()
   f=min(1.,float(f));pb=order[int(rng.integers(top))]
   r1=int(rng.choice([j for j in range(n) if j!=i]));r2=int(rng.choice([j for j in range(len(union)) if j not in (i,r1)]))
   v=midpoint(pop[i]+f*(pop[pb]-pop[i])+f*(pop[r1]-union[r2]),pop[i])
   mask=rng.random(d)<cr;mask[int(rng.integers(d))]=True
   out[i]=np.where(mask,v,pop[i]);self.tf[i]=f;self.tc[i]=cr
  return out
 def update_memory(self,success,pairs):
  if not success:return
  w,mode=success_weights(pairs);f=self.tf[success];cr=self.tc[success];k=self.mi
  self.mf[k]=float(np.sum(w*f*f)/np.sum(w*f))
  self.mcr[k]=-1. if self.mcr[k]<0 or max(cr)==0 else float(np.sum(w*cr*cr)/np.sum(w*cr))
  self.mi=(k+1)%len(self.mf);self.updates+=1;self.weight_modes.append(mode)
 def accept(self):
  success=[];pairs=[]
  for i,r in enumerate(self.received):
   old,new=rank(self.records[i]),rank(r)
   if new<old:self.archive.append(self.population[i].copy());success.append(i);pairs.append((old,new))
   if new<=old:self.population[i]=self.pending[i];self.records[i]=r
  self.update_memory(success,pairs)
  size=max(4,int(np.floor(self.n0+(4-self.n0)*self.evaluations/self.budget+.5)))
  size=min(len(self.population),size)
  ids=sorted(range(len(self.population)),key=lambda j:rank(self.records[j]))[:size]
  self.population=self.population[ids];self.records=[self.records[j] for j in ids]
  limit=int(np.floor(self.archive_rate*size+.5))
  if len(self.archive)>limit:self.archive=[self.archive[int(j)] for j in self.rng.choice(len(self.archive),limit,replace=False)]

class HyDEDF(Base):
 def __init__(self,initial,seed,budget):
  super().__init__(initial,seed,budget)
  if self.n0<5:raise ValueError('Author source requires at least five members')
  self.f=np.full((self.n0,3),.5);self.cr=np.full(self.n0,.5);self.decays=[];self.updates=0
 def propose(self):
  rng=self.rng;n,d=self.population.shape
  self.tf=np.where(rng.random((n,3))<.1,.1+.9*rng.random((n,3)),self.f)
  self.tc=np.where(rng.random(n)<.1,rng.random(n),self.cr)
  # Author source: a=(I_itermax-gen)/I_itermax; evaluations=n*gen here.
  delta=decay(self.evaluations/self.budget);self.decays.append(delta)
  shift=int(rng.permutation(4)[0])+1;a=rng.permutation(n);b=a[(np.arange(n)+shift)%n]
  eps=self.tf[:,1,None]+rng.normal(size=(n,d))
  v=self.population+self.tf[:,2,None]*(self.population[a]-self.population[b])+delta*self.tf[:,0,None]*(self.best['x']*eps-self.population)
  # The author's implementation does not force a crossover coordinate.
  mask=rng.random((n,d))<self.tc[:,None]
  return np.clip(np.where(mask,v,self.population),0,1)
 def accept(self):
  for i,r in enumerate(self.received):
   if rank(r)<rank(self.records[i]):
    self.population[i]=self.pending[i];self.records[i]=r;self.f[i]=self.tf[i];self.cr[i]=self.tc[i];self.updates+=1

class GVNSVND(Base):
 """GVNS shaking followed by first-improvement finite VND.

 VND neighborhoods contain +/- coordinate moves at radii .02,.10,.35,
 in seeded coordinate order. Shaking changes 1, 2, or up to 4 coordinates.
 Every neighbor and shake consumes one call. Descent resets on improvement;
 exhaustive failure of the three finite neighborhoods returns to shaking.
 No claim of local optimality over the original continuous decision space.
 """
 def __init__(self,initial,seed,budget):
  super().__init__(initial,seed,budget);self.phase='initial';self.k=0;self.l=0
  self.radii=(.02,.10,.35);self.center=None;self.local=None;self.deck=[]
  self.shakes=0;self.descent_steps=0;self.local_resets=0;self.neighborhood_changes=0
 def propose(self):
  if self.phase=='initial':self.center=copy.deepcopy(self.best);self.phase='shake'
  if self.phase=='shake':
   x=self.center['x'].copy();ids=self.rng.choice(self.d,min(self.d,2**self.k),replace=False)
   x[ids]=np.clip(x[ids]+self.rng.uniform(-1,1,len(ids))*self.radii[self.k],0,1)
   self.shakes+=1;return x[None,:]
  if not self.deck:
   order=self.rng.permutation(self.d);self.deck=[(int(j),s) for j in order for s in (-1.,1.)]
  j,s=self.deck.pop(0);x=self.local['x'].copy();x[j]=np.clip(x[j]+s*self.radii[self.l],0,1)
  self.descent_steps+=1;return x[None,:]
 def accept(self):
  evaluated=dict(x=self.pending[0].copy(),record=self.received[0])
  if self.phase=='shake':
   self.local=evaluated;self.phase='descent';self.l=0;self.deck=[];return
  if rank(evaluated['record'])<rank(self.local['record']):
   self.local=evaluated;self.l=0;self.deck=[];self.local_resets+=1
  elif not self.deck:
   self.l+=1;self.neighborhood_changes+=1
   if self.l==len(self.radii):
    if rank(self.local['record'])<rank(self.center['record']):self.center=copy.deepcopy(self.local);self.k=0
    else:self.k=(self.k+1)%len(self.radii)
    self.phase='shake'
  # Reporting population remains the evaluated common initial sample;
  # trajectory incumbent and archive best are explicitly separate.

NEW={'L-SHADE':LSHADE,'HyDE-DF':HyDEDF,'GVNS-VND':GVNSVND}
