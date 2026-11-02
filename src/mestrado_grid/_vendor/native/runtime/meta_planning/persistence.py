"""Lossless, checked array bundles and shared-input evaluation storage.

No pickle. Numeric arrays keep their exact dtype/shape/bytes. Every member is
hashed, the metadata is hashed, and callers also bind files into run checkpoints.
"""
from pathlib import Path
import gzip,hashlib,io,json,os,zipfile
import numpy as np
from .schema import plain,fingerprint,read,write_new


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_bundle(path,value,replace=False):
    path=Path(path)
    if path.exists() and not replace:raise FileExistsError(path)
    members={}
    def encode(x):
        if isinstance(x,np.ndarray):
            if x.dtype.kind not in 'bifu':raise ValueError('Only real numeric arrays are supported')
            name=f'arrays/{len(members)}.npy';f=io.BytesIO();np.save(f,x,allow_pickle=False)
            members[name]=f.getvalue();return {'__numeric_array__':name}
        if isinstance(x,dict):return {str(k):encode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [encode(v) for v in x]
        return plain(x)
    payload=encode(value)
    content=dict(version=1,payload=payload,members={n:hashlib.sha256(v).hexdigest() for n,v in members.items()})
    metadata=json.dumps(dict(sha256=fingerprint(content),content=content),separators=(',',':'),allow_nan=False).encode()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending')
    with tmp.open('wb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            z.writestr('metadata.json',metadata)
            for n,v in members.items():z.writestr(n,v)
        stream.flush();os.fsync(stream.fileno())
    # Same-volume atomic commit, identical retry policy to JSON checkpoints.
    import time
    for attempt in range(6):
        try:os.replace(tmp,path);break
        except PermissionError as exc:
            if os.name!='nt' or exc.winerror not in (5,32) or attempt==5:raise
            time.sleep(.02*2**attempt)


def read_bundle(path):
    with zipfile.ZipFile(path) as z:
        envelope=json.loads(z.read('metadata.json'))
        if set(envelope)!={'sha256','content'} or fingerprint(envelope['content'])!=envelope['sha256']:raise ValueError('Array metadata changed')
        data=envelope['content']
        if set(data)!={'version','payload','members'} or data['version']!=1:raise ValueError('Invalid array bundle')
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!={'metadata.json',*data['members']}:raise ValueError('Unexpected array member')
        arrays={}
        for name,digest in data['members'].items():
            blob=z.read(name)
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Array bytes changed')
            a=np.load(io.BytesIO(blob),allow_pickle=False)
            if a.dtype.kind not in 'bifu':raise ValueError('Invalid array dtype')
            arrays[name]=a.tolist()
    used=set()
    def decode(x):
        if isinstance(x,dict):
            if set(x)=={'__numeric_array__'}:
                name=x['__numeric_array__'];used.add(name);return arrays[name]
            return {k:decode(v) for k,v in x.items()}
        if isinstance(x,list):return [decode(v) for v in x]
        return x
    result=decode(data['payload'])
    if used!=set(arrays):raise ValueError('Unreferenced array member')
    return result


def save_compact_evaluation(directory,study,candidate,result,shared_study):
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Use a new result directory')
    expected=directory.resolve().parent.parent/'study.json'
    if shared_study is None or Path(shared_study).resolve()!=expected:raise ValueError('Shared input must belong to this run')
    snapshot=read(expected)
    if set(snapshot)!={'sha256','payload'} or fingerprint(snapshot['payload'])!=snapshot['sha256']:raise ValueError('Shared input envelope changed')
    if fingerprint(snapshot['payload'])!=fingerprint(dict(data=study.data,electrical=study.electrical)):raise ValueError('Shared input mismatch')
    value=candidate.to_dict();value['new_capacity']=candidate.new_capacity;value['operation']=vars(candidate.operation)
    if candidate.physical is not None:
        value['physical']={k:v if k=='version' else np.asarray(v) for k,v in candidate.physical.items()}
    write_bundle(directory/'candidate.npz',value)
    payload=json.dumps(plain(result),separators=(',',':'),allow_nan=False).encode()
    with (directory/'result.json.gz').open('xb') as f:f.write(gzip.compress(payload,compresslevel=1,mtime=0))
    write_new(directory/'study_ref.json',dict(path='../../study.json',sha256=sha(expected)))
    write_new(directory/'manifest.json',{n:sha(directory/n) for n in ('candidate.npz','result.json.gz','study_ref.json')})


def load_evaluation(directory):
    directory=Path(directory);manifest=read(directory/'manifest.json')
    standard={'candidate.json','study.json','result.json'}
    compact={'candidate.npz','result.json.gz','study_ref.json'}
    if set(manifest) not in (standard,compact):raise ValueError('Incomplete evaluation manifest')
    for name,digest in manifest.items():
        if sha(directory/name)!=digest:raise ValueError('Evaluation artifact changed: '+name)
    if set(manifest)==standard:return read(directory/'study.json'),read(directory/'candidate.json'),read(directory/'result.json')
    ref=read(directory/'study_ref.json')
    if set(ref)!={'path','sha256'} or ref['path']!='../../study.json':raise ValueError('Invalid shared input reference')
    path=directory.resolve().parent.parent/'study.json'
    if sha(path)!=ref['sha256']:raise ValueError('Shared input changed')
    snapshot=read(path)
    if set(snapshot)!={'sha256','payload'} or fingerprint(snapshot['payload'])!=snapshot['sha256']:raise ValueError('Shared input envelope changed')
    return snapshot['payload'],read_bundle(directory/'candidate.npz'),json.loads(gzip.decompress((directory/'result.json.gz').read_bytes()))
