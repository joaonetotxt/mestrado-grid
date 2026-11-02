"""Crash-safe, append-only storage for suspended own-arm experiments."""
import hashlib, json, gzip, os
from pathlib import Path
from atomic_io import atomic_bytes, shared_bytes, read_json, write_json


class BranchVault:
    def __init__(self, path, state=None):
        resolved = str(Path(path).resolve())
        # Content hashes plus deep arm names exceed the legacy Windows 260-char
        # limit, especially the atomic writer's temporary suffix.
        self.path = Path('\\\\?\\'+resolved) if os.name=='nt' and not resolved.startswith('\\\\?\\') else Path(resolved)
        (self.path/'objects').mkdir(parents=True, exist_ok=True)
        (self.path/'nodes').mkdir(exist_ok=True)
        self.head, self.cursor, self.count = None, None, 0
        if state:
            self.head, self.cursor, self.count = state['head'], state['cursor'], state['count']
            if self.head: self.node(self.head)

    def export(self):
        return dict(head=self.head, cursor=self.cursor, count=self.count)

    def node(self, digest):
        raw = shared_bytes(self.path/'nodes'/(digest+'.json'))
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Suspended experiment index hash mismatch')
        return json.loads(raw)

    def put(self, work, metadata):
        raw = json.dumps(work, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        digest = hashlib.sha256(raw).hexdigest()
        target = self.path/'objects'/(digest+'.json.gz')
        if not target.exists(): atomic_bytes(target, gzip.compress(raw, compresslevel=2))
        if hashlib.sha256(gzip.decompress(shared_bytes(target))).hexdigest() != digest:
            raise ValueError('Suspended experiment payload hash mismatch')
        entry = dict(previous=self.head, sequence=self.count, payload=digest, **metadata)
        blob = json.dumps(entry, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        head = hashlib.sha256(blob).hexdigest()
        atomic_bytes(self.path/'nodes'/(head+'.json'), blob)
        self.head = head
        self.count += 1

    def select(self, recent=4, rotating=4):
        """Bounded metadata working set; older branches remain reachable across reviews."""
        rows = {}; pointer = self.head
        for _ in range(recent):
            if not pointer: break
            row = self.node(pointer); rows[pointer] = row; pointer = row['previous']
        pointer = self.cursor or self.head
        for _ in range(rotating):
            if not pointer: break
            row = self.node(pointer); rows[pointer] = row; pointer = row['previous']
        self.cursor = pointer or self.head
        return list(rows.values())

    def load(self, entry):
        raw = gzip.decompress(shared_bytes(self.path/'objects'/(entry['payload']+'.json.gz')))
        if hashlib.sha256(raw).hexdigest() != entry['payload']:
            raise ValueError('Suspended experiment payload hash mismatch')
        return json.loads(raw)
