"""Lossless in-memory array representation; persisted candidate IDs unchanged."""
import copy
import numpy as np
from .codec import Codec

def array_candidate(candidate):
    if candidate.physical is None:return candidate
    c=copy.copy(candidate);c.physical=candidate.physical.copy()
    for name in ('new_q','bess_q','cap_state','tap'):
        c.physical[name]=np.asarray(candidate.physical[name],dtype=float)
    return c

class ArrayCodec(Codec):
    def decode(self,genome):
        c,repairs=super().decode(genome)
        return array_candidate(c),repairs

    def encode(self,candidate):return super().encode(array_candidate(candidate))
