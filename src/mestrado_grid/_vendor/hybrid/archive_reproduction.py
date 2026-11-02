"""Roundoff check for already-infeasible exploration archives only.

This never changes an evaluator record, ranking, physical limit or target test.
The incumbent and feasible records retain the original reproduction checks.
"""
import math

ABS_TOLERANCE = 1e-11
REL_TOLERANCE = 1e-8


def archive_roundoff_only(a, b):
    if a.get("status") != "violated" or b.get("status") != "violated":
        return False
    for name in ("signature","candidate_id","approved_cases","expected_cases","failed_cases"):
        if a.get(name) != b.get(name):
            return False
    av=a.get("normalized_violations",{});bv=b.get("normalized_violations",{})
    if not av or set(av)!=set(bv):
        return False
    if not any(v>0 for v in av.values()) or not any(v>0 for v in bv.values()):
        return False
    for key,x in av.items():
        y=bv[key]
        if not (math.isfinite(x) and math.isfinite(y)) or x<0 or y<0:
            return False
        if (x==0)!=(y==0):
            return False
        delta=abs(x-y)
        if delta>ABS_TOLERANCE or (x and delta>REL_TOLERANCE*min(x,y)):
            return False
    af=a.get("metrics",{}).get("selection_objective")
    bf=b.get("metrics",{}).get("selection_objective")
    return (isinstance(af,(int,float)) and isinstance(bf,(int,float)) and
            math.isfinite(af) and math.isfinite(bf) and abs(af-bf)<=1e-5)
