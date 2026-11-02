"""Structured physical proposals, not arbitrary code or changes to the model."""
import math
from hybrid_core import FAMILIES

PROMPT = """You are the analyst of ONE isolated electricity distribution search arm.
You may propose concrete investments, topology exchanges, capacity reductions, device
controls, source alternatives and P/Q/SOC refinements, including coupled actions.
Use the supplied full catalogue and own-arm measurements, not other experiment arms.
All catalogue locations remain eligible; suggestions are NOT a mandatory shortlist.
Optimize the ORIGINAL feasibility-first ranking and exact risk objective. Reliability,
AC constraints, alpha, beta, tariffs, existing plants and the cyclic-day BESS model
are immutable. An external frozen evaluator alone accepts proposals.
You are allowed to reject an unhelpful strategy, revisit an alternative and explain a
physical hypothesis. Prefer a testable plan of 3..8 proposals. Costs include PC1 plus
(1-beta)*expected annual operation + beta*CVaR operation; no VP30 substitution.
Use exact catalogue indices/IDs. New generation amount is ADDITIONAL MW. planned_reduce
fraction is the fraction RETAINED. Existing generation cannot be removed.
For each proposal source is 'incumbent' or a supplied archive candidate ID.
actions is a sequence of up to three resource actions, applied together. Empty actions
means dispatch only. cycles contains {stage,cycle} daily-cycle indices. mode 'cycle'
optimizes simultaneous P/Q/DR/EV/SOC; mode 'cases' optimizes each state P/Q separately.
Both have measured AC backtracking, exact global cost acceptance and full daily checks.
Unused action fields must be null. BESS uses unit,stage,q; new_* uses j,amount,stage;
capacitor uses j,option,stage; regulator uses j,option,stage,tap; retrofit uses j,stage;
cable uses edge,option,stage; topology uses close,drop,stage; planned_reduce uses j,fraction;
storage_remove uses unit; device_controls uses j,control ('tap'/'cap_state'),fraction.
Alternatively device_controls uses j,control,stage,tap for an explicit integer setpoint
at every state of that stage. capacitor/regulator may replace an existing planned
device; option=-1 cancels that planned device (tap=0 for regulator cancellation).
cable option=-1 cancels only the purchase at that exact edge and stage.
rounds must be 1..12 and radius .03125..1. The executor interleaves experiments one daily
cycle at a time, retaining their own evolving candidates; it may exhaust the equal
physical-evaluation budget before completing your entire plan. Prioritize 3..8 useful
experiments rather than assuming unlimited evaluations.
Return a detailed reason for each experiment. Preserve the option with no new investment.
Do not claim feasibility from estimates. No tools, no code or economic/constraint changes."""

FIELDS = {
    "family": dict(type="string", enum=[f for f in FAMILIES if f != "dispatch"]),
    **{k: dict(type=["integer", "null"]) for k in ("stage", "j", "option", "tap")},
    **{k: dict(type=["string", "null"]) for k in ("edge", "close", "drop", "unit", "control")},
    **{k: dict(type=["number", "null"]) for k in ("amount", "fraction")},
    "q": dict(type=["boolean", "null"]),
}
ACTION = dict(type="object", properties=FIELDS, required=list(FIELDS), additionalProperties=False)
CYCLE = dict(type="object", properties={k: dict(type="integer") for k in ("stage", "cycle")},
             required=["stage", "cycle"], additionalProperties=False)
PROPOSAL = dict(type="object", properties=dict(
    source=dict(type="string"), actions=dict(type="array", items=ACTION),
    cycles=dict(type="array", items=CYCLE, minItems=1, maxItems=24),
    radius=dict(type="number", minimum=.03125, maximum=1),
    mode=dict(type="string", enum=["cycle", "cases"]), rounds=dict(type="integer", minimum=1, maximum=12),
    reason=dict(type="string")),
    required=["source", "actions", "cycles", "radius", "mode", "rounds", "reason"],
    additionalProperties=False)
SCHEMA = dict(type="object", properties=dict(reason=dict(type="string"),
              proposals=dict(type="array", items=PROPOSAL)), required=["reason", "proposals"],
              additionalProperties=False)


def validate(d):
    if not isinstance(d, dict) or set(d) != {"reason", "proposals"}:
        raise ValueError("Unexpected decision fields")
    if not isinstance(d["reason"], str) or not d["reason"].strip():
        raise ValueError("Missing hypothesis")
    if not isinstance(d["proposals"], list) or not 1 <= len(d["proposals"]) <= 8:
        raise ValueError("Expected one to eight experiments")
    for p in d["proposals"]:
        if set(p) != set(PROPOSAL["required"]):
            raise ValueError("Unexpected proposal fields")
        if p["mode"] not in ("cycle", "cases") or type(p["rounds"]) != int or not 1 <= p["rounds"] <= 12:
            raise ValueError("Invalid refinement")
        if type(p["radius"]) not in (float, int) or not math.isfinite(p["radius"]) or not .03125 <= p["radius"] <= 1:
            raise ValueError("Invalid radius")
        if not isinstance(p["source"], str) or not isinstance(p["reason"], str):
            raise ValueError("Invalid source/reason")
        if not 1 <= len(p["cycles"]) <= 24 or len(p["actions"]) > 3:
            raise ValueError("Invalid experiment size")
        for cy in p["cycles"]:
            if set(cy) != {"stage", "cycle"} or any(type(v) != int for v in cy.values()):
                raise ValueError("Invalid daily cycle")
        for a in p["actions"]:
            if set(a) != set(FIELDS) or a["family"] not in FIELDS["family"]["enum"]:
                raise ValueError("Invalid action")
            for k in ("stage", "j", "option", "tap"):
                if a[k] is not None and type(a[k]) != int:
                    raise ValueError("Non-integer action index")
            for k in ("edge", "close", "drop", "unit", "control"):
                if a[k] is not None and not isinstance(a[k], str):
                    raise ValueError("Invalid action ID")
            for k in ("amount", "fraction"):
                if a[k] is not None and (type(a[k]) not in (int, float) or not math.isfinite(a[k])):
                    raise ValueError("Non-finite action")
            if a["q"] is not None and type(a["q"]) != bool:
                raise ValueError("Invalid Q switch")
    return d
