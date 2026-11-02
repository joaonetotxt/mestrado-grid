"""IA may allocate search effort; immutable evaluator remains the only acceptance authority."""
from hybrid_core import FAMILIES
PROMPT='''Schedule an isolated grid-search experiment using ONLY this arm's supplied evidence.
Return positive integer family_weights (1..6) for all fourteen families: these allocate FULL
EVALUATION shares, not changes to the objective. repair_slots (1..3) controls how many of four
visits may start from a retained alternative with better reliability and electrical violations.
Choose a local LP radius from .125,.25,.5. Feasibility first. Original FO, beta, alpha, CVaR,
catalogue and all AC/reliability constraints are immutable. No candidate, code, tools, financial
values or cross-arm information. Your scheduling is experimental, not a feasibility certificate.'''
SCHEMA=dict(type='object',properties=dict(
    family_weights=dict(type='object',properties={f:dict(type='integer',minimum=1,maximum=6) for f in FAMILIES},required=FAMILIES,additionalProperties=False),
    repair_slots=dict(type='integer',minimum=1,maximum=3),
    radius=dict(type='number',enum=[.125,.25,.5]),reason=dict(type='string')),
    required=['family_weights','repair_slots','radius','reason'],additionalProperties=False)

def validate(d):
    if set(d)!={'family_weights','repair_slots','radius','reason'}:raise ValueError('Unknown allocation fields')
    if not isinstance(d['family_weights'],dict) or set(d['family_weights'])!=set(FAMILIES):raise ValueError('All families must remain eligible')
    if any(type(x)!=int or not 1<=x<=6 for x in d['family_weights'].values()):raise ValueError('Invalid effort share')
    if type(d['repair_slots'])!=int or not 1<=d['repair_slots']<=3:raise ValueError('Invalid archive allocation')
    if isinstance(d['radius'],bool) or d['radius'] not in (.125,.25,.5):raise ValueError('Invalid radius')
    if not isinstance(d['reason'],str) or not d['reason'].strip() or len(d['reason'])>2500:raise ValueError('Reason required')
    return d
