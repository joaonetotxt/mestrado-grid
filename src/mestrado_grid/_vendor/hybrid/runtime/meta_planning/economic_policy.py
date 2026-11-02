"""Ownership, economic incidence and reporting units; never reprice an input.

Castro (2024), Tables 4/5/13 and Eqs. 12, 20/21, 31, 35.
The Table 4 dispatchable curtailment quote is retained as evidence, but no
penalty is charged on unused dispatchable capacity under Eq. 31.
"""
import math

BENCHMARK = 'CASTRO_ADAPTADO_VP_T4_v1'
INCIDENCE_VERSION = 'CASTRO_ECONOMIC_INCIDENCE_v1'
REFERENCE = {
    'SUB': dict(owner='operator', energy=55., carbon=.6079, curtailment_quote=300., curtailment_applies=False),
    'BIO': dict(owner='operator', energy=45., carbon=.755, curtailment_quote=300., curtailment_applies=False),
    'ESS': dict(owner='operator', energy=40., carbon=.2012, curtailment_quote=300., curtailment_applies=False),
    'PV': dict(owner='third_party', energy=45., carbon=.0584, curtailment_quote=150., curtailment_applies=True),
    'WT': dict(owner='third_party', energy=45., carbon=.0276, curtailment_quote=150., curtailment_applies=True),
}
CASTRO_ESS_BETA0 = dict(type='ESS1', count=3, buses=(7,104,144), purchase_each=200000.,
                       maintenance_each_year=7000., maintenance_includes_replacement_at_year=15,
                       reported_maintenance=601249., classification='compatible_with_three_ESS1')

def capital_recovery_factor(rate, years):
    """End-of-year uniform annuity factor; stable at zero and small rates."""
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<0:
        raise ValueError('Discount rate must be a finite nonnegative fraction, not a percentage string')
    if type(years) is not int or years<1:
        raise ValueError('Horizon must be a positive integer number of years')
    return 1./years if rate==0 else rate/-math.expm1(-years*math.log1p(rate))

def equivalent_annual_cost(fixed, expected_operation, cvar_operation, beta, rate, years):
    """Read-only presentation of the complete VP objective, including risk."""
    numbers=(fixed,expected_operation,cvar_operation,beta)
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in numbers) or not 0<=beta<=1:
        raise ValueError('Invalid finite economic metrics or beta')
    crf=capital_recovery_factor(rate,years)
    risk_op=(1-beta)*expected_operation+beta*cvar_operation
    return dict(CRF=crf, PC1_VP=fixed, operation_expected_VP=expected_operation,
                operation_risk_weighted_VP=risk_op, objective_VP=fixed+risk_op,
                EAC_PC1=crf*fixed, EAC_operation_expected=crf*expected_operation,
                EAC_operation_CVaR=crf*cvar_operation, EAC_operation_risk_weighted=crf*risk_op,
                EAC_expected=crf*(fixed+expected_operation),EAC_CVaR_total=crf*(fixed+cvar_operation),
                EAC_objective=crf*(fixed+risk_op))

def normalize_owner(owner):
    names={'operator':'operator','DSO':'operator','dso':'operator',
           'producer':'third_party','third_party':'third_party'}
    if owner not in names:raise ValueError('Unknown economic resource owner')
    return names[owner]

def resource_incidence(resource, *, existing, config):
    """Return explicit asset ownership separately from any Q-interface owner.

    Existing PV/WT belong to producers even when an added Q interface is paid
    by the operator. New PV/WT without a PPA retain the historical owned-DG
    policy. No field in resource/config is changed and no prices are replaced.
    """
    tech=resource['technology']
    if tech not in REFERENCE:raise ValueError('Unsupported technology for Castro economic incidence')
    ref=REFERENCE[tech]
    if existing:owner=ref['owner']
    else:owner='third_party' if 'contract' in resource else 'operator'
    if 'owner' in resource and normalize_owner(resource['owner'])!=owner:
        raise ValueError('Resource owner contradicts economic incidence/contract')
    applicable=tech in ('PV','WT')
    if existing and bool(resource['renewable'])!=applicable:
        raise ValueError('Renewable flag contradicts Eq.31 curtailment scope')
    if tech=='ESS':
        energy=config['prices_per_MWh']['ESS'];raw_curt=resource.get('curtail_price',0.)
    elif existing:energy=resource['price'];raw_curt=resource['curtail_price']
    else:energy=resource['variable_per_MWh'];raw_curt=resource['curtail_per_MWh']
    return dict(version=INCIDENCE_VERSION,technology=tech,asset_owner=owner,existing=existing,
                capex_payer=owner,maintenance_payer=owner,
                incremental_generation_capex_in_DSO_FO=not existing and owner=='operator',
                variable_operation_per_MWh=energy,
                variable_incidence='third_party_payment' if owner=='third_party' else 'operator_use_cost',
                carbon_payer='operator',carbon_price=config['carbon_price_per_ton'],
                reference_emissions=ref['carbon'],reference_curtailment_quote=ref['curtailment_quote'],
                curtailment_applicable=applicable,configured_curtailment_price=raw_curt,
                effective_curtailment_price=raw_curt if applicable else 0.,
                direct_charge_purchase_tariff=0. if tech=='ESS' else None,
                charging_supply='priced_at_generating_sources' if tech=='ESS' else None,
                Q_interface_owner='separate_from_generation_asset; retain physical ledger')

def validate_study_incidence(data):
    """Reject ownership/scope contradictions without changing numerical inputs."""
    for g in data['generators']:
        resource_incidence(g,existing=True,config=data['config'])
    for g in data.get('extension',{}).get('generators',[]):
        row=resource_incidence(g,existing=False,config=data['config'])
        if row['asset_owner']=='third_party':
            if g['technology'] not in ('PV','WT') or g['capex_per_MW']!=0 or g['fixed_per_MW_year']!=0:
                raise ValueError('Third-party capital/O&M must not be duplicated in operator coefficients')
    for u in data.get('storage',[]):
        if 'owner' in u and normalize_owner(u['owner'])!='operator':
            raise ValueError('Private BESS requires a separately declared ownership/contract policy')
