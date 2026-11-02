"""Risk-aware ledger checks; equations remain those of the frozen evaluator."""
import math
def finance_checks(result):
    m=result.get('metrics')
    if not m:raise ValueError('Missing financial evidence')
    f=m['finance'];v=m['financial_views']['vp30']
    checks=dict(fixed=sum(f['fixed_parts_VP'].values())-f['PC1'],operating=sum(f['operating_parts_annual'].values())-f['expected_operation'],objective=f['objective']-(f['PC1']+(1-f['beta'])*f['expected_operation']+f['beta']*f['CVaR_operation']),vp30=v['accounting_residual'],selection=m['selection_objective']-f['objective'])
    if not all(math.isfinite(x) and abs(x)<1e-5 for x in checks.values()):raise ValueError('Risk ledger fails reconciliation')
    if f['alpha']!=.9 or not 0<=f['beta']<=1:raise ValueError('Unexpected risk class')
    if m['financial_policy']!='CASTRO_ALIGNED_ANNUAL_WEIGHTED_v1':raise ValueError('Unexpected objective basis')
    return dict(all_pass=True,residuals_exact=checks,FO=f['objective'],PC1=f['PC1'],expected_operation=f['expected_operation'],CVaR_operation=f['CVaR_operation'],alpha=f['alpha'],beta=f['beta'])
