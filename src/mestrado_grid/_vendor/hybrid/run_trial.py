"""Equal-budget isolated trials. No production imports, target files or launchers."""
from analytic_core import *
from analytic_ai import Analyst
from analytic_policy import validate
import argparse


def proposal(st, source, cycles, actions=None, mode="cycle", reason="deterministic physical policy"):
    return dict(source=source, cycles=[dict(stage=k, cycle=st.cycles.index(list(cy))) for k, cy in cycles],
                actions=actions or [], radius=.5, mode=mode, rounds=1, reason=reason)


def deterministic(kernel, visit, seed, directed=True):
    st = kernel.st
    # The explicit baseline has access to exactly the same action families.
    alternative = kernel.bank.repair_source(kernel.best) if directed and visit % 2 else None
    c, r = alternative[:2] if alternative else (kernel.current, kernel.best)
    source = c.candidate_id if alternative else "incumbent"
    k, cy = kernel.coverage.select(st, r)
    rel, other, cost = coordinates(r)
    if directed and rel == 0:
        # Electrical recovery and economics: full daily cycle or state-wise sensitivities.
        return proposal(st, source, [(k, cy)], mode="cycle" if other > 0 or visit % 2 else "cases")
    family = "topology" if directed and rel > 0 else FAMILIES[1 + (visit // 4) % (len(FAMILIES) - 1)]
    try:
        menu = opportunities(st, c, r, seed, visit)
        selected = [a for a in menu if a["family"] == family]
        if selected:
            a = selected[0] if family == "topology" else selected[(visit // 3) % len(selected)]
            # No asset credit until an equally refined control has been evaluated.
            return proposal(st, source, [(k, cy)], [a])
    except ValueError:
        pass
    return proposal(st, source, [(k, cy)], mode="cases" if visit % 2 else "cycle")


def run(fixture, method, budget, seed, label):
    beta = "00" if fixture.endswith("b0") else "05"
    st = Study(**read(D / "inputs" / f"beta_{beta}.json"))
    out = D / "trials" / label / fixture / method
    kernel = Kernel(st, out, budget)
    sup = Analyst(out / "ai") if method == "AI" else None
    safe_save(out / "identity.json", dict(pid=os.getpid(), executable=sys.executable, argv=sys.argv,
              utc=now(), fixture=fixture, method=method, seed=seed, input_id=st.input_id,
              search_budget_full_calls=budget, audit_calls_additional=1, production=False))
    if fixture.startswith("mat"):
        codec = ArrayCodec(st)
        initial = [codec.decode(x)[0] for x in np.load(D / "inputs/sim1.npy", allow_pickle=False)]
    else:
        initial = [Candidate.from_dict(read(D / "fixtures/incumbent.json"))]
    safe_save(out / "initialization.json", dict(candidate_ids=[c.candidate_id for c in initial],
          study_sha256=sha(D / "inputs" / f"beta_{beta}.json"),
          population_sha256=sha(D / ("inputs/sim1.npy" if fixture.startswith("mat") else "fixtures/incumbent.json")),
          historical_finalists_imported=False))
    queue = []
    refresh = 0
    visit = 0
    try:
        for i, c in enumerate(initial):
            kernel.evaluate(c, dict(role="initialization", member=i))
        initial_record = compact(kernel.best)
        safe_save(out / "initial.json", initial_record)
        # Inference is refreshed from the actual new record, not a stale cached global policy.
        while kernel.ev.calls < budget:
            before = kernel.ev.calls
            visit += 1
            if sup and not queue:
                obs = observation(st, kernel.current, kernel.best, kernel.bank, kernel.feedback,
                                  kernel.coverage, budget - kernel.ev.calls)
                kernel.progress("waiting_for_AI")
                decision = sup.choose(obs, heartbeat=lambda: kernel.progress("waiting_for_AI"))
                queue = [(f"AI_{sup.calls}_{i}", p) for i, p in enumerate(decision["proposals"])]
            experiment_id = None
            if visit % 4 == 1:
                # Both arms have the same starvation guard; this is not credited to AI.
                k, cy = kernel.coverage.select(st, kernel.best)
                p = proposal(st, "incumbent", [(k, cy)], mode="cycle" if visit % 2 else "cases")
                owner = "shared_cycle_coverage"
            elif visit % 4 == 2:
                p = deterministic(kernel, visit, seed, directed=False)
                owner = "shared_resource_coverage"
            elif sup:
                experiment_id, p = queue.pop(0)
                owner = "AI_physical_proposal"
            else:
                p = deterministic(kernel, visit, seed)
                owner = "deterministic_physical_proposal"
            append(out / "proposals.jsonl", dict(call=kernel.ev.calls, source=owner, proposal=p))
            try:
                done = kernel.execute(p, owner, experiment_id)
                if experiment_id is not None and not done:
                    queue.append((experiment_id, p))
            except (ValueError, KeyError, IndexError) as exc:
                kernel.counts["rejected_proposals"] += 1
                append(out / "proposal_rejections.jsonl", dict(call=kernel.ev.calls, proposal=p, error=str(exc)))
            if kernel.ev.calls == before:
                # An invalid/no-op proposal never consumes phantom AC budget or loops forever.
                k, cy = kernel.coverage.select(st, kernel.best)
                kernel.refine(copy.deepcopy(kernel.current), kernel.best, k, cy, .25, "cases", "fallback")
                if kernel.ev.calls == before:
                    kernel.evaluate(kernel.current, dict(role="explicit_noop_retest"))
    except BudgetFinished:
        pass
    except BaseException:
        safe_save(out / "error.json", dict(utc=now(), traceback=traceback.format_exc(), calls=kernel.ev.calls))
        kernel.progress("failed")
        raise
    assert kernel.ev.calls == budget
    return kernel.finish(initial_record, dict(fixture=fixture, method=method, seed=seed, input_id=st.input_id,
           AI_requests=sup.calls if sup else 0, AI_wait_s=sup.wait_s if sup else 0,
           AI_usage=sup.usage if sup else [], strict_original_reliability=True,
           final_record_is_unmodified_official_evaluator=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fixture", choices=["mat_b0", "mat_b05", "incumbent_b0", "incumbent_b05"])
    p.add_argument("method", choices=["AI", "NO_AI"])
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--seed", type=int, default=96101)
    p.add_argument("--label", default="comparison")
    a = p.parse_args()
    print(json.dumps(run(a.fixture, a.method, a.budget, a.seed, a.label)), flush=True)
