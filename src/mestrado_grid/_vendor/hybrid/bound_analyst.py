"""Physical plans are anchored to the exact own-arm candidate analyzed by the model."""
from analytic_ai import Analyst
from analytic_core import copy, append, safe_save, atomic_bytes, gzip, json, plain
from complete_modes import CompleteKernel


class BoundAnalyst(Analyst):
    def choose(self, observation, *args, **kwargs):
        kernel = CompleteKernel.active_instance
        c, r = copy.deepcopy(kernel.current), copy.deepcopy(kernel.best)
        assert observation["current"]["candidate_id"] == c.candidate_id
        prefix = self.path / f"{self.calls:06d}"
        safe_save(prefix.with_suffix(".source_candidate.json"), c.to_dict())
        atomic_bytes(prefix.with_suffix(".source_validation.json.gz"),
                     gzip.compress(json.dumps(plain(r), allow_nan=False).encode()))
        observation = copy.deepcopy(observation)
        observation["actuator_semantics"] = {
            "source": "incumbent means this exact supplied candidate; it is retained independently while other experiments run",
            "topology_stage": "Exchange at stage k propagates to all later stages wherever the closing edge is absent and the dropping edge lies on the cycle. Do not repeat an identical exchange in later stages already changed.",
            "investments": "Purchases start at the chosen stage and persist according to the unmodified study. Stage indices are 0,1,2.",
            "experiments": "Each experiment retains its own control and asset candidates across interleaved visits. Global incumbent acceptance remains unchanged.",
        }
        decision = super().choose(observation, *args, **kwargs)
        source = kernel.pin_source(c, r)
        answer = copy.deepcopy(decision)
        for p in answer["proposals"]:
            if p["source"] == "incumbent":
                p["source"] = source
            elif p["source"] in kernel.bank.entries:
                cc, rr, origin = kernel.bank.entries[p["source"]]
                kernel.pin_source(cc, rr)
        append(self.path / "source_bindings.jsonl", dict(request=self.calls - 1,
               analyzed_candidate=source, plan_id=c.plan_id, source="own arm, pre-inference snapshot",
               bound_proposals=answer["proposals"]))
        return answer
