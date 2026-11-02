"""Honor state-wise P/Q requests for resource experiments as well as dispatch.

The original interleaved kernel already handles daily-cycle resource refinements.
This extension only affects the previously unexercised assets + mode='cases' branch.
"""
from analytic_core import *


class CompleteKernel(Kernel):
    active_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinned_sources = {}
        CompleteKernel.active_instance = self

    def pin_source(self, c, r):
        if c.candidate_id != r["candidate_id"] or r["signature"] != self.ev.signature:
            raise ValueError("Candidate/context mismatch in analyst snapshot")
        self.pinned_sources[c.candidate_id] = (copy.deepcopy(c), copy.deepcopy(r),
                                             dict(kind="own_analyst_snapshot"))
        return c.candidate_id

    def compose(self, candidate, record, actions):
        """Resolve declarative redundant exchanges without inventing a new action."""
        c = copy.deepcopy(candidate)
        kept, redundant = [], []
        for action in actions:
            d = {k: v for k, v in action.items() if v is not None}
            if d["family"] == "topology":
                k = d["stage"]
                if type(k) != int or not 0 <= k < self.st.K:
                    raise ValueError("Stage outside catalogue")
                prior = any(a["family"] == "topology" and a["close"] == d["close"] and
                            a["drop"] == d["drop"] and a["stage"] <= k for a in kept)
                already = all(d["close"] in c.plan.trees[j] and d["drop"] not in c.plan.trees[j]
                              for j in range(k, self.st.K))
                if prior and already:
                    redundant.append(d)
                    continue
            c = physical_action(self.st, c, record, action)
            kept.append(action)
        if actions and not kept:
            raise ValueError("No physical action in experiment")
        return kept, redundant

    def execute(self, proposal, label, experiment_id=None):
        if experiment_id is not None and experiment_id in self.work:
            return self.advance_work(experiment_id, self.work[experiment_id])
        p = copy.deepcopy(proposal)
        source = p["source"]
        injected = source in self.pinned_sources and source not in self.bank.entries
        if source in self.pinned_sources:
            c, r, origin = self.pinned_sources[source]
            self.bank.entries[source] = (c, r, origin)
        elif source == "incumbent":
            c, r = self.current, self.best
        elif source in self.bank.entries:
            c, r, origin = self.bank.entries[source]
        else:
            raise ValueError("Unknown own-arm source")
        previous_archive_count = self.counts["archive_experiments"]
        try:
            p["actions"], redundant = self.compose(c, r, p["actions"])
            if redundant:
                append(self.out / "composed_actions.jsonl", dict(experiment=experiment_id,
                    source=source, redundant_already_applied=redundant, physical_actions=p["actions"],
                    reason="Exact same exchange already propagated; no extra candidate or optimization introduced"))
            return super().execute(p, label, experiment_id)
        finally:
            if source in self.pinned_sources:
                delta = self.counts["archive_experiments"] - previous_archive_count
                self.counts["archive_experiments"] -= delta
                self.counts["analyst_snapshot_experiments"] += delta
            if injected:
                self.bank.entries.pop(source, None)

    def advance_work(self, key, work):
        if work["asset"] is None or work["proposal"]["mode"] != "cases":
            return super().advance_work(key, work)
        p = work["proposal"]
        k, cy = work["cycles"][0]
        offset = work.get("state_offset", 0)
        s = cy[offset]
        if self.ev.calls + 10 > self.budget:
            append(self.out / "incomplete_pairs.jsonl", dict(call=self.ev.calls, experiment=key,
                   actions=p["actions"], mode="cases", next_state=s))
            del self.work[key]
            return True
        if offset == 0:
            self.coverage.mark(k, cy)
        for branch, record_key in (("control", "cr"), ("asset", "ar")):
            c, r = work[branch], work[record_key]
            row = next(x for x in r["cases"] if x["stage"] == k and x["state_id"] == s)
            step = None
            if row.get("converged"):
                step, meta = ca.linear_step(self.st, c, row, p["radius"])
                self.counts["LP_proposals"] += 1
            bc, br = c, r
            for fraction in (1., .5, .25, .125, .0625):
                z = ca.apply_step(self.st, c, k, s, step, fraction) if step is not None else c
                rr = self.evaluate(z, dict(role="paired_" + branch, mode="cases", stage=k, state=s,
                           fraction=fraction, experiment=key, proposer=work["label"], no_op=step is None))
                if rank(rr) < rank(br):
                    bc, br = z, rr
            work[branch], work[record_key] = bc, br
        info = dict(actions=p["actions"], control=compact(work["cr"]), asset=compact(work["ar"]),
                    source=work["source"], proposer=work["label"], mode="cases", stage=k, state=s,
                    **credit_pair(work["before"], work["cr"], work["ar"]))
        self.feedback.append(info)
        self.feedback = self.feedback[-12:]
        append(self.out / "paired_feedback.jsonl", info)
        offset += 1
        if offset == len(cy):
            work["cycles"].pop(0)
            work["completed_cycles"] += 1
            offset = 0
        work["state_offset"] = offset
        done = not work["cycles"]
        append(self.out / "experiment_progress.jsonl", dict(id=key, proposer=work["label"], call=self.ev.calls,
            completed_cycles=work["completed_cycles"], remaining_cycles=len(work["cycles"]),
            state_offset=offset, control=work["control"].candidate_id, asset=work["asset"].candidate_id))
        if done:
            del self.work[key]
        return done
