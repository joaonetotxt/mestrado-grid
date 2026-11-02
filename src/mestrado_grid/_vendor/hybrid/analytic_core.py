"""Isolated exact-evaluator search and transparent analytical observations."""
from campaign_support import *
from hybrid_core import RiskRefiner, FAMILIES, apply, opportunities, day_order
from rescue_core import Portfolio, coordinates
from search_scope import broad_proposal
from collections import Counter
from atomic_io import atomic_bytes
import math


def safe_save(path, obj):
    atomic_bytes(path, json.dumps(plain(obj), ensure_ascii=False, indent=2, allow_nan=False).encode())


class BudgetFinished(Exception):
    pass


class CycleCoverage:
    """Priority operates within the least-visited tier: no starvation."""
    def __init__(self):
        self.visits = Counter()

    def select(self, st, record):
        ordered = day_order(st, record)
        key = lambda x: (x[0], tuple(x[1]))
        chosen = min(ordered, key=lambda x: self.visits[key(x)])
        return chosen

    def mark(self, k, cy):
        self.visits[k, tuple(cy)] += 1


class RepairPortfolio(Portfolio):
    def repair_source(self, incumbent):
        rel, other, cost = coordinates(incumbent)
        choices = []
        for c, r, origin in self.entries.values():
            rr, oo, cc = coordinates(r)
            # Keep feasibility acquisition, and also enable electrical rescue once reliability is zero.
            if (rr < rel and oo > 0) or (rr <= rel and cc < cost and oo > 0):
                choices.append((c, r, origin))
        if not choices:
            return None
        e = min(choices, key=lambda x: (self.uses.get(x[0].candidate_id, 0),
                                       coordinates(x[1])[0], coordinates(x[1])[1], coordinates(x[1])[2]))
        self.uses[e[0].candidate_id] = self.uses.get(e[0].candidate_id, 0) + 1
        return copy.deepcopy(e)


def physical_action(st, c, record, action):
    """Reject invalid AI indices before calling existing catalogue transformations."""
    d = {k: v for k, v in action.items() if v is not None}
    f = d["family"]
    allowed = {
        "topology": ("stage", "close", "drop"), "cable": ("stage", "edge", "option"),
        "BESS_ESS1": ("stage", "unit", "q"), "BESS_ESS2": ("stage", "unit", "q"),
        "new_PV": ("stage", "j", "amount"), "new_WT": ("stage", "j", "amount"),
        "new_BIO": ("stage", "j", "amount"), "capacitor": ("stage", "j", "option"),
        "regulator": ("stage", "j", "option", "tap"), "retrofit": ("stage", "j"),
        "planned_reduce": ("j", "fraction"), "storage_remove": ("unit",),
        "device_controls": ("j", "control", "fraction")}
    absolute_control = f == "device_controls" and set(d) == {"family", "j", "control", "stage", "tap"}
    if f not in allowed or (not absolute_control and set(d) != {"family", *allowed[f]}):
        raise ValueError("Wrong fields for resource family")
    if "stage" in d and (type(d["stage"]) != int or not 0 <= d["stage"] < st.K):
        raise ValueError("Stage outside catalogue")
    ph = st.electrical["physical"]
    if "j" in d:
        n = (st.N if f.startswith("new_") or f == "planned_reduce" else
             st.G if f == "retrofit" else
             len(ph["capacitor_sites"]) if f == "capacitor" or d.get("control") == "cap_state"
             else len(ph["regulator_sites"]))
        if type(d["j"]) != int or not 0 <= d["j"] < n:
            raise ValueError("Site outside catalogue")
    if "fraction" in d and not 0 <= d["fraction"] <= 1:
        raise ValueError("Invalid retained fraction")
    if f.startswith("new_"):
        if st.new[d["j"]]["technology"] != f[4:]:
            raise ValueError("Technology/site mismatch")
        if not 0 < d["amount"] <= st.new[d["j"]]["max_MW"] - c.new_capacity[:, d["j"]].sum() + 1e-12:
            raise ValueError("Capacity exceeds remaining catalogue")
    if f.startswith("BESS_"):
        if d["unit"] not in [u["id"] for u in st.units] or not d["unit"].startswith(f[5:] + "@"):
            raise ValueError("Unknown BESS technology/site")
    if f == "storage_remove" and d["unit"] not in [u for u, k in c.plan.storage_buys]:
        raise ValueError("Storage not installed")
    if f == "device_controls" and d["control"] not in ("tap", "cap_state"):
        raise ValueError("Unknown device control")
    if f in ("capacitor", "regulator", "cable"):
        if f == "cable":
            edge = next((e for e in st.lines if e["id"] == d["edge"]), None)
            if edge is None or not -1 <= d["option"] < len(edge["options"]) or (edge["existing"] and d["option"] == 0):
                raise ValueError("Invalid purchased conductor")
        elif not -1 <= d["option"] < len(ph["capacitors" if f == "capacitor" else "regulators"]):
            raise ValueError("Unknown device")
    if f == "regulator" and not -16 <= d["tap"] <= 16:
        raise ValueError("Tap outside catalogue")
    if f == "topology":
        ids = {e["id"] for e in st.lines}
        k = d["stage"]
        from meta_planning.guided_blocks import cycle_path
        if d["close"] not in ids or d["close"] in c.plan.trees[k] or d["drop"] not in cycle_path(st.data, c.plan.trees[k], d["close"]):
            raise ValueError("Exchange does not define a tree")
    if f == "cable" and d["option"] == -1:
        z = copy.deepcopy(c)
        z.plan.line_buys = [p for p in z.plan.line_buys if not (p[0] == d["edge"] and p[2] == d["stage"])]
        line_schedule(st.data, z.plan)
    elif f in ("capacitor", "regulator"):
        z = copy.deepcopy(c)
        choice, control = ("cap_choice", "cap_state") if f == "capacitor" else ("reg_choice", "tap")
        j, opt, start = d["j"], d["option"], d["stage"]
        z.physical[choice][j] = 0 if opt == -1 else physical.choice_options(st, choice).index((opt, start))
        a = np.asarray(z.physical[control])
        a[:, :, j] = 0
        if opt >= 0:
            dev = ph["capacitors" if f == "capacitor" else "regulators"][opt]
            val = dev["steps"] if f == "capacitor" else d["tap"]
            for k in range(start, st.K):
                if f == "capacitor" or ph["regulator_sites"][j] in z.plan.trees[k]:
                    a[k, :, j] = val * np.asarray(dev["electrical_available"])
        z.physical[control] = a.tolist()
        normalize(st, z)
    elif absolute_control:
        z = copy.deepcopy(c)
        control, j, k, value = d["control"], d["j"], d["stage"], d["tap"]
        if control == "tap" and not -16 <= value <= 16:
            raise ValueError("Tap outside catalogue")
        if control == "cap_state":
            installed = z.physical["cap_choice"][j]
            if not installed:
                raise ValueError("Capacitor not installed")
            option, start = physical.choice_options(st, "cap_choice")[installed]
            if not 0 <= value <= ph["capacitors"][option]["steps"]:
                raise ValueError("Capacitor step outside catalogue")
        a = np.asarray(z.physical[control])
        a[k, :, j] = value
        z.physical[control] = a.tolist()
        normalize(st, z)
    else:
        z = apply(st, c, d, record)
    if z.candidate_id == c.candidate_id:
        raise ValueError("Action is a no-op")
    st.shape_check(z)
    return z


def credit_pair(source, control, asset):
    """A no-asset control can NEVER be credited to an installed resource."""
    cr, ar = coordinates(control), coordinates(asset)
    base = coordinates(source)
    both = valid(control) and valid(asset)
    return dict(control_dispatch_gain=(base[2] - cr[2]) if valid(source) and valid(control) else None,
                asset_incremental_gain=(cr[2] - ar[2]) if both else None,
                control_violation_reduction=sum(base[:2]) - sum(cr[:2]),
                asset_violation_reduction_vs_control=sum(cr[:2]) - sum(ar[:2]),
                asset_beats_control=rank(asset) < rank(control),
                causal_claim=False)


def observation(st, c, r, bank, feedback, coverage, remaining):
    """All physical sites/options plus all 96 cases; no access to a competing arm."""
    ph = st.electrical["physical"]
    cases = []
    for row in r.get("cases", []):
        k, s = row["stage"], row["state_id"]
        volts = np.hypot(row.get("voltage_real", []), row.get("voltage_imag", []))
        cases.append(dict(stage=k, state=s, cycle=next(i for i, cy in enumerate(st.cycles) if s in cy),
            violations=row.get("normalized_violations"), sub_P=row.get("sub_P_MW"), sub_Q=row.get("sub_Q_MVAr"),
            loss=row.get("loss_MW"), costs=row.get("hourly_costs"),
            worst_voltage_buses=sorted(zip(row.get("bus_order", []), volts), key=lambda x: -abs(x[1] - 1))[:4],
            highest_cut_buses=sorted([(b, float(c.operation.cut[k, j, s])) for j, b in enumerate(st.buses)],
                                    key=lambda x: -x[1])[:4],
            highest_flow_edges=sorted(row.get("branches", []), key=lambda x: -max(x["I_from_A"], x["I_to_A"]))[:3]))
    # Preserve full option economics, impedances, ratings and directional reliability in compact rows.
    lines = [dict(id=e["id"], i=e["i"], j=e["j"], existing=e["existing"],
             options=[dict(option=o["id"], capex=o["capex"], maintenance=o["maintenance_per_year"],
               capacity=o["capacity_MW"], current=o["nominal_current_A"], failure=o["failure_rate"],
               repair=o["repair_hours"], r=st.eopts[e["id"], o["id"]]["r"],
               x=st.eopts[e["id"], o["id"]]["x"]) for o in e["options"]]) for e in st.lines]
    selected = bank.repair_source(r)  # Selection accounting is not consumed by observation.
    if selected:
        key = selected[0].candidate_id
        bank.uses[key] -= 1
    answer = plain(dict(input_id=st.input_id, alpha=st.c["alpha"], beta=st.c["beta"], regime="extremes",
        full_evaluations_remaining=remaining, current=compact(r),
        finance=r.get("metrics", {}).get("finance"), cycles=st.cycles, probability=st.probabilities,
        strict_limits={k: st.c[k] for k in ("SAIDI_limits", "SAIFI_limits")},
        installed=dict(trees=c.plan.trees, line_buys=c.plan.line_buys, storage_buys=c.plan.storage_buys,
                       new_capacity=c.new_capacity, choices={k: v for k, v in c.physical.items() if k.endswith("_choice")}),
        catalogue=dict(lines=lines, storage=st.units,
            new_generation=[dict(j=j, **{key: g[key] for key in ("id", "bus", "technology", "max_MW", "availability",
                                "capex_per_MW", "fixed_per_MW_year", "variable_per_MWh", "curtail_per_MWh")}) for j, g in enumerate(st.new)],
            existing_generation=[dict(j=j, **g) for j, g in enumerate(st.old)],
            capacitor_sites=ph["capacitor_sites"], regulator_sites=ph["regulator_sites"],
            capacitors=ph["capacitors"], regulators=ph["regulators"]),
        bus_load_MW=st.base_load, cases=cases,
        archives=[dict(id=cc.candidate_id, plan_id=cc.plan_id, result=compact(rr),
                       trees=cc.plan.trees, line_buys=cc.plan.line_buys, storage=cc.plan.storage_buys,
                       capacities=cc.new_capacity) for cc, rr, origin in bank.entries.values()],
        feedback=feedback, coverage=[dict(stage=k, states=cy, visits=n) for (k, cy), n in coverage.visits.items()],
        information_boundary="Only this arm. No other trial result or historical optimized finalist supplied."))
    # Compact, labelled tables preserve EVERY catalogue location and option.
    # Numeric display compression is never fed back into physics or finance.
    answer["catalogue"]["lines"] = dict(
        columns=["id", "from", "to", "existing", "options"],
        option_columns=["option", "capex", "maintenance", "capacity", "current", "r", "x", "failure", "repair"],
        rows=[[e["id"], e["i"], e["j"], e["existing"],
               [[o[k] for k in ("option", "capex", "maintenance", "capacity", "current", "r", "x", "failure", "repair")]
                for o in e["options"]]] for e in lines])
    types = {}
    sites = []
    for u in st.units:
        typ = u["id"].split("@")[0]
        types[typ] = {k: v for k, v in u.items() if k not in ("id", "bus")}
        sites.append([u["id"], u["bus"]])
    answer["catalogue"]["storage"] = dict(types=types, id_bus=sites)
    answer["catalogue"]["new_generation"] = dict(
        columns=["j", "id", "bus", "technology", "max_MW", "capex_per_MW", "fixed_per_MW_year",
                 "variable_per_MWh", "curtail_per_MWh", "availability_by_state"],
        rows=[[j, *[g[k] for k in ("id", "bus", "technology", "max_MW", "capex_per_MW",
              "fixed_per_MW_year", "variable_per_MWh", "curtail_per_MWh")],
               [g["availability"][str(s)] for s in range(st.S)]] for j, g in enumerate(st.new)])
    for kind in ("capacitors", "regulators"):
        answer["catalogue"][kind] = [{k: v for k, v in x.items() if k not in ("provenance", "electrical_available")}
                                     for x in ph[kind]]
    answer.pop("bus_load_MW")
    answer["bus_demand"] = dict(columns=["bus", "peak_MW_by_stage", "mean_MW_by_stage", "Q_over_P_min", "Q_over_P_max"],
        rows=[[b, st.base_load[:, j].max(axis=1).tolist(), (st.base_load[:, j] @ st.probabilities).tolist(),
               min(st.electrical["q_ratio"][j]), max(st.electrical["q_ratio"][j])] for j, b in enumerate(st.buses)])
    cost_keys = sorted({key for x in cases for key in (x["costs"] or {})})
    answer["cases"] = dict(
        columns=["stage", "state", "cycle", "nonzero_violations", "sub_P", "sub_Q", "loss", "cost_components",
                 "worst_voltage_buses", "highest_cut_buses", "highest_flow_edges"],
        flow_columns=["edge", "option", "P_from", "Q_from", "I_from", "I_to"],
        cost_columns=cost_keys,
        rows=[[x["stage"], x["state"], x["cycle"], {k: v for k, v in (x["violations"] or {}).items() if v != 0},
               x["sub_P"], x["sub_Q"], x["loss"], [(x["costs"] or {}).get(k, 0) for k in cost_keys],
               x["worst_voltage_buses"], x["highest_cut_buses"],
               [[b[k] for k in ("id", "option", "P_from_MW", "Q_from_MVAr", "I_from_A", "I_to_A")]
                for b in x["highest_flow_edges"]]] for x in cases])
    def display(x):
        if isinstance(x, float):
            return float(format(x, ".7g"))
        if isinstance(x, list):
            return [display(v) for v in x]
        if isinstance(x, dict):
            return {k: display(v) for k, v in x.items()}
        return x
    answer["display_precision"] = "7 significant digits in analyst observations ONLY; exact stored controls and evaluator values unchanged."
    return display(answer)


class Kernel:
    def __init__(self, st, out, budget):
        self.st, self.out, self.budget = st, Path(out), budget
        self.out.mkdir(parents=True, exist_ok=False)
        self.ev = CountingEvaluator(st)
        self.ref = RiskRefiner(st)
        self.best = self.current = None
        self.bank = RepairPortfolio()
        self.coverage = CycleCoverage()
        self.feedback = []
        self.counts = Counter()
        self.started = time.monotonic()
        self.cpu = time.process_time()
        self.seen = {}
        self.work = {}

    def progress(self, state="running"):
        safe_save(self.out / "status.json", dict(state=state, utc=now(), pid=os.getpid(),
            full_calls=self.ev.calls, AC_cases=self.ev.ac_cases, budget=self.budget,
            elapsed_s=time.monotonic() - self.started, best=compact(self.best) if self.best else None,
            counters=dict(self.counts)))

    def evaluate(self, c, event):
        if self.ev.calls >= self.budget:
            raise BudgetFinished()
        r = self.ev.evaluate(c)
        accepted = self.best is None or rank(r) < rank(self.best)
        if accepted:
            self.current, self.best = copy.deepcopy(c), r
        self.bank.offer(c, r, event)
        append(self.out / "history.jsonl", dict(call=self.ev.calls, event=event, accepted=accepted, result=compact(r)))
        self.progress()
        return r

    def refine(self, c, r, k, cy, radius=.5, mode="cycle", role="dispatch", max_trials=5):
        self.coverage.mark(k, cy)
        if mode == "cases":
            for s in cy:
                row = next(x for x in r["cases"] if x["stage"] == k and x["state_id"] == s)
                if not row.get("converged"):
                    continue
                step, meta = ca.linear_step(self.st, c, row, radius)
                self.counts["LP_proposals"] += 1
                if step is None:
                    continue
                c, r = self.backtrack(c, r, lambda fraction: ca.apply_step(self.st, c, k, s, step, fraction),
                                     dict(role=role, mode=mode, stage=k, state=s), max_trials)
            return c, r
        rows = {(x["stage"], x["state_id"]): x for x in r.get("cases", [])}
        if not all(rows.get((k, s), {}).get("converged") for s in cy):
            return c, r
        step, meta = self.ref.propose(c, k, cy, rows, radius, record=r)
        self.counts["LP_proposals"] += 1
        if step is None:
            return c, r
        return self.backtrack(c, r, lambda fraction: self.ref.apply(c, k, cy, step, fraction),
                              dict(role=role, mode=mode, stage=k, states=cy), max_trials)

    def backtrack(self, c, r, build, event, max_trials=5):
        for fraction in (1., .5, .25, .125, .0625)[:max_trials]:
            z = build(fraction)
            if z.candidate_id == c.candidate_id:
                self.counts["no_op_dispatch"] += 1
                continue
            rr = self.evaluate(z, event | dict(fraction=fraction))
            self.counts["backtracking_trials"] += 1
            if rank(rr) < rank(r):
                self.counts["accepted_reduced_steps" if fraction < 1 else "accepted_full_steps"] += 1
                return z, rr
        return c, r

    def execute(self, proposal, label, experiment_id=None):
        """One complete daily visit; retain both branches for subsequent visits."""
        key = experiment_id if experiment_id is not None else f"{label}_{self.ev.calls}"
        if key in self.work:
            work = self.work[key]
            return self.advance_work(key, work)
        source = proposal["source"]
        if source == "incumbent":
            c, r = copy.deepcopy(self.current), self.best
        elif source in self.bank.entries:
            c, r, origin = copy.deepcopy(self.bank.entries[source])
            self.counts["archive_experiments"] += 1
        else:
            raise ValueError("Unknown or expired own-arm source")
        for cy in proposal["cycles"]:
            if not 0 <= cy["stage"] < self.st.K or not 0 <= cy["cycle"] < len(self.st.cycles):
                raise ValueError("Unknown daily cycle")
        cycles = [(x["stage"], self.st.cycles[x["cycle"]]) for x in proposal["cycles"]]
        work = dict(proposal=proposal, label=label, control=c, cr=r, before=r,
                    cycles=cycles * proposal["rounds"], completed_cycles=0, source=c.candidate_id)
        if not proposal["actions"]:
            work.update(asset=None, ar=None)
        else:
            asset = copy.deepcopy(c)
            for a in proposal["actions"]:
                asset = physical_action(self.st, asset, r, a)
                self.counts["proposed_" + a["family"]] += 1
            ar = self.evaluate(asset, dict(role="asset", source=c.candidate_id, actions=proposal["actions"], proposer=label))
            work.update(asset=asset, ar=ar)
        self.work[key] = work
        return self.advance_work(key, work)

    def advance_work(self, key, work):
        proposal, label = work["proposal"], work["label"]
        k, cy = work["cycles"][0]
        control, cr, asset, ar = work["control"], work["cr"], work["asset"], work["ar"]
        if asset is None:
            control, cr = self.refine(control, cr, k, cy, proposal["radius"], proposal["mode"], label)
        else:
        # Both treatments receive exactly the same number of AC calls in a completed paired experiment.
        # Search the five fractions on BOTH sides; do not give the accepted side fewer calls.
            self.coverage.mark(k, cy)
            if self.ev.calls + 10 > self.budget:
                append(self.out / "incomplete_pairs.jsonl", dict(call=self.ev.calls, actions=proposal["actions"]))
                del self.work[key]
                return True
            control, cr = self.paired_refine(control, cr, k, cy, proposal["radius"], "control")
            asset, ar = self.paired_refine(asset, ar, k, cy, proposal["radius"], "asset")
            info = dict(actions=proposal["actions"], control=compact(cr), asset=compact(ar),
                        source=work["source"], proposer=label, cycles_so_far=work["completed_cycles"] + 1,
                        **credit_pair(work["before"], cr, ar))
            self.feedback.append(info)
            self.feedback = self.feedback[-12:]
            append(self.out / "paired_feedback.jsonl", info)
        work.update(control=control, cr=cr, asset=asset, ar=ar)
        work["cycles"].pop(0)
        work["completed_cycles"] += 1
        append(self.out / "experiment_progress.jsonl", dict(id=key, proposer=label, call=self.ev.calls,
               completed_cycles=work["completed_cycles"], remaining_cycles=len(work["cycles"]),
               control=control.candidate_id, asset=asset.candidate_id if asset else None))
        done = not work["cycles"]
        if done:
            del self.work[key]
        return done

    def paired_refine(self, c, r, k, cy, radius, branch):
        rows = {(x["stage"], x["state_id"]): x for x in r.get("cases", [])}
        step = None
        if all(rows.get((k, s), {}).get("converged") for s in cy):
            step, meta = self.ref.propose(c, k, cy, rows, radius, record=r)
            self.counts["LP_proposals"] += 1
        best_c, best_r = c, r
        for fraction in (1., .5, .25, .125, .0625):
            z = self.ref.apply(c, k, cy, step, fraction) if step is not None else c
            rr = self.evaluate(z, dict(role="paired_" + branch, fraction=fraction, stage=k, states=cy,
                                      no_op=step is None))
            if rank(rr) < rank(best_r):
                best_c, best_r = z, rr
        return best_c, best_r

    def finish(self, initial, extra):
        ev = CountingEvaluator(self.st)
        result = ev.evaluate(self.current)
        if not audit_agrees(result, self.best):
            raise AssertionError("Final audit differs")
        finance = finance_checks(result)
        safe_save(self.out / "candidate.json", self.current.to_dict())
        atomic_bytes(self.out / "validation.json.gz", gzip.compress(json.dumps(plain(result), allow_nan=False).encode()))
        summary = dict(initial=initial, final=compact(result), budget=self.budget, full_calls=self.ev.calls,
                       AC_cases=self.ev.ac_cases, audit_calls=ev.calls, audit_AC_cases=ev.ac_cases,
                       finance=finance, counters=dict(self.counts), elapsed_s=time.monotonic() - self.started,
                       cpu_s=time.process_time() - self.cpu,
                       cycles_visited=len(self.coverage.visits), expected_cycles=self.st.K * len(self.st.cycles),
                       production_modified=False, **extra)
        safe_save(self.out / "summary.json", summary)
        self.progress("completed")
        return summary
