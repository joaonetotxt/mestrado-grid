"""Selective native representation, without adding outer visits.

The unchanged interleaved scheduler owns resource and original-search slots.
Only an already allocated focused slot can be replaced. No auxiliary solver,
candidate sharing, altered objective, relaxed constraint or battery maturation.
"""
from collections import Counter
from interleaved_resources import InterleavedResourceScheduler
from effective_scheduler import Adapter
from selective_components import Coverage, SpatialBlock, PriorityTreeBlock

VERSION = 'selective_native_v1_20261016'
_ORIGINAL_FACTORY = None


class SelectiveScheduler(InterleavedResourceScheduler):
    def bind_stream(self, method, beta, regime='normais'):
        if regime not in ('normais', 'extremos'):
            raise ValueError('Unknown scenario regime')
        super().bind_stream(method, beta)
        self.selective_version = VERSION
        self.scenario_regime = regime
        self.selective_coverage = Coverage(self.codec.study.K, self.codec.study.S, 96101)
        self.selective_tree_slots = 0
        self.selective_counts = Counter()
        return self

    def choose(self, reference, contexts):
        context, block = super().choose(reference, contexts)
        if block is None or block.metadata.get('resource_coverage'):
            return context, block
        scope = block.metadata.get('scope')
        # Parent offers these slots only while official reliability is violated.
        # Replacing every third preserves both chronological and single-stage moves.
        if scope in ('effective_tree', 'effective_chronological_tree'):
            self.selective_tree_slots += 1
            if self.selective_tree_slots % 3 == 0:
                block = PriorityTreeBlock(self.codec, self.base, reference,
                                          96101 + self.selective_tree_slots)
                block.metadata.update(selective_version=VERSION,
                                      original_slot=scope, reliability_only=True)
                self.selective_counts['priority_tree'] += 1
            else:
                self.selective_counts['retained_tree'] += 1
            return None, block
        if scope != 'effective_channels':
            return context, block
        case, mode = self.selective_coverage.next(self.feedback)
        row = self.feedback.get(case, {})
        service = row.get('hourly_cut', 0) > row.get('hourly_curtailment', 0)
        regional = self.scenario_regime == 'normais' and mode in ('local', 'PQ')
        if regional:
            block = SpatialBlock(self.codec, self.base, reference, case,
                                 'P' if mode == 'local' else 'PQ', 1.,
                                 self.selective_coverage.steps - 1, 4, service)
            self.selective_counts['regional'] += 1
        else:
            broadmode = 'Q' if mode == 'Q' else ('P_service' if service else 'P')
            block = Adapter(self.codec, self.base, reference, 'channels',
                            case=case, mode=broadmode,
                            radius=.1 if broadmode in ('Q', 'P_service') else 1.)
            self.selective_counts['broad_' + broadmode] += 1
        block.metadata.update(selective_version=VERSION, fair_case=case,
                              coverage_requested_mode=mode, regional_enabled=regional,
                              scenario_regime=self.scenario_regime)
        return None, block

    def report(self):
        out = super().report()
        # Parent proposes an economic case before this wrapper replaces that
        # choice. Report actual selected coverage, not the discarded suggestion.
        out['focused_case_visits'] = {str(k): v for k, v in self.selective_coverage.counts.items()}
        out['stage_counter_scope'] = 'Parent reliability scheduling; priority-tree proposals span all stages'
        out['selective'] = dict(
            version=VERSION, regime=self.scenario_regime,
            counts=dict(self.selective_counts),
            focused_case_visits={str(k): v for k, v in self.selective_coverage.counts.items()},
            coverage_steps=self.selective_coverage.steps,
            case_sweep_focused_visits=len(self.selective_coverage.domain),
            all_modes_focused_visits=5 * len(self.selective_coverage.domain),
            resource_slots_preserved=True, original_slots_preserved=True,
            new_outer_visits=0, conditional_maturation=False,
            priority_tree_fraction_of_reliability_focus='1/3',
            regional_fraction_of_economic_focus='2/5 normal; 0 extreme',
            coverage_scope='Focused economic opportunities only; not wall time or exhaustive combinations',
            risk_and_constraint_comparator='unchanged', auxiliary_optimizer=False,
            bias='Declared physical line attributes, seeded spatial groups and fair case schedule')
        return out


def install(method, beta, regime):
    from meta_planning import persistent_search as p
    global _ORIGINAL_FACTORY
    if _ORIGINAL_FACTORY is None:
        _ORIGINAL_FACTORY = p.make_scheduler
    # One stream identity per worker process; checkpoints retain their own scheduler.
    p.make_scheduler = lambda codec, seed, mass, options: SelectiveScheduler(
        codec, seed, mass, options).bind_stream(method, beta, regime) if options.protected_resource_coverage else _ORIGINAL_FACTORY(codec, seed, mass, options)
