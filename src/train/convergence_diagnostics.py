"""Opt-in convergence diagnostics; no changes to the formal PN/PP evaluator."""
import bisect
import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback


def plateau_gate(rows, config):
    """Require stability of both the unmasked and fixed-corruption objectives.

    This is an operational optimization check, never a proof of capacity fit.
    Validation does not authorize stopping while the training probe improves.
    """
    window = int(config.get('convergence_patience', 8))
    minimum = float(config.get('convergence_min_epochs', 36))
    tolerance = float(config.get('convergence_relative_tolerance', 0.002))
    if len(rows) < window or rows[-1]['epoch'] < minimum:
        return False, {'reason': 'minimum_observation_not_met'}
    recent = rows[-window:]
    names = ['train_eval_nll', 'train_eval_velocity',
             'train_masked_objective', 'train_masked_velocity']
    if float(config.get('pn_cached_velocity_cdf_w1_lambda', 0)) > 0:
        names.append('train_masked_cdf_w1')
    spans = {}
    for name in names:
        values = [r.get(name, float('nan')) for r in recent]
        if not all(math.isfinite(v) for v in values):
            return False, {'reason': 'missing_or_nonfinite_probe', 'metric': name}
        spans[name] = (max(values) - min(values)) / max(abs(sum(values) / len(values)), 1e-8)
    passed = all(span <= tolerance for span in spans.values())
    confirmations = int(config.get('convergence_full_train_confirmations', 3))
    tail = rows[-confirmations:]
    full_spans = {}
    if len(tail) < confirmations or any('full_train_masked_objective' not in row for row in tail):
        passed = False
    else:
        full_names = ['full_train_eval_nll', 'full_train_eval_velocity',
                      'full_train_masked_objective', 'full_train_masked_velocity']
        if float(config.get('pn_cached_velocity_cdf_w1_lambda', 0)) > 0:
            full_names.append('full_train_masked_cdf_w1')
        for name in full_names:
            values = [row[name] for row in tail]
            full_spans[name] = ((max(values) - min(values)) / max(abs(sum(values) / len(values)), 1e-8)
                                if all(math.isfinite(v) for v in values) else float('inf'))
        passed = passed and all(span <= tolerance for span in full_spans.values())
    return passed, {'reason': 'training_probe_plateau' if passed else 'training_probe_still_changes',
                    'relative_spans': spans, 'full_train_relative_spans': full_spans,
                    'full_train_confirmations': confirmations, 'window': window, 'tolerance': tolerance}


@contextmanager
def isolated_rng(seed):
    """Diagnostics must not advance training's dropout/masking/data RNGs."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def fixed_probe_indices(dataset, limit, seed=2026):
    """Round-robin scores; hash-sort rows independently of training seed/order."""
    groups = []
    previous = 0
    for item, end in zip(dataset.items, dataset.cumulative_sizes):
        candidates = list(range(previous, end))
        key = str(item['path'])
        candidates.sort(key=lambda index: hashlib.sha256(
            f'{seed}:{key}:{index - previous}'.encode()).digest())
        groups.append((key, candidates))
        previous = end
    groups.sort(key=lambda pair: hashlib.sha256(f'{seed}:{pair[0]}'.encode()).digest())
    indices = []
    depth = 0
    while len(indices) < min(limit, len(dataset)):
        added = False
        for _, group in groups:
            if depth < len(group):
                indices.append(group[depth])
                added = True
                if len(indices) == min(limit, len(dataset)):
                    break
        if not added:
            break
        depth += 1
    return indices


class ConvergenceDiagnostics(TrainerCallback):
    def __init__(self, trainer, config, component_fn, cdf_fn):
        self.trainer, self.config = trainer, dict(config)
        self.component_fn, self.cdf_fn = component_fn, cdf_fn
        self.output = Path(trainer.args.output_dir)
        self.history_path = self.output / 'convergence_history.jsonl'
        self.rows = []
        self.probes = None

    def _prepare_probes(self):
        dataset = copy.copy(self.trainer.train_dataset)
        dataset._prepared_cache = copy.copy(dataset._prepared_cache)
        dataset._cache = copy.copy(dataset._cache)
        # Diagnostic target uses only training GT, including for the baseline.
        dataset.pn_velocity_target_cache = True
        count = int(self.config.get('convergence_probe_examples', 512))
        indices = fixed_probe_indices(dataset, count)
        self.probes = [dataset[index] for index in indices]
        identity = []
        for index in indices:
            item_index = bisect.bisect_right(dataset.cumulative_sizes, index)
            start = 0 if item_index == 0 else dataset.cumulative_sizes[item_index - 1]
            item = dataset.items[item_index]
            local = index - start
            perf, window = divmod(local, len(item['windows']))
            sources = item.get('selected_performance_sources')
            identity.append({'path': item['path'], 'performance_slot': perf,
                             'performance_source': sources[perf] if sources else None,
                             'window': item['windows'][window]})
        # The actual post-sidecar-filter dataset, not an estimated manifest.
        canonical = sorted([
            {k: item.get(k) for k in ['path', 'windows', 'selected_performance_sources',
                                     'effective_performances', 'effective_examples']}
            for item in dataset.items], key=lambda x: str(x['path']))
        manifest = {'schema': 1, 'probe_identity': identity,
                    'probe_hash': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                    'dataset_hash': hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
                    'actual_train_examples': len(dataset), 'actual_train_scores': len(dataset.items),
                    'multi_gt_scores': sum(i['effective_performances'] >= 2 for i in dataset.items),
                    'aggregation': 'valid-note weighted batch component means',
                    'train_probe_selection': 'hash-ordered score round-robin, seed2026',
                    'validation_target': 'NLL only; no validation GT enters the train cache'}
        target = self.output / 'convergence_probe_manifest.json'
        if target.exists():
            old = json.loads(target.read_text())
            if old['probe_hash'] != manifest['probe_hash'] or old['dataset_hash'] != manifest['dataset_hash']:
                raise ValueError('Convergence resume changed the probe or training dataset')
        target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')

    def _measure(self, dataset, training_mode=False):
        trainer = self.trainer
        model = trainer.model
        config = trainer._model_config(model)
        collator = copy.deepcopy(trainer.data_collator)
        collator.dagger_prefix_training = False
        collator.stable_dynamics_training = False
        loader = DataLoader(dataset, batch_size=trainer.args.per_device_eval_batch_size,
                            shuffle=False, num_workers=0, collate_fn=collator)
        sums, denominator, cdf_denominator = {}, 0.0, 0.0
        was_training = model.training
        try:
            model.train(training_mode)
            with isolated_rng(9071), torch.no_grad(), trainer.compute_loss_context_manager():
                for batch in loader:
                    batch = trainer._prepare_inputs(batch)
                    target = batch.pop('pn_velocity_target_pmf', None)
                    target_mask = batch.pop('pn_velocity_target_mask', None)
                    batch.pop('pn_group_index', None)
                    batch.pop('pn_group_sizes', None)
                    outputs = model(**batch)
                    weight = float(batch['attention_mask'].sum())
                    components = self.component_fn(
                        config, outputs.logits, batch['labels_continuous'], batch['attention_mask'],
                        labels_epr_bins=batch.get('labels_epr_bins'),
                        score_shared_raw=batch.get('score_shared_raw'),
                        label_valid_mask=batch.get('label_valid_mask'))
                    values = {'nll': float(outputs.loss),
                              **{name: float(value) for name, value in components.items()
                                 if name in ('ioi', 'duration', 'velocity', 'pedal')}}
                    objective = values['nll']
                    if target is not None:
                        cdf_weight = float((target_mask & batch['attention_mask'].bool()).sum())
                        cdf = self.cdf_fn(config, outputs.logits, batch['attention_mask'], target, target_mask)
                        sums['cdf_w1'] = sums.get('cdf_w1', 0.0) + cdf_weight * float(cdf)
                        cdf_denominator += cdf_weight
                        objective += float(self.config.get('pn_cached_velocity_cdf_w1_lambda', 0)) * float(cdf)
                    values['objective'] = objective
                    for name, value in values.items():
                        sums[name] = sums.get(name, 0.0) + weight * value
                    denominator += weight
        finally:
            model.train(was_training)
        values = {name: value / max(denominator, 1) for name, value in sums.items() if name != 'cdf_w1'}
        if denominator == 0:
            raise ValueError('Convergence diagnostics require a nonempty train/validation dataset')
        if cdf_denominator:
            values['cdf_w1'] = sums['cdf_w1'] / cdf_denominator
        values['eligible_note_fraction'] = cdf_denominator / max(denominator, 1)
        return values

    def on_train_begin(self, args, state, control, **kwargs):
        if args.world_size != 1:
            raise ValueError('Convergence experiment currently requires one training GPU per run')
        with isolated_rng(2026):
            self._prepare_probes()
        if self.history_path.exists():
            self.rows = [json.loads(line) for line in self.history_path.read_text().splitlines() if line]
            self.rows = [row for row in self.rows if row['step'] <= state.global_step]
            self.history_path.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def on_epoch_end(self, args, state, control, **kwargs):
        eval_probe = self._measure(self.probes)
        masked_probe = self._measure(self.probes, training_mode=True)
        validation = self._measure(self.trainer.eval_dataset)
        row = {'event': 'convergence_diagnostic', 'epoch': float(state.epoch),
               'step': int(state.global_step), 'learning_rate': self.trainer.optimizer.param_groups[0]['lr']}
        for prefix, values in [('train_eval', eval_probe), ('train_masked', masked_probe), ('validation', validation)]:
            row.update({f'{prefix}_{key}': value for key, value in values.items()})
        confirmations = int(self.config.get('convergence_full_train_confirmations', 3))
        if state.epoch >= args.num_train_epochs - confirmations + 1 - 1e-6:
            diagnostic_dataset = copy.copy(self.trainer.train_dataset)
            diagnostic_dataset.pn_velocity_target_cache = True
            for prefix, training in [('full_train_eval', False), ('full_train_masked', True)]:
                values = self._measure(diagnostic_dataset, training_mode=training)
                row.update({f'{prefix}_{key}': value for key, value in values.items()})
        self.rows.append(row)
        passed, details = plateau_gate(self.rows, self.config)
        row['plateau_passed'], row['gate'] = passed, details
        with self.history_path.open('a') as file:
            file.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
        # Equal optimizer-step budgets across conditions: no per-arm stopping.
        return control

    def on_train_end(self, args, state, control, **kwargs):
        passed, details = plateau_gate(self.rows, self.config)
        valid_rows = [row for row in self.rows if math.isfinite(float(row.get('validation_nll', float('nan'))))]
        best = min(valid_rows, key=lambda row: row['validation_nll']) if valid_rows else None
        result = {'status': 'optimization_plateau' if passed else 'inconclusive_not_at_plateau',
                  'last_epoch': state.epoch, 'last_step': state.global_step,
                  'gate': details, 'best_validation_epoch': best['epoch'] if best else None,
                  'best_validation_nll': best['validation_nll'] if best else None,
                  'warning': 'Operational probe criterion; does not prove capacity fit or global optimum.'}
        (self.output / 'convergence_result.json').write_text(json.dumps(result, indent=2) + '\n')
