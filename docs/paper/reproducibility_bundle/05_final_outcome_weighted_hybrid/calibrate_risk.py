"""Fit and evaluate post-hoc calibration for auxiliary risk heads.

The model was trained with positive-class weighting so its sigmoid outputs are
useful danger scores, but not literal event probabilities. This script keeps
model fitting, probability calibration, and final evaluation disjoint:

* model training:   [0, calibration_begin)
* Platt fitting:    [calibration_begin, calibration_end)
* final evaluation: [calibration_end, test_end)

Logits are accumulated into streaming histograms, so fitting does not retain
millions of predictions in memory and needs no scikit-learn.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MahjongGBDataset
from model import CNNModel


class BinaryHistogram:
    def __init__(self, bins=1000, reliability_bins=10):
        self.bins = bins
        self.pos = np.zeros(bins, dtype=np.int64)
        self.neg = np.zeros(bins, dtype=np.int64)
        self.rel_count = np.zeros(reliability_bins, dtype=np.int64)
        self.rel_positive = np.zeros(reliability_bins, dtype=np.int64)
        self.rel_probability = np.zeros(reliability_bins, dtype=np.float64)
        self.brier_sum = 0.0
        self.count = 0

    def update(self, probability, target):
        probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
        target = np.asarray(target, dtype=np.int8)
        index = np.minimum((probability * self.bins).astype(np.int64), self.bins - 1)
        self.pos += np.bincount(index[target == 1], minlength=self.bins)
        self.neg += np.bincount(index[target == 0], minlength=self.bins)
        rel_index = np.minimum(
            (probability * len(self.rel_count)).astype(np.int64),
            len(self.rel_count) - 1)
        self.rel_count += np.bincount(rel_index, minlength=len(self.rel_count))
        self.rel_positive += np.bincount(
            rel_index, weights=target, minlength=len(self.rel_count)).astype(np.int64)
        self.rel_probability += np.bincount(
            rel_index, weights=probability, minlength=len(self.rel_count))
        self.brier_sum += float(np.square(probability - target).sum())
        self.count += int(target.size)

    def result(self):
        positives = int(self.pos.sum())
        negatives = int(self.neg.sum())
        tp = np.cumsum(self.pos[::-1], dtype=np.float64)
        fp = np.cumsum(self.neg[::-1], dtype=np.float64)
        recall = tp / max(1, positives)
        false_positive_rate = fp / max(1, negatives)
        precision = tp / np.maximum(1.0, tp + fp)
        auroc = float(np.trapezoid(
            np.r_[0.0, recall, 1.0], np.r_[0.0, false_positive_rate, 1.0]))
        auprc = float(np.trapezoid(
            np.r_[1.0, precision], np.r_[0.0, recall]))
        reliability = []
        ece = 0.0
        for i, count in enumerate(self.rel_count):
            if count:
                confidence = float(self.rel_probability[i] / count)
                frequency = float(self.rel_positive[i] / count)
                ece += count / max(1, self.count) * abs(confidence - frequency)
            else:
                confidence = frequency = 0.0
            reliability.append({
                'low': i / len(self.rel_count),
                'high': (i + 1) / len(self.rel_count),
                'count': int(count),
                'mean_probability': confidence,
                'positive_frequency': frequency,
            })
        return {
            'count': self.count,
            'positives': positives,
            'prevalence': positives / max(1, self.count),
            'auroc_histogram_approx': auroc,
            'auprc_histogram_approx': auprc,
            'brier_score': self.brier_sum / max(1, self.count),
            'expected_calibration_error_10bin': float(ece),
            'reliability': reliability,
        }


class LogitHistogram:
    """Streaming logit/label counts used to fit a two-parameter calibrator."""

    def __init__(self, bins=4000, low=-20.0, high=20.0):
        self.bins = bins
        self.low = float(low)
        self.high = float(high)
        self.pos = np.zeros(bins, dtype=np.int64)
        self.neg = np.zeros(bins, dtype=np.int64)

    def update(self, logit, target):
        logit = np.clip(np.asarray(logit, dtype=np.float64), self.low, self.high)
        target = np.asarray(target, dtype=np.int8)
        scaled = (logit - self.low) / (self.high - self.low)
        index = np.minimum((scaled * self.bins).astype(np.int64), self.bins - 1)
        self.pos += np.bincount(index[target == 1], minlength=self.bins)
        self.neg += np.bincount(index[target == 0], minlength=self.bins)

    def fit_platt(self, max_iter=100):
        total = self.pos + self.neg
        active = total > 0
        if not active.any() or self.pos.sum() == 0 or self.neg.sum() == 0:
            raise RuntimeError('Platt fitting requires positive and negative samples')
        width = (self.high - self.low) / self.bins
        centre = self.low + (np.arange(self.bins) + 0.5) * width
        x = torch.from_numpy(centre[active]).double()
        positive = torch.from_numpy(self.pos[active].astype(np.float64)).double()
        count = torch.from_numpy(total[active].astype(np.float64)).double()

        # Constrain the slope to be positive so calibration preserves ranking.
        raw_scale = torch.tensor(
            float(np.log(np.expm1(1.0))), dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw_scale, bias], max_iter=max_iter, tolerance_grad=1e-12,
            tolerance_change=1e-12, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            scale = F.softplus(raw_scale) + 1e-8
            calibrated_logit = scale * x + bias
            loss = ((count - positive) * F.softplus(calibrated_logit)
                    + positive * F.softplus(-calibrated_logit)).sum() / count.sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = float(closure().detach())
        scale = float((F.softplus(raw_scale) + 1e-8).detach())
        fitted_bias = float(bias.detach())
        return {
            'scale': scale,
            'bias': fitted_bias,
            'fit_log_loss': final_loss,
            'count': int(total.sum()),
            'positives': int(self.pos.sum()),
            'prevalence': float(self.pos.sum() / total.sum()),
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, device):
    model = CNNModel()
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def make_loader(begin, end, args, device):
    dataset = MahjongGBDataset(begin, end, False, cache_size=args.cache_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    return dataset, loader


def model_auxiliary(model, obs, mask, seq, player, device):
    inputs = {
        'observation': obs.to(device, non_blocking=True),
        'action_mask': mask.to(device, non_blocking=True),
        'discard_seq': seq.to(device, non_blocking=True),
        'discard_player': player.to(device, non_blocking=True),
    }
    with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == 'cuda'):
        _, auxiliary = model(inputs, return_aux=True)
    return auxiliary


def fit_calibrators(model, loader, args, device):
    histograms = {
        'aggregate_risk': LogitHistogram(),
        'opponent_risk': LogitHistogram(),
        'opponent_tenpai': LogitHistogram(),
    }
    processed_samples = 0
    with torch.no_grad():
        for batch_index, (obs, mask, _, _, seq, player, target) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            auxiliary = model_auxiliary(model, obs, mask, seq, player, device)
            legal = mask[:, 2:36].bool()
            opponent_legal = legal[:, None, :].expand(-1, 3, -1)
            histograms['aggregate_risk'].update(
                auxiliary['risk'].float().cpu()[legal].numpy(),
                target['risk'][legal].numpy())
            histograms['opponent_risk'].update(
                auxiliary['risk_opp'].float().cpu()[opponent_legal].numpy(),
                target['risk_opp'][opponent_legal].numpy())
            histograms['opponent_tenpai'].update(
                auxiliary['tenpai_opp'].float().cpu().reshape(-1).numpy(),
                target['tenpai_opp'].reshape(-1).numpy())
            processed_samples += int(obs.shape[0])
            if (batch_index + 1) % 100 == 0:
                print('[fit-calibration] batches=%d' % (batch_index + 1), flush=True)
    return ({name: histogram.fit_platt() for name, histogram in histograms.items()},
            processed_samples)


def calibrated_probability(logit, parameters):
    value = parameters['scale'] * np.asarray(logit, dtype=np.float64) + parameters['bias']
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def evaluate_calibration(model, loader, calibrators, args, device):
    raw = {name: BinaryHistogram() for name in calibrators}
    calibrated = {name: BinaryHistogram() for name in calibrators}
    severity_error = 0.0
    severity_count = 0
    processed_samples = 0
    with torch.no_grad():
        for batch_index, (obs, mask, _, _, seq, player, target) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            auxiliary = model_auxiliary(model, obs, mask, seq, player, device)
            legal = mask[:, 2:36].bool()
            opponent_legal = legal[:, None, :].expand(-1, 3, -1)
            values = {
                'aggregate_risk': (
                    auxiliary['risk'].float().cpu()[legal].numpy(),
                    target['risk'][legal].numpy()),
                'opponent_risk': (
                    auxiliary['risk_opp'].float().cpu()[opponent_legal].numpy(),
                    target['risk_opp'][opponent_legal].numpy()),
                'opponent_tenpai': (
                    auxiliary['tenpai_opp'].float().cpu().reshape(-1).numpy(),
                    target['tenpai_opp'].reshape(-1).numpy()),
            }
            for name, (logit, label) in values.items():
                raw_probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
                raw[name].update(raw_probability, label)
                calibrated[name].update(
                    calibrated_probability(logit, calibrators[name]), label)
            positive = legal & target['risk'].bool()
            if positive.any():
                prediction = auxiliary['risk_loss'].sigmoid().float().cpu()[positive]
                truth = target['risk_loss'][positive]
                severity_error += float(torch.abs(prediction - truth).sum())
                severity_count += int(positive.sum())
            processed_samples += int(obs.shape[0])
            if (batch_index + 1) % 100 == 0:
                print('[test-calibration] batches=%d' % (batch_index + 1), flush=True)
    return {
        'processed_samples': processed_samples,
        'raw': {name: histogram.result() for name, histogram in raw.items()},
        'calibrated': {name: histogram.result() for name, histogram in calibrated.items()},
        'positive_deal_in_severity_mae': severity_error / max(1, severity_count),
        'positive_deal_in_severity_count': severity_count,
    }


def validate_splits(args, parser):
    if not (0.0 <= args.calibration_begin < args.calibration_end < args.test_end <= 1.0):
        parser.error('require 0 <= calibration-begin < calibration-end < test-end <= 1')


def main():
    parser = argparse.ArgumentParser(description='Fit and test risk calibration')
    parser.add_argument('--checkpoint', default='model.pkl')
    parser.add_argument('--data-dir', required=True,
                        help='Processed directory containing count.json and numbered .npz files')
    parser.add_argument('--calibration-begin', type=float, default=0.90)
    parser.add_argument('--calibration-end', type=float, default=0.95)
    parser.add_argument('--test-end', type=float, default=1.0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--cache-size', type=int, default=256)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='debug only: limit both passes; 0 uses complete splits')
    parser.add_argument('--output', default='results/risk_calibration.json')
    parser.add_argument('--calibrator-output', default='results/risk_calibrator.json')
    args = parser.parse_args()
    validate_splits(args, parser)

    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(device_name)
    checkpoint = os.path.abspath(args.checkpoint)
    data_dir = os.path.abspath(args.data_dir)
    output = os.path.abspath(args.output)
    calibrator_output = os.path.abspath(args.calibrator_output)
    if not os.path.isfile(os.path.join(data_dir, 'count.json')):
        raise FileNotFoundError('count.json not found in %s' % data_dir)

    # MahjongGBDataset uses data/... relative to the data directory's parent.
    os.chdir(os.path.dirname(data_dir))
    fit_dataset, fit_loader = make_loader(
        args.calibration_begin, args.calibration_end, args, device)
    test_dataset, test_loader = make_loader(
        args.calibration_end, args.test_end, args, device)
    model = load_model(checkpoint, device)
    calibrators, fit_samples = fit_calibrators(model, fit_loader, args, device)
    test_result = evaluate_calibration(model, test_loader, calibrators, args, device)

    checkpoint_hash = file_sha256(checkpoint)
    calibrator_payload = {
        'schema_version': 1,
        'method': 'platt_scaling_positive_slope',
        'checkpoint': checkpoint,
        'checkpoint_sha256': checkpoint_hash,
        'fit_split': [args.calibration_begin, args.calibration_end],
        'parameters': calibrators,
    }
    result = {
        'checkpoint': checkpoint,
        'checkpoint_sha256': checkpoint_hash,
        'split': {
            'model_training': [0.0, args.calibration_begin],
            'calibrator_fit': [args.calibration_begin, args.calibration_end],
            'calibrator_test': [args.calibration_end, args.test_end],
        },
        'fit_dataset_samples': len(fit_dataset),
        'fit_processed_samples': fit_samples,
        'test_dataset_samples': len(test_dataset),
        'test': test_result,
        'calibrators': calibrators,
        'metric_note': (
            'AUROC/AUPRC use 1000-bin streaming approximations. '
            'Platt parameters are fit only on the calibration split.'),
    }
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(calibrator_output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(calibrator_output, 'w', encoding='utf-8') as f:
        json.dump(calibrator_payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('[calibrator] wrote %s' % calibrator_output, flush=True)


if __name__ == '__main__':
    main()
