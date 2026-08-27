'''Reproducible offline evaluation for the submitted SL bot.

Each challenger plays the same walls in all four seats against one frozen
opponent model.  The evaluator can compare raw argmax, light control, and
risk-aware control in one run, writes per-game and aggregate CSV files, and
reports wall-clustered paired bootstrap confidence intervals relative to the
first challenger.

Example:
    python evaluate.py --opponent model/baseline.pkl \
      --challenger baseline=model/baseline.pkl \
      --challenger temporal=model/temporal.pkl \
      --risk-challenger temporal --walls 2000 --seed 20260702 \
      --output results/ablation
'''

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.environ.get('MAHJONG_RL_DIR', os.path.abspath(
    os.path.join(HERE, '..', '..', '..', '..', 'src', 'RL')))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if RL_DIR not in sys.path:
    sys.path.append(RL_DIR)

from feature import FeatureAgent  # noqa: E402
from model import CNNModel  # noqa: E402
from env import MahjongGBEnv  # noqa: E402


def _load_bot_runtime():
    spec = importlib.util.spec_from_file_location('sl_bot_runtime', os.path.join(HERE, '__main__.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOT = _load_bot_runtime()


def _tensor_from_buffer(value, numpy_dtype, torch_dtype, device):
    """Convert through Python's buffer protocol, avoiding torch's NumPy C API."""
    array = np.array(value, dtype=numpy_dtype, order='C', copy=True)
    tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(array.shape)
    return tensor.unsqueeze(0).to(device)


def load_model(path, device, allow_partial=False):
    model = CNNModel()
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    current = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in current and current[k].shape == v.shape}
    missing = sorted(set(current) - set(compatible))
    unexpected = sorted(set(state) - set(compatible))
    optional_new_heads = ('_action_value_head.', '_risk_loss_opp_head.')
    required_missing = [k for k in missing if not k.startswith(optional_new_heads)]
    if (required_missing or unexpected) and not allow_partial:
        raise RuntimeError(
            'Checkpoint architecture mismatch for %s: %d missing and %d unexpected tensors. '
            'Evaluate it with its matching historical model.py/feature.py; partial loading would '
            'produce random layers and invalid results. Use --allow-partial-checkpoint only for debugging.' %
            (path, len(required_missing), len(unexpected)))
    current.update(compatible)
    model.load_state_dict(current)
    model.to(device).eval()
    print('[load] %s: %d/%d tensors' % (path, len(compatible), len(current)), flush=True)
    return model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_wall(seed):
    tiles = []
    for _ in range(4):
        tiles.extend('W%d' % i for i in range(1, 10))
        tiles.extend('T%d' % i for i in range(1, 10))
        tiles.extend('B%d' % i for i in range(1, 10))
        tiles.extend('F%d' % i for i in range(1, 5))
        tiles.extend('J%d' % i for i in range(1, 4))
    random.Random(seed).shuffle(tiles)
    return ' '.join(tiles)


def select_action(model, agent, obs, postprocess, use_risk_head, use_aux_rank,
                  risk_calibration=None, risk_head_mode='opponent',
                  zero_sequence=False):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    seq_tile, seq_player = BOT._discard_sequence_tensors(agent)
    if zero_sequence:
        seq_tile = np.full_like(seq_tile, 34)
        seq_player = np.zeros_like(seq_player)
    # The SL and RL directories both contain top-level modules.  In some
    # environments their extension imports can leave observations backed by a
    # second NumPy module instance.  torch.from_numpy then rejects the object
    # with the confusing message "expected np.ndarray (got numpy.ndarray)".
    # Convert through Python's buffer protocol so no ndarray identity check is
    # involved at the PyTorch boundary.
    input_dict = {
        'observation': _tensor_from_buffer(
            obs['observation'], np.float32, torch.float32, device),
        'action_mask': _tensor_from_buffer(
            obs['action_mask'], np.float32, torch.float32, device),
        'discard_seq': _tensor_from_buffer(
            seq_tile, np.int64, torch.int64, device),
        'discard_player': _tensor_from_buffer(
            seq_player, np.int64, torch.int64, device),
    }
    with torch.no_grad():
        if use_risk_head or use_aux_rank:
            logits, aux = model(input_dict, return_aux=True)
            # Keep auxiliary outputs independent of NumPy module identity.
            # The postprocessor only indexes scalar values; Python lists are
            # sufficient, and np.asarray inside the risk path will then create
            # arrays owned by the runtime's NumPy instance.
            aux = {k: v.detach().cpu().reshape(-1).tolist()
                   for k, v in aux.items()}
        else:
            logits = model(input_dict)
            aux = None
    # Do not mix the ndarray instance created by torch.Tensor.numpy() with the
    # one imported by the bot runtime.  Compiled Mahjong extensions can cause
    # both NumPy module identities to coexist in one process.
    logits = np.asarray(
        logits.detach().cpu().reshape(-1).tolist(), dtype=np.float32)
    old_mode, old_risk, old_rank = BOT.POSTPROCESS_MODE, BOT.USE_RISK_HEAD, BOT.USE_AUX_RANK
    old_risk_head_mode = BOT.RISK_HEAD_MODE
    old_calibration_enabled = BOT.USE_RISK_CALIBRATION
    old_calibration = BOT.RISK_CALIBRATION
    try:
        BOT.POSTPROCESS_MODE = postprocess
        BOT.USE_RISK_HEAD = use_risk_head
        BOT.RISK_HEAD_MODE = risk_head_mode
        BOT.USE_AUX_RANK = use_aux_rank
        BOT.set_risk_calibration(risk_calibration)
        action = BOT._postprocess_action(agent, logits, obs['action_mask'], aux)
    finally:
        BOT.POSTPROCESS_MODE, BOT.USE_RISK_HEAD, BOT.USE_AUX_RANK = old_mode, old_risk, old_rank
        BOT.RISK_HEAD_MODE = old_risk_head_mode
        BOT.USE_RISK_CALIBRATION = old_calibration_enabled
        BOT.RISK_CALIBRATION = old_calibration
    legal_mask = obs['action_mask'].astype(bool)
    if action < 0 or action >= len(legal_mask) or not legal_mask[action]:
        legal_logits = logits.copy()
        legal_logits[~legal_mask] = -np.inf
        action = int(legal_logits.argmax())
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return int(action), (time.perf_counter() - start) * 1000.0


def _rank(values, seat):
    value = values[seat]
    return 1.0 + sum(v > value for v in values) + 0.5 * (sum(v == value for v in values) - 1)


def run_game(challenger_name, challenger, opponent, wall_seed, seat, args):
    env = MahjongGBEnv({
        'agent_clz': FeatureAgent,
        'duplicate': True,
        'reward_gang': 0.0,
        'reward_peng': 0.0,
        'reward_chi': 0.0,
        'reward_tenpai': 0.0,
        'reward_notenpai': 0.0,
        'track_terminal_stats': True,
    })
    prevalent_wind = random.Random(wall_seed ^ 0x5A17).randrange(4)
    obs = env.reset(prevalentWind=prevalent_wind, tileWall=make_wall(wall_seed))
    rewards = [0.0] * 4
    decision_ms = []
    action_counts = defaultdict(int)
    steps = 0
    while not env.done and steps < args.max_steps:
        actions = {}
        for name, player_obs in obs.items():
            player = env.agent_names.index(name)
            model = challenger if player == seat else opponent
            use_risk = (challenger_name in args.risk_challenger
                        if player == seat else args.opponent_use_risk_head)
            use_rank = (challenger_name in args.aux_rank_challenger
                        if player == seat else args.opponent_use_aux_rank)
            risk_head_mode = ('aggregate'
                              if player == seat and challenger_name in args.aggregate_risk_challenger
                              else 'opponent')
            zero_sequence = (player == seat and
                             challenger_name in args.zero_sequence_challenger)
            postprocess = (('none' if challenger_name in args.raw_challenger
                            else args.postprocess)
                           if player == seat else args.opponent_postprocess)
            risk_calibration = (args.risk_calibrations.get(challenger_name)
                                if player == seat else args.opponent_risk_calibration_config)
            action, elapsed = select_action(
                model, env.agents[player], player_obs, postprocess,
                use_risk, use_rank, risk_calibration,
                risk_head_mode=risk_head_mode,
                zero_sequence=zero_sequence)
            actions[name] = action
            if player == seat:
                decision_ms.append(elapsed)
                response = env.agents[player].action2response(action).split()[0]
                action_counts['Gang' if response in ('Gang', 'BuGang') else response] += 1
        obs, step_rewards, done = env.step(actions)
        for name, value in step_rewards.items():
            rewards[env.agent_names.index(name)] += float(value)
        steps += 1

    result = env.last_result or {}
    invalid = bool(result.get('invalid')) or not env.done
    winner = result.get('winner')
    discarder = result.get('discarder')
    fan = int(result.get('fan', 0) or 0)
    tenpai = result.get('tenpai')
    decisions = max(1, sum(action_counts.values()))
    return {
        'model': challenger_name,
        'wall_seed': wall_seed,
        'seat': seat,
        'prevalent_wind': prevalent_wind,
        'score': rewards[seat],
        'rank': _rank(rewards, seat),
        'win': int(winner == seat),
        'self_draw': int(winner == seat and result.get('self_draw', False)),
        'deal_in': int(discarder == seat),
        'deal_in_fan': fan if discarder == seat else 0,
        'big_deal_in': int(discarder == seat and fan >= 32),
        'draw': int(bool(result.get('draw'))),
        'draw_tenpai': int(bool(tenpai and tenpai[seat])),
        'call_rate': (action_counts['Chi'] + action_counts['Peng'] + action_counts['Gang']) / decisions,
        'decisions': len(decision_ms),
        'mean_decision_ms': float(np.mean(decision_ms)) if decision_ms else 0.0,
        'p95_decision_ms': float(np.percentile(decision_ms, 95)) if decision_ms else 0.0,
        'max_decision_ms': max(decision_ms, default=0.0),
        'invalid_or_timeout': int(invalid),
        'steps': steps,
    }


def summarise(rows):
    result = []
    for name in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == name]
        deal_ins = [r for r in rs if r['deal_in']]
        draws = [r for r in rs if r['draw']]
        result.append({
            'model': name,
            'games': len(rs),
            'avg_score': float(np.mean([r['score'] for r in rs])),
            'avg_rank': float(np.mean([r['rank'] for r in rs])),
            'win_rate': float(np.mean([r['win'] for r in rs])),
            'self_draw_rate': float(np.mean([r['self_draw'] for r in rs])),
            'deal_in_rate': float(np.mean([r['deal_in'] for r in rs])),
            'avg_deal_in_fan': float(np.mean([r['deal_in_fan'] for r in deal_ins])) if deal_ins else 0.0,
            'big_deal_in_rate': float(np.mean([r['big_deal_in'] for r in rs])),
            'draw_tenpai_rate': float(np.mean([r['draw_tenpai'] for r in draws])) if draws else 0.0,
            'call_rate': float(np.mean([r['call_rate'] for r in rs])),
            'mean_decision_ms': float(np.mean([r['mean_decision_ms'] for r in rs])),
            'p95_game_decision_ms': float(np.percentile([r['p95_decision_ms'] for r in rs], 95)),
            'max_decision_ms': float(max(r['max_decision_ms'] for r in rs)),
            'invalid_or_timeout_rate': float(np.mean([r['invalid_or_timeout'] for r in rs])),
        })
    return result


def paired_bootstrap(rows, baseline, samples, seed, include_invalid_walls=False):
    by_model = defaultdict(dict)
    for row in rows:
        by_model[row['model']][(row['wall_seed'], row['seat'])] = row
    base = by_model[baseline]
    rng = np.random.default_rng(seed)
    output = {}
    for name, values in by_model.items():
        if name == baseline:
            continue
        candidate_walls = sorted({wall for wall, _ in set(base) & set(values)})
        wall_deltas = []
        excluded_walls = 0
        for wall in candidate_walls:
            keys = [(wall, seat) for seat in range(4)]
            if any(key not in base or key not in values for key in keys):
                excluded_walls += 1
                continue
            if (not include_invalid_walls and
                    any(base[key]['invalid_or_timeout'] or
                        values[key]['invalid_or_timeout'] for key in keys)):
                excluded_walls += 1
                continue
            wall_deltas.append(np.mean([
                float(values[key]['score']) - float(base[key]['score'])
                for key in keys
            ]))
        delta = np.asarray(wall_deltas, dtype=np.float64)
        if not len(delta):
            continue
        means = np.empty(samples, dtype=np.float64)
        for i in range(samples):
            means[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
        output[name] = {
            'baseline': baseline,
            'paired_walls': len(delta),
            'paired_games': 4 * len(delta),
            'excluded_walls': excluded_walls,
            'mean_score_delta': float(delta.mean()),
            'win_pair_fraction': float((delta > 0).mean()),
            'ci95_low': float(np.percentile(means, 2.5)),
            'ci95_high': float(np.percentile(means, 97.5)),
        }
    return output


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_challenger(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('expected NAME=CHECKPOINT')
    name, path = value.split('=', 1)
    return name, path


def load_risk_calibrator(path, checkpoint_path):
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    expected_hash = file_sha256(checkpoint_path)
    calibration_hash = payload.get('checkpoint_sha256')
    if calibration_hash and calibration_hash != expected_hash:
        raise ValueError(
            'calibrator %s was fit for checkpoint %s, not %s' %
            (path, calibration_hash, expected_hash))
    parameters = BOT._normalise_risk_calibration(payload)
    manifest = {
        'path': path,
        'sha256': file_sha256(path),
        'checkpoint_sha256': calibration_hash,
        'method': payload.get('method', 'unknown'),
        'fit_split': payload.get('fit_split'),
        'parameters': parameters,
    }
    return parameters, manifest


def main():
    parser = argparse.ArgumentParser(description='Reproducible SL bot evaluation')
    parser.add_argument('--opponent', required=True, help='Frozen opponent checkpoint')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--allow-partial-checkpoint', action='store_true',
                        help='debug only; partial loads are not valid model comparisons')
    parser.add_argument('--challenger', action='append', type=parse_challenger, required=True,
                        metavar='NAME=CHECKPOINT', help='Repeat for every model to compare')
    parser.add_argument('--walls', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260702)
    parser.add_argument('--output', default='results/sl_evaluation')
    parser.add_argument('--postprocess', choices=('none', 'light'), default='light')
    parser.add_argument('--opponent-postprocess', choices=('none', 'light'), default='light',
                        help='Keep the frozen opponent policy identical across ablations')
    parser.add_argument('--raw-challenger', action='append', default=[], metavar='NAME',
                        help='Use masked neural argmax for this challenger; repeat as needed')
    parser.add_argument('--risk-challenger', action='append', default=[], metavar='NAME',
                        help='Enable learned risk heads for this challenger; repeat as needed')
    parser.add_argument('--aggregate-risk-challenger', action='append', default=[], metavar='NAME',
                        help='Use aggregate rather than opponent-specific learned risk')
    parser.add_argument('--zero-sequence-challenger', action='append', default=[], metavar='NAME',
                        help='Replace the ordered discard sequence with padding (stress test)')
    parser.add_argument('--risk-calibration', action='append', type=parse_challenger,
                        default=[], metavar='NAME=JSON',
                        help='Apply a checkpoint-matched risk calibrator to this challenger')
    parser.add_argument('--aux-rank-challenger', action='append', default=[], metavar='NAME')
    parser.add_argument('--opponent-use-risk-head', action='store_true')
    parser.add_argument('--opponent-risk-calibration', metavar='JSON',
                        help='Checkpoint-matched calibrator for the frozen opponent')
    parser.add_argument('--opponent-use-aux-rank', action='store_true')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--bootstrap-samples', type=int, default=5000)
    parser.add_argument('--include-invalid-walls', action='store_true',
                        help='Include walls with an invalid action/timeout in paired statistics')
    parser.add_argument('--progress-every', type=int, default=100)
    args = parser.parse_args()

    challenger_names = {name for name, _ in args.challenger}
    selected_names = set(args.raw_challenger + args.risk_challenger +
                         args.aggregate_risk_challenger +
                         args.zero_sequence_challenger + args.aux_rank_challenger)
    unknown = selected_names - challenger_names
    if unknown:
        parser.error('configuration names are not challengers: %s' % ', '.join(sorted(unknown)))
    overlap = set(args.raw_challenger) & set(args.risk_challenger)
    if overlap:
        parser.error('raw challengers cannot also enable risk control: %s' %
                     ', '.join(sorted(overlap)))
    aggregate_without_risk = set(args.aggregate_risk_challenger) - set(args.risk_challenger)
    if aggregate_without_risk:
        parser.error('aggregate-risk challengers must also enable risk control: %s' %
                     ', '.join(sorted(aggregate_without_risk)))
    calibration_names = [name for name, _ in args.risk_calibration]
    if len(calibration_names) != len(set(calibration_names)):
        parser.error('duplicate --risk-calibration name')
    unknown_calibration = set(calibration_names) - set(args.risk_challenger)
    if unknown_calibration:
        parser.error('risk calibration requires --risk-challenger for: %s' %
                     ', '.join(sorted(unknown_calibration)))
    if args.opponent_risk_calibration and not args.opponent_use_risk_head:
        parser.error('--opponent-risk-calibration requires --opponent-use-risk-head')

    challenger_paths = dict(args.challenger)
    args.risk_calibrations = {}
    args.risk_calibration_manifest = {}
    try:
        for name, path in args.risk_calibration:
            parameters, metadata = load_risk_calibrator(path, challenger_paths[name])
            args.risk_calibrations[name] = parameters
            args.risk_calibration_manifest[name] = metadata
        if args.opponent_risk_calibration:
            (args.opponent_risk_calibration_config,
             args.opponent_risk_calibration_manifest) = load_risk_calibrator(
                 args.opponent_risk_calibration, args.opponent)
        else:
            args.opponent_risk_calibration_config = None
            args.opponent_risk_calibration_manifest = None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if args.device == 'auto' else args.device)
    device = torch.device(device_name)
    opponent = load_model(args.opponent, device, args.allow_partial_checkpoint)
    challengers = [(name, load_model(path, device, args.allow_partial_checkpoint))
                   for name, path in args.challenger]
    wall_seeds = [args.seed + i for i in range(args.walls)]
    rows = []
    total = len(challengers) * len(wall_seeds) * 4
    for name, model in challengers:
        for wall_seed in wall_seeds:
            for seat in range(4):
                rows.append(run_game(name, model, opponent, wall_seed, seat, args))
                if args.progress_every and len(rows) % args.progress_every == 0:
                    print('[evaluate] %d/%d games' % (len(rows), total), flush=True)

    summary = summarise(rows)
    paired = paired_bootstrap(
        rows, challengers[0][0], args.bootstrap_samples, args.seed,
        include_invalid_walls=args.include_invalid_walls)
    write_csv(args.output + '_games.csv', rows)
    write_csv(args.output + '_summary.csv', summary)
    with open(args.output + '_paired.json', 'w', encoding='utf-8') as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)
    manifest = {
        'seed': args.seed,
        'device': str(device),
        'walls': args.walls,
        'seats_per_wall': 4,
        'postprocess': args.postprocess,
        'opponent_postprocess': args.opponent_postprocess,
        'raw_challenger': args.raw_challenger,
        'tenpai_learned_weight': BOT.TENPAI_LEARNED_WEIGHT,
        'normalize_discard_logits': BOT.NORMALIZE_DISCARD_LOGITS,
        'fan_wait_weight': BOT.FAN_WAIT_WEIGHT,
        'action_value_weight': BOT.ACTION_VALUE_WEIGHT,
        'fan_route_weight': BOT.FAN_ROUTE_WEIGHT,
        'use_expected_loss_head': BOT.USE_EXPECTED_LOSS_HEAD,
        'risk_challenger': args.risk_challenger,
        'aggregate_risk_challenger': args.aggregate_risk_challenger,
        'zero_sequence_challenger': args.zero_sequence_challenger,
        'risk_calibration': args.risk_calibration_manifest,
        'aux_rank_challenger': args.aux_rank_challenger,
        'opponent_use_risk_head': args.opponent_use_risk_head,
        'opponent_risk_calibration': args.opponent_risk_calibration_manifest,
        'opponent_use_aux_rank': args.opponent_use_aux_rank,
        'max_steps': args.max_steps,
        'bootstrap_samples': args.bootstrap_samples,
        'bootstrap_cluster': 'wall',
        'include_invalid_walls': args.include_invalid_walls,
        'opponent': {'path': args.opponent, 'sha256': file_sha256(args.opponent)},
        'challengers': [{'name': name, 'path': path, 'sha256': file_sha256(path)}
                        for name, path in args.challenger],
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    with open(args.output + '_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({'summary': summary, 'paired': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
