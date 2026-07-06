'''Four-bot duplicate evaluation with all 24 seat permutations.'''

import argparse
import csv
import itertools
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

import evaluate as E


def parse_bot(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('expected NAME=CHECKPOINT')
    return tuple(value.split('=', 1))


def run_lineup(lineup, models, wall_seed, permutation_id, args):
    env = E.MahjongGBEnv({
        'agent_clz': E.FeatureAgent,
        'duplicate': True,
        'reward_gang': 0.0,
        'reward_peng': 0.0,
        'reward_chi': 0.0,
        'reward_tenpai': 0.0,
        'reward_notenpai': 0.0,
        'track_terminal_stats': True,
    })
    prevalent_wind = random.Random(wall_seed ^ 0x5A17).randrange(4)
    obs = env.reset(prevalentWind=prevalent_wind, tileWall=E.make_wall(wall_seed))
    rewards = [0.0] * 4
    latency = [[] for _ in range(4)]
    counts = [defaultdict(int) for _ in range(4)]
    steps = 0
    while not env.done and steps < args.max_steps:
        actions = {}
        for player_name, player_obs in obs.items():
            seat = env.agent_names.index(player_name)
            bot_name = lineup[seat]
            action, elapsed = E.select_action(
                models[bot_name], env.agents[seat], player_obs,
                args.postprocess, bot_name in args.risk_bot,
                bot_name in args.aux_rank_bot)
            actions[player_name] = action
            latency[seat].append(elapsed)
            response = env.agents[seat].action2response(action).split()[0]
            counts[seat]['Gang' if response in ('Gang', 'BuGang') else response] += 1
        obs, step_rewards, _ = env.step(actions)
        for player_name, reward in step_rewards.items():
            rewards[env.agent_names.index(player_name)] += float(reward)
        steps += 1

    result = env.last_result or {}
    winner = result.get('winner')
    discarder = result.get('discarder')
    fan = int(result.get('fan', 0) or 0)
    tenpai = result.get('tenpai')
    invalid = bool(result.get('invalid')) or not env.done
    rows = []
    for seat, bot_name in enumerate(lineup):
        decisions = max(1, sum(counts[seat].values()))
        rows.append({
            'model': bot_name,
            'wall_seed': wall_seed,
            'permutation': permutation_id,
            'lineup': '|'.join(lineup),
            'seat': seat,
            'prevalent_wind': prevalent_wind,
            'score': rewards[seat],
            'rank': E._rank(rewards, seat),
            'win': int(winner == seat),
            'win_fan': fan if winner == seat else 0,
            'self_draw': int(winner == seat and result.get('self_draw', False)),
            'deal_in': int(discarder == seat),
            'deal_in_fan': fan if discarder == seat else 0,
            'big_deal_in': int(discarder == seat and fan >= 32),
            'draw': int(bool(result.get('draw'))),
            'draw_tenpai': int(bool(tenpai and tenpai[seat])),
            'call_rate': (counts[seat]['Chi'] + counts[seat]['Peng'] + counts[seat]['Gang']) / decisions,
            'decisions': len(latency[seat]),
            'mean_decision_ms': float(np.mean(latency[seat])) if latency[seat] else 0.0,
            'p95_decision_ms': float(np.percentile(latency[seat], 95)) if latency[seat] else 0.0,
            'max_decision_ms': max(latency[seat], default=0.0),
            'invalid_or_timeout': int(invalid),
            'steps': steps,
        })
    return rows


def summarise(rows):
    output = []
    for name in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == name]
        wins = [r for r in rs if r['win']]
        deals = [r for r in rs if r['deal_in']]
        draws = [r for r in rs if r['draw']]
        output.append({
            'model': name,
            'games': len(rs),
            'avg_score': float(np.mean([r['score'] for r in rs])),
            'avg_rank': float(np.mean([r['rank'] for r in rs])),
            'win_rate': float(np.mean([r['win'] for r in rs])),
            'avg_win_fan': float(np.mean([r['win_fan'] for r in wins])) if wins else 0.0,
            'deal_in_rate': float(np.mean([r['deal_in'] for r in rs])),
            'avg_deal_in_fan': float(np.mean([r['deal_in_fan'] for r in deals])) if deals else 0.0,
            'big_deal_in_rate': float(np.mean([r['big_deal_in'] for r in rs])),
            'draw_tenpai_rate': float(np.mean([r['draw_tenpai'] for r in draws])) if draws else 0.0,
            'call_rate': float(np.mean([r['call_rate'] for r in rs])),
            'mean_decision_ms': float(np.mean([r['mean_decision_ms'] for r in rs])),
            'invalid_or_timeout_rate': float(np.mean([r['invalid_or_timeout'] for r in rs])),
        })
    return output


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='24-permutation four-bot duplicate evaluation')
    parser.add_argument('--bot', action='append', type=parse_bot, required=True,
                        metavar='NAME=CHECKPOINT', help='exactly four unique bots')
    parser.add_argument('--risk-bot', action='append', default=[])
    parser.add_argument('--aux-rank-bot', action='append', default=[])
    parser.add_argument('--postprocess', choices=('none', 'light'), default='light')
    parser.add_argument('--walls', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260706)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--progress-every', type=int, default=24)
    parser.add_argument('--allow-partial-checkpoint', action='store_true')
    parser.add_argument('--output', default='results/league')
    args = parser.parse_args()
    if len(args.bot) != 4 or len({name for name, _ in args.bot}) != 4:
        parser.error('--bot must be supplied exactly four times with unique names')

    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if args.device == 'auto' else args.device)
    device = torch.device(device_name)
    paths = dict(args.bot)
    models = {name: E.load_model(path, device, args.allow_partial_checkpoint)
              for name, path in args.bot}
    names = tuple(paths)
    permutations = list(itertools.permutations(names))
    rows = []
    total = args.walls * 24
    games = 0
    for offset in range(args.walls):
        wall_seed = args.seed + offset
        for permutation_id, lineup in enumerate(permutations):
            rows.extend(run_lineup(lineup, models, wall_seed, permutation_id, args))
            games += 1
            if args.progress_every and games % args.progress_every == 0:
                print('[league] %d/%d games' % (games, total), flush=True)

    summary = summarise(rows)
    write_csv(args.output + '_games.csv', rows)
    write_csv(args.output + '_summary.csv', summary)
    manifest = {
        'seed': args.seed,
        'walls': args.walls,
        'games_per_wall': 24,
        'postprocess': args.postprocess,
        'risk_bot': args.risk_bot,
        'aux_rank_bot': args.aux_rank_bot,
        'bots': [{'name': name, 'path': path, 'sha256': E.file_sha256(path)}
                 for name, path in args.bot],
        'action_value_weight': E.BOT.ACTION_VALUE_WEIGHT,
        'fan_route_weight': E.BOT.FAN_ROUTE_WEIGHT,
        'use_expected_loss_head': E.BOT.USE_EXPECTED_LOSS_HEAD,
    }
    with open(args.output + '_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
