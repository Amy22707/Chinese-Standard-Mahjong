# Agent part
from feature import FeatureAgent

# Model part
from model import CNNModel

# Botzone interaction
import numpy as np
import torch
import os
import sys

# Final configuration selected by the fixed-wall paired evaluation:
# value_off (best18) + learned risk, with all unproven new heads disabled.
USE_AUX_RANK = False
USE_RISK_HEAD = True
POSTPROCESS_MODE = 'light'
TENPAI_LEARNED_WEIGHT = 1.0
NORMALIZE_DISCARD_LOGITS = False
FAN_WAIT_WEIGHT = 0.0
ACTION_VALUE_WEIGHT = 0.0
FAN_ROUTE_WEIGHT = 0.0
USE_EXPECTED_LOSS_HEAD = False
MAX_SEQ_LEN = 80


def _discard_danger(agent, tile, aux = None):
    '''Combined multi-opponent danger; genbutsu is safe only versus its owner.'''
    idx = FeatureAgent.OFFSET_TILE.get(tile, -1)
    per_opp = [agent._estimate_discard_danger(p, tile) for p in range(1, 4)]
    aggregate_learned = None
    tenpai_prob = []
    total_discards = len(getattr(agent, 'discardEvents', []))
    for p in range(1, 4):
        melds = len(agent.packs[p])
        river = len(agent.history[p])
        since_meld = sum(1 for ep, _, _, after in getattr(agent, 'discardEvents', [])
                         if ep == p and after)
        proxy = 0.08 + 0.012 * total_discards + 0.10 * melds
        if since_meld >= 2:
            proxy += 0.08
        if river >= 10:
            proxy += 0.08
        tenpai_prob.append(min(0.85, proxy))
    if USE_RISK_HEAD and aux is not None and idx < 34:
        if 'tenpai_opp' in aux and np.asarray(aux['tenpai_opp']).size == 3:
            learned_tenpai = 1.0 / (1.0 + np.exp(-np.asarray(aux['tenpai_opp']).reshape(3)))
            rule_tenpai = np.asarray(tenpai_prob)
            tenpai_prob = ((1.0 - TENPAI_LEARNED_WEIGHT) * rule_tenpai
                           + TENPAI_LEARNED_WEIGHT * learned_tenpai).tolist()
        if 'risk_opp' in aux and np.asarray(aux['risk_opp']).size == 102:
            learned = 1.0 / (1.0 + np.exp(-np.asarray(aux['risk_opp']).reshape(3, 34)[:, idx]))
            per_opp = [max(per_opp[p], float(learned[p])) for p in range(3)]
        elif 'risk' in aux:
            aggregate_learned = float(1.0 / (1.0 + np.exp(-aux['risk'][idx])))
    per_opp = [p * (0.55 + 0.75 * tp) for p, tp in zip(per_opp, tenpai_prob)]
    probability = 1.0 - float(np.prod([1.0 - min(0.99, p) for p in per_opp]))
    if aggregate_learned is not None:
        probability = max(probability, aggregate_learned)
    if USE_RISK_HEAD and aux is not None and 'risk_loss' in aux and idx < 34:
        severity = float(1.0 / (1.0 + np.exp(-aux['risk_loss'][idx])))
        probability *= 0.65 + 0.70 * severity
    if (USE_EXPECTED_LOSS_HEAD and aux is not None
            and 'risk_opp' in aux and 'risk_loss_opp' in aux
            and np.asarray(aux['risk_opp']).size == 102
            and np.asarray(aux['risk_loss_opp']).size == 102):
        hit = 1.0 / (1.0 + np.exp(-np.asarray(aux['risk_opp']).reshape(3, 34)[:, idx]))
        loss = 1.0 / (1.0 + np.exp(-np.asarray(aux['risk_loss_opp']).reshape(3, 34)[:, idx]))
        # Minimum hands still cost points; severity progressively represents
        # the long tail of large MCR hands.
        expected_loss = float(np.sum(hit * (0.25 + 0.75 * loss)))
        probability = max(probability, min(1.0, expected_loss))
    return min(1.0, probability)


def _discard_sequence_tensors(agent):
    # FeatureAgent records the real table-wide order.  Reconstructing from four
    # independent rivers loses interleaving and feeds the GRU a false timeline.
    events = [(order, rel_player, FeatureAgent.OFFSET_TILE[tile])
              for rel_player, tile, order, _ in getattr(agent, 'discardEvents', [])
              if tile in FeatureAgent.OFFSET_TILE]
    events = events[-MAX_SEQ_LEN:]
    tile_arr = np.full(MAX_SEQ_LEN, 34, dtype = np.int64)
    player_arr = np.zeros(MAX_SEQ_LEN, dtype = np.int64)
    start = MAX_SEQ_LEN - len(events)
    for j, (_, rel_player, tile_id) in enumerate(events):
        tile_arr[start + j] = tile_id
        player_arr[start + j] = rel_player
    return tile_arr, player_arr


def _attack_context(agent):
    ctx = {
        'shanten': 3,
        'chiitoi_shanten': 6,
        'best_flush_shanten': 13,
        'high_value': False,
        'pair_tiles': set(),
        'flush_suit': None,
    }
    if not hasattr(agent, 'hand') or not hasattr(agent, 'packs'):
        return ctx
    try:
        from MahjongGB import MahjongShanten
        from collections import Counter

        shanten = MahjongShanten(hand = tuple(agent.hand), pack = tuple(agent.packs[0]))
        hand_cnt = Counter(agent.hand)
        pairs = sum(1 for c in hand_cnt.values() if c >= 2)
        chiitoi_shanten = max(0, 6 - pairs)

        best_flush_shanten = 13
        best_flush_suit = None
        for suit in 'WTB':
            suit_tiles = [t for t in agent.hand if t[0] == suit]
            for pack_type, tile, _ in agent.packs[0]:
                if tile[0] == suit:
                    if pack_type == 'CHI':
                        num = int(tile[1])
                        suit_tiles.extend([suit + str(num - 1), tile, suit + str(num + 1)])
                    elif pack_type == 'PENG':
                        suit_tiles.extend([tile] * 3)
                    elif pack_type == 'GANG':
                        suit_tiles.extend([tile] * 4)
            suit_shanten = max(0, 13 - len(suit_tiles))
            if suit_shanten < best_flush_shanten:
                best_flush_shanten = suit_shanten
                best_flush_suit = suit

        ctx.update({
            'shanten': max(0, shanten),
            'chiitoi_shanten': chiitoi_shanten,
            'best_flush_shanten': best_flush_shanten,
            'high_value': chiitoi_shanten <= 2 or best_flush_shanten <= 4,
            'pair_tiles': {tile for tile, cnt in hand_cnt.items() if cnt >= 2},
            'flush_suit': best_flush_suit,
        })
    except Exception:
        pass
    return ctx


def _threat_context(agent):
    '''Estimate how much the table asks us to fold.

    Uses only public information: wall progress, opponents' open melds, and discard count.
    '''
    try:
        total_discards = sum(len(h) for h in agent.history)
        my_wall = agent.tileWall[0] if hasattr(agent, 'tileWall') else 21
        late_hand = total_discards >= 36 or my_wall <= 10
        very_late = total_discards >= 52 or my_wall <= 5
        max_opp_melds = max((len(agent.packs[p]) for p in range(1, 4)), default = 0)
        route_threat = 0.0
        for p in range(1, 4):
            packs = [x for x in agent.packs[p] if x[1] != 'CONCEALED']
            suit_counts = {s: sum(x[1][0] == s for x in packs) for s in 'WTB'}
            honours = sum(x[1][0] in 'FJ' for x in packs)
            if max(suit_counts.values(), default = 0) >= 2:
                route_threat = max(route_threat, 0.35)  # flush / half-flush warning
            if honours >= 2:
                route_threat = max(route_threat, 0.40)  # honour-heavy high-value warning
            if len(packs) >= 3:
                route_threat = max(route_threat, 0.45)
        many_open_melds = max_opp_melds >= 2
        threat = 0.0
        if late_hand:
            threat += 0.35
        if very_late:
            threat += 0.30
        if many_open_melds:
            threat += 0.25
        elif max_opp_melds >= 1 and late_hand:
            threat += 0.12
        threat += route_threat
        return {
            'late': late_hand,
            'very_late': very_late,
            'open_threat': many_open_melds,
            'level': min(1.0, threat),
        }
    except Exception:
        return {'late': False, 'very_late': False, 'open_threat': False, 'level': 0.0}


def _push_level(shanten, high_value, threat_level):
    if shanten <= 0:
        return 1.0
    if shanten == 1 and high_value:
        return 0.80 - 0.10 * threat_level
    if shanten == 1:
        return 0.58 - 0.30 * threat_level
    if high_value:
        return 0.48 - 0.25 * threat_level
    return 0.28 - 0.28 * threat_level


def _postprocess_action(agent, logits, mask, aux = None):
    legal_mask = mask.astype(bool)
    if not legal_mask.any():
        return FeatureAgent.OFFSET_ACT['Pass']

    # Winning is always preferred once it is legal; the 8-fan check is already in FeatureAgent.
    if legal_mask[FeatureAgent.OFFSET_ACT['Hu']]:
        return FeatureAgent.OFFSET_ACT['Hu']
    if POSTPROCESS_MODE == 'none':
        legal_logits = logits.copy()
        legal_logits[~legal_mask] = -np.inf
        return int(legal_logits.argmax())

    adjusted = logits.copy()
    raw = logits.copy()
    play_begin = FeatureAgent.OFFSET_ACT['Play']
    chi_begin = FeatureAgent.OFFSET_ACT['Chi']
    peng_begin = FeatureAgent.OFFSET_ACT['Peng']
    gang_begin = FeatureAgent.OFFSET_ACT['Gang']
    angang_begin = FeatureAgent.OFFSET_ACT['AnGang']
    bugang_begin = FeatureAgent.OFFSET_ACT['BuGang']

    ctx = _attack_context(agent)
    threat = _threat_context(agent)
    shanten = ctx['shanten']
    shanten_factor = min(1.0, max(0, shanten) / 3.0)
    push = _push_level(shanten, ctx['high_value'], threat['level'])
    risk_factor = max(shanten_factor, 0.08 + 0.12 * threat['level'])
    risk_coeff = (1.50 if ctx['high_value'] else 1.85) * (1.0 + 0.75 * threat['level']) * (1.20 - 0.42 * push)
    legal_logits = raw.copy()
    legal_logits[~legal_mask] = -np.inf
    raw_best = int(legal_logits.argmax())

    # Optional scale calibration.  Checkpoints can have very different logit
    # magnitudes, while the hand-written offence/risk terms have a fixed scale.
    # Normalising only legal discards makes those terms comparable without
    # disturbing the policy's preference for calls versus discards.
    if NORMALIZE_DISCARD_LOGITS:
        discard_legal = legal_mask[play_begin:chi_begin]
        values = raw[play_begin:chi_begin][discard_legal]
        if values.size >= 2:
            mean = float(values.mean())
            std = float(values.std())
            if std > 1e-6:
                calibrated = (raw[play_begin:chi_begin] - mean) / std
                adjusted[play_begin:chi_begin][discard_legal] = calibrated[discard_legal]

    if ACTION_VALUE_WEIGHT > 0 and aux is not None and 'action_value' in aux:
        q = np.tanh(np.asarray(aux['action_value']).reshape(-1))
        if q.size == adjusted.size:
            adjusted[legal_mask] += ACTION_VALUE_WEIGHT * q[legal_mask]

    # Balanced risk control: still push good hands, but avoid obvious far-hand deals.
    for tile, idx in FeatureAgent.OFFSET_TILE.items():
        a = play_begin + idx
        if legal_mask[a]:
            danger = _discard_danger(agent, tile, aux)
            adjusted[a] -= risk_coeff * danger * risk_factor
            if danger <= 0.05:
                adjusted[a] += (0.20 + 0.12 * threat['level']) * (1.0 + (1.0 - shanten_factor))
            if USE_AUX_RANK and aux is not None and 'discard_rank' in aux and idx < 34:
                adjusted[a] += 0.08 * (float(aux['discard_rank'][idx]) - 0.5)

    # Counterfactual offence: prefer lower shanten, more remaining ukeire, and
    # waits that are actually legal under the MCR eight-fan minimum.
    discard_analysis = {}
    for tile, idx in FeatureAgent.OFFSET_TILE.items():
        a = play_begin + idx
        if not legal_mask[a]:
            continue
        info = agent.analyze_discard(tile)
        discard_analysis[idx] = info
        adjusted[a] += 0.42 * max(-2, shanten - info['shanten'])
        adjusted[a] += 0.025 * min(20, info['ukeire'])
        if info['shanten'] == 0:
            if info['legal_waits']:
                remaining_waits = sum(max(0, 4 - agent.hand.count(w) - agent.shownTiles[w])
                                      for w in info['legal_waits'])
                adjusted[a] += 0.22 + 0.08 * min(6, remaining_waits)
                # Two equally wide waits are not equivalent in MCR: reward the
                # route that wins more fan, but use log scaling so one rare huge
                # hand cannot dominate the policy.
                fan_mass = 0.0
                for wait in info['legal_waits']:
                    remaining = max(0, 4 - agent.hand.count(wait) - agent.shownTiles[wait])
                    fan = max(8, int(info['wait_fan'].get(wait, 8)))
                    fan_mass += remaining * np.log2(fan / 8.0)
                fan_quality = fan_mass / max(1, remaining_waits)
                adjusted[a] += FAN_WAIT_WEIGHT * min(4.0, fan_quality)
            else:
                adjusted[a] -= 0.55  # nominal tenpai but cannot yet make eight fan

    # Slightly more aggressive than the old bot, but still keeps some defensive flexibility.
    if push < 0.45:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.45, 0.35, 0.30, 0.25
    elif shanten >= 2 and not ctx['high_value']:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.42, 0.33, 0.28, 0.24
    elif shanten == 1 or ctx['high_value']:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.25, 0.20, 0.16, 0.14
    else:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.12, 0.10, 0.08, 0.08
    adjusted[chi_begin:peng_begin] -= chi_penalty
    adjusted[peng_begin:gang_begin] -= peng_penalty
    adjusted[gang_begin:angang_begin] -= gang_penalty
    adjusted[bugang_begin:] -= bugang_penalty
    if threat['level'] >= 0.6 and shanten >= 1 and not ctx['high_value']:
        adjusted[chi_begin:peng_begin] -= 0.15
        adjusted[peng_begin:gang_begin] -= 0.12
        adjusted[gang_begin:angang_begin] -= 0.10
        adjusted[bugang_begin:] -= 0.10

    # Bonus for high-value Peng/Gang: restore penalty if holding 3+ copies of the tile.
    if hasattr(agent, 'hand') and hasattr(agent, 'curTile') and agent.curTile is not None:
        tile = agent.curTile
        if agent.hand.count(tile) >= 2 and tile in FeatureAgent.OFFSET_TILE:
            idx = FeatureAgent.OFFSET_TILE[tile]
            peng_action = peng_begin + idx
            if legal_mask[peng_action]:
                adjusted[peng_action] += 0.15
            gang_action = gang_begin + idx
            if legal_mask[gang_action]:
                adjusted[gang_action] += 0.10

    # Tenpai still attacks, but no longer ignores deal-in danger completely.
    if shanten == 0:
        adjusted[FeatureAgent.OFFSET_ACT['Pass']] -= 1.0

    # Seven-pairs awareness: avoid breaking pairs when chiitoi is close.
    if ctx['chiitoi_shanten'] <= shanten and ctx['chiitoi_shanten'] <= 2:
        for tile in ctx['pair_tiles']:
            if tile in FeatureAgent.OFFSET_TILE:
                a = play_begin + FeatureAgent.OFFSET_TILE[tile]
                if legal_mask[a]:
                    adjusted[a] -= 0.5
    # Route head is deliberately advisory: it preserves structures rather than
    # forcing a route.  Enable only after the new head has been trained.
    if FAN_ROUTE_WEIGHT > 0 and aux is not None and 'fan_route' in aux:
        route_logits = np.asarray(aux['fan_route']).reshape(-1)
        if route_logits.size == 5:
            route_prob = np.exp(route_logits - route_logits.max())
            route_prob /= max(1e-6, route_prob.sum())
            route = int(route_prob.argmax())
            strength = FAN_ROUTE_WEIGHT * float(route_prob[route])
            for tile, idx in FeatureAgent.OFFSET_TILE.items():
                a = play_begin + idx
                if not legal_mask[a]:
                    continue
                count = agent.hand.count(tile) if hasattr(agent, 'hand') else 0
                if route == 1 and count >= 2:       # seven pairs: preserve pairs
                    adjusted[a] -= strength
                elif route == 2 and ctx['flush_suit']:
                    adjusted[a] += strength * (0.45 if tile[0] != ctx['flush_suit'] and tile[0] not in 'FJ' else -0.20)
                elif route == 3 and count >= 2:     # triplet route: preserve pairs/triplets
                    adjusted[a] -= 0.70 * strength
                elif route == 4 and tile[0] in 'FJ':
                    adjusted[a] -= 0.60 * strength
    adjusted[~legal_mask] = -np.inf
    final_action = int(adjusted.argmax())

    # Danger gate: fold only far hands, or low-value one-shanten hands, when a safe close choice exists.
    should_gate = shanten >= 2 or (shanten == 1 and push < 0.72)
    if should_gate and play_begin <= final_action < chi_begin:
        final_tile = FeatureAgent.TILE_LIST[final_action - play_begin]
        final_danger = _discard_danger(agent, final_tile, aux)
        danger_threshold = 0.74 - 0.14 * threat['level']
        if final_danger >= danger_threshold:
            close_margin = 1.10 if shanten >= 2 else 0.70
            if threat['level'] >= 0.6:
                close_margin += 0.25
            best_safe = None
            best_safe_logit = -np.inf
            for tile, idx in FeatureAgent.OFFSET_TILE.items():
                a = play_begin + idx
                if not legal_mask[a]:
                    continue
                if raw[a] < raw[final_action] - close_margin:
                    continue
                tile_danger = _discard_danger(agent, tile, aux)
                if tile_danger <= 0.12 and raw[a] > best_safe_logit:
                    best_safe = a
                    best_safe_logit = raw[a]
            if best_safe is not None:
                return int(best_safe)

    # Do not let heuristics override a clearly preferred non-discard action.
    if raw_best < play_begin or raw_best >= chi_begin:
        if legal_logits[raw_best] >= adjusted[final_action] + 0.25:
            return raw_best
    return final_action


def _fallback_draw_response(obs):
    legal_mask = obs['action_mask'].astype(bool)
    play_begin = FeatureAgent.OFFSET_ACT['Play']
    chi_begin = FeatureAgent.OFFSET_ACT['Chi']
    legal_plays = np.flatnonzero(legal_mask[play_begin:chi_begin])
    if len(legal_plays):
        return 'PLAY %s' % FeatureAgent.TILE_LIST[int(legal_plays[0])]
    if legal_mask[FeatureAgent.OFFSET_ACT['Hu']]:
        return 'HU'
    for begin, text in (
        (FeatureAgent.OFFSET_ACT['AnGang'], 'GANG %s'),
        (FeatureAgent.OFFSET_ACT['BuGang'], 'BUGANG %s'),
    ):
        legal_tiles = np.flatnonzero(legal_mask[begin:begin + 34])
        if len(legal_tiles):
            return text % FeatureAgent.TILE_LIST[int(legal_tiles[0])]
    return 'PASS'

def obs2response(model, obs):
    legal_mask = obs['action_mask'].astype(bool)
    try:
        with torch.no_grad():
            seq_tile, seq_player = _discard_sequence_tensors(agent)
            input_dict = {
                'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0)),
                'discard_seq': torch.from_numpy(np.expand_dims(seq_tile, 0)),
                'discard_player': torch.from_numpy(np.expand_dims(seq_player, 0)),
            }
            if USE_AUX_RANK or USE_RISK_HEAD:
                logits, aux = model(input_dict, return_aux = True)
                aux_np = {k: v.detach().cpu().reshape(-1).tolist() for k, v in aux.items()}
            else:
                logits = model(input_dict)
                aux_np = None
        # Convert through Python values to avoid cross-module ndarray identity
        # conflicts between PyTorch, NumPy and the compiled Mahjong extension.
        logits_np = np.asarray(
            logits.detach().cpu().reshape(-1).tolist(), dtype=np.float32)
        action = _postprocess_action(agent, logits_np, obs['action_mask'], aux_np)
        # Final legality firewall: no post-processing bug may emit an illegal
        # action.  Fall back to the model's best legal action if necessary.
        if action < 0 or action >= len(legal_mask) or not legal_mask[action]:
            masked = logits_np.copy()
            masked[~legal_mask] = -np.inf
            action = int(masked.argmax())
    except Exception as exc:
        print('INFO inference fallback: %s' % exc, file = sys.stderr)
        hu = FeatureAgent.OFFSET_ACT['Hu']
        play_begin = FeatureAgent.OFFSET_ACT['Play']
        chi_begin = FeatureAgent.OFFSET_ACT['Chi']
        if hu < len(legal_mask) and legal_mask[hu]:
            action = hu
        else:
            legal_plays = np.flatnonzero(legal_mask[play_begin:chi_begin])
            if len(legal_plays):
                action = play_begin + int(legal_plays[0])
            elif legal_mask[FeatureAgent.OFFSET_ACT['Pass']]:
                action = FeatureAgent.OFFSET_ACT['Pass']
            else:
                legal = np.flatnonzero(legal_mask)
                action = int(legal[0]) if len(legal) else FeatureAgent.OFFSET_ACT['Pass']
    return agent.action2response(action)

if __name__ == '__main__':
    model = CNNModel()
    data_dir = os.environ.get('MODEL_PATH', '/data/best18.pkl')
    state = torch.load(data_dir, map_location = torch.device('cpu'))
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    missing = sorted(set(current) - set(compatible))
    unexpected = sorted(set(state) - set(compatible))
    if missing or unexpected:
        raise RuntimeError(
            'Final checkpoint mismatch: %d missing and %d unexpected tensors' %
            (len(missing), len(unexpected)))
    model.load_state_dict(compatible)
    model.eval()
    zimo = False
    angang = None
    input() # 1
    while True:
        request = input()
        while not request.strip(): request = input()
        request = request.split()
        if request[0] == '0':
            seatWind = int(request[1])
            agent = FeatureAgent(seatWind)
            zimo = False
            angang = None
            agent.request2obs('Wind %s' % request[2])
            print('PASS')
        elif request[0] == '1':
            agent.request2obs(' '.join(['Deal', *request[5:]]))
            print('PASS')
        elif request[0] == '2':
            obs = agent.request2obs('Draw %s' % request[1])
            response = obs2response(model, obs)
            response = response.split()
            if response[0] == 'Hu':
                print('HU')
            elif response[0] == 'Play':
                print('PLAY %s' % response[1])
            elif response[0] == 'Gang':
                print('GANG %s' % response[1])
                angang = response[1]
            elif response[0] == 'BuGang':
                print('BUGANG %s' % response[1])
            else:
                print(_fallback_draw_response(obs))
        elif request[0] == '3':
            p = int(request[1])
            if request[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif request[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif request[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if request[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, request[3]))
                elif request[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    response = response.split()
                    if response[0] == 'Hu':
                        print('HU')
                    elif response[0] == 'Pass':
                        print('PASS')
                    elif response[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif response[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d '% seatWind + ' '.join(response))
                        response2 = obs2response(model, obs)
                        print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + ' '.join(response))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
