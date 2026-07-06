from dataset import MahjongGBDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import CNNModel
import torch.nn.functional as F
import torch
import argparse
import os
import time


def get_device(preferred):
    if preferred == 'auto':
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device('npu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(preferred)


def move_batch(batch, device):
    obs, mask, act, wt, seq_tile, seq_player, aux = batch
    # non_blocking is only safe on CUDA; disable on NPU to avoid task-scheduler errors
    non_blocking = device.type == 'cuda'
    input_dict = {
        'observation': obs.to(device, non_blocking = non_blocking),
        'action_mask': mask.to(device, non_blocking = non_blocking),
        'discard_seq': seq_tile.to(device, non_blocking = non_blocking),
        'discard_player': seq_player.to(device, non_blocking = non_blocking),
    }
    aux = {k: v.to(device, non_blocking = non_blocking) for k, v in aux.items()}
    return input_dict, act.long().to(device, non_blocking = non_blocking), wt.float().to(device, non_blocking = non_blocking), aux


def conditional_subaction_loss(model, logits, target, wt_norm):
    type_target = model.action_type_targets(target)
    ranges = {
        2: (2, 36),
        3: (36, 99),
        4: (99, 133),
        5: (133, 167),
        6: (167, 201),
        7: (201, 235),
    }
    losses = []
    for action_type, (begin, end) in ranges.items():
        mask = type_target == action_type
        if not mask.any():
            continue
        local_target = target[mask] - begin
        local_loss = F.cross_entropy(logits[mask, begin:end], local_target, reduction = 'none')
        losses.append((local_loss * wt_norm[mask]).mean())
    if not losses:
        return target.new_tensor(0.0, dtype = torch.float32)
    return torch.stack(losses).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 2048)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--min-lr', type = float, default = 1e-5)
    parser.add_argument('--weight-decay', type = float, default = 1e-4)
    parser.add_argument('--label-smoothing', type = float, default = 0.0)
    parser.add_argument('--type-loss-weight', type = float, default = 0.2)
    parser.add_argument('--subaction-loss-weight', type = float, default = 1.0)
    parser.add_argument('--full-policy-loss-weight', type = float, default = 0.25)
    parser.add_argument('--win-loss-weight', type = float, default = 0.05)
    parser.add_argument('--fan-loss-weight', type = float, default = 0.10)
    parser.add_argument('--shanten-loss-weight', type = float, default = 0.05)
    parser.add_argument('--discard-rank-loss-weight', type = float, default = 0.10)
    parser.add_argument('--risk-loss-weight', type = float, default = 0.15)
    parser.add_argument('--risk-opp-loss-weight', type = float, default = 0.20)
    parser.add_argument('--risk-severity-loss-weight', type = float, default = 0.10)
    parser.add_argument('--risk-severity-opp-loss-weight', type = float, default = 0.10)
    parser.add_argument('--tenpai-opp-loss-weight', type = float, default = 0.10)
    parser.add_argument('--fan-route-loss-weight', type = float, default = 0.05)
    parser.add_argument('--action-value-loss-weight', type = float, default = 0.20)
    parser.add_argument('--weighted-sampler', action = 'store_true',
                        help = 'oversample high-weight/high-score samples instead of only weighting the loss')
    parser.add_argument('--split-ratio', type = float, default = 0.9)
    parser.add_argument('--num-workers', type = int, default = 8)
    parser.add_argument('--cache-size', type = int, default = 512,
                        help = 'number of match .npz files cached per DataLoader worker')
    parser.add_argument('--prefetch-factor', type = int, default = 4,
                        help = 'DataLoader prefetch factor when num-workers > 0')
    parser.add_argument('--no-persistent-workers', action = 'store_true',
                        help = 'disable persistent DataLoader workers')
    parser.add_argument('--device', default = 'auto')
    parser.add_argument('--amp', choices = ('auto', 'bf16', 'fp16', 'none'), default = 'auto',
                        help = 'CUDA mixed precision; auto selects BF16 on modern GPUs')
    parser.add_argument('--compile', action = 'store_true',
                        help = 'enable torch.compile (optional; benchmark before long runs)')
    parser.add_argument('--no-fused-adamw', action = 'store_true',
                        help = 'disable CUDA fused AdamW')
    parser.add_argument('--no-cudnn-benchmark', action = 'store_true')
    parser.add_argument('--logdir', default = 'model')
    parser.add_argument('--resume', type = str, default = None,
                        help = 'path to an epoch checkpoint (.pkl) to resume training from')
    return parser.parse_args()
 
if __name__ == '__main__':
    args = parse_args()
    checkpoint_dir = os.path.join(args.logdir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok = True)
    device = get_device(args.device)
    amp_mode = args.amp
    if amp_mode == 'auto':
        amp_mode = 'bf16' if device.type == 'cuda' else 'none'
    amp_enabled = device.type == 'cuda' and amp_mode in ('bf16', 'fp16')
    amp_dtype = torch.bfloat16 if amp_mode == 'bf16' else torch.float16
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        props = torch.cuda.get_device_properties(device)
        print('CUDA device:', props.name, 'capability %d.%d' % (props.major, props.minor),
              'torch', torch.__version__, 'CUDA runtime', torch.version.cuda,
              'AMP', amp_mode, flush = True)
        cuda_version = tuple(int(x) for x in torch.version.cuda.split('.')[:2]) if torch.version.cuda else (0, 0)
        if props.major >= 12 and cuda_version < (12, 8):
            raise RuntimeError('Blackwell GPU requires a PyTorch CUDA 12.8+ build')
    
    # Load dataset
    trainDataset = MahjongGBDataset(0, args.split_ratio, True, cache_size = args.cache_size)
    validateDataset = MahjongGBDataset(args.split_ratio, 1, False, cache_size = args.cache_size)
    # pin_memory is only beneficial on CUDA; disable on NPU
    pin_memory = device.type == 'cuda'
    sampler = None
    shuffle = True
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(
            weights = torch.from_numpy(trainDataset.sample_weights()).double(),
            num_samples = len(trainDataset),
            replacement = True
        )
        shuffle = False
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = not args.no_persistent_workers
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = shuffle,
                        sampler = sampler, **loader_kwargs)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False,
                         **loader_kwargs)
    
    # Load model
    model = CNNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay,
                                  fused = device.type == 'cuda' and not args.no_fused_adamw)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = args.min_lr)
    best_acc = 0
    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location = 'cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            # Full training checkpoint
            current = model.state_dict()
            compatible = {k: v for k, v in ckpt['model'].items()
                          if k in current and current[k].shape == v.shape}
            current.update(compatible)
            model.load_state_dict(current)
            exact = len(compatible) == len(current)
            if exact:
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                start_epoch = ckpt['epoch'] + 1
                best_acc = ckpt.get('best_acc', 0)
            else:
                print('Loaded %d/%d tensors; new heads start fresh.' %
                      (len(compatible), len(current)), flush = True)
            print('Resumed from epoch %d, best_acc=%.4f' % (ckpt['epoch'], best_acc), flush = True)
        else:
            # Legacy: plain state dict (e.g. best.pkl)
            current = model.state_dict()
            compatible = {k: v for k, v in ckpt.items()
                          if k in current and current[k].shape == v.shape}
            current.update(compatible)
            model.load_state_dict(current)
            print('Loaded model weights from %s (no optimizer state)' % args.resume, flush = True)
        model.to(device)

    checkpoint_model = model
    if args.compile:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile requires PyTorch 2.x')
        model = torch.compile(model)
        print('Enabled torch.compile; the first iterations include compilation time.', flush = True)
    scaler_enabled = amp_enabled and amp_mode == 'fp16'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled = scaler_enabled)
    else:  # compatibility with the legacy Ascend/PyTorch 2.1 environment
        scaler = torch.cuda.amp.GradScaler(enabled = scaler_enabled)
    if args.resume and 'ckpt' in locals() and isinstance(ckpt, dict) and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    
    # Train and validate
    for e in range(start_epoch, args.epochs):
        print('Epoch', e, flush = True)
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train(True)
        for i, d in enumerate(loader):
            input_dict, target, wt, aux_target = move_batch(d, device)
            with torch.autocast(device_type = 'cuda', dtype = amp_dtype, enabled = amp_enabled):
                logits, type_logits, aux_pred = model(input_dict, return_type_logits = True, return_aux = True)
            # Keep numerically sensitive, highly imbalanced auxiliary losses in FP32.
            logits = logits.float()
            type_logits = type_logits.float()
            aux_pred = {k: v.float() for k, v in aux_pred.items()}
            # Weighted policy loss: higher-fan games contribute more
            per_sample_loss = F.cross_entropy(logits, target,
                                              label_smoothing = args.label_smoothing,
                                              reduction = 'none')
            wt_norm = wt / wt.mean().clamp(min = 1e-6)  # zero for non-winner policy samples
            policy_loss = (per_sample_loss * wt_norm).mean()
            type_target = model.action_type_targets(target)
            type_loss = (F.cross_entropy(type_logits, type_target, reduction = 'none') * wt_norm).mean()
            subaction_loss = conditional_subaction_loss(model, logits, target, wt_norm)
            win_loss = F.binary_cross_entropy_with_logits(aux_pred['win_logit'], aux_target['win'].float())
            fan_loss = F.smooth_l1_loss(aux_pred['fan'].sigmoid(), aux_target['fan'].float())
            shanten_loss = F.smooth_l1_loss(aux_pred['shanten'].sigmoid(), aux_target['shanten'].float())
            discard_mask = aux_target['discard_rank'].float() >= 0
            if discard_mask.any():
                discard_rank_loss = F.mse_loss(
                    aux_pred['discard_rank'].sigmoid()[discard_mask],
                    aux_target['discard_rank'].float()[discard_mask]
                )
            else:
                discard_rank_loss = logits.new_tensor(0.0)
            play_legal = input_dict['action_mask'][:, 2:36].bool()
            if play_legal.any():
                aggregate_target = aux_target['risk'].float()[play_legal]
                aggregate_pos = aggregate_target.sum().clamp(min = 1.0)
                aggregate_neg = aggregate_target.numel() - aggregate_pos
                risk_loss = F.binary_cross_entropy_with_logits(
                    aux_pred['risk'][play_legal], aggregate_target,
                    pos_weight = (aggregate_neg / aggregate_pos).clamp(min = 1.0, max = 50.0))
            else:
                risk_loss = logits.new_tensor(0.0)
            opp_mask = play_legal.unsqueeze(1).expand(-1, 3, -1)
            if opp_mask.any():
                opp_target = aux_target['risk_opp'].float()[opp_mask]
                positives = opp_target.sum().clamp(min = 1.0)
                negatives = opp_target.numel() - positives
                pos_weight = (negatives / positives).clamp(min = 1.0, max = 50.0)
                risk_opp_loss = F.binary_cross_entropy_with_logits(
                    aux_pred['risk_opp'][opp_mask], opp_target, pos_weight = pos_weight)
                risk_severity_loss = F.smooth_l1_loss(
                    aux_pred['risk_loss'].sigmoid()[play_legal],
                    aux_target['risk_loss'].float()[play_legal])
                risk_severity_opp_loss = F.smooth_l1_loss(
                    aux_pred['risk_loss_opp'].sigmoid()[opp_mask],
                    aux_target['risk_loss_opp'].float()[opp_mask])
            else:
                risk_opp_loss = logits.new_tensor(0.0)
                risk_severity_loss = logits.new_tensor(0.0)
                risk_severity_opp_loss = logits.new_tensor(0.0)
            fan_route_loss = F.cross_entropy(aux_pred['fan_route'], aux_target['fan_route'].long())
            chosen_value = aux_pred['action_value'].gather(1, target.unsqueeze(1)).squeeze(1)
            action_value_loss = F.smooth_l1_loss(
                chosen_value.tanh(), aux_target['score'].float())
            tenpai_target = aux_target['tenpai_opp'].float()
            tenpai_pos = tenpai_target.sum().clamp(min = 1.0)
            tenpai_neg = tenpai_target.numel() - tenpai_pos
            tenpai_opp_loss = F.binary_cross_entropy_with_logits(
                aux_pred['tenpai_opp'], tenpai_target,
                pos_weight = (tenpai_neg / tenpai_pos).clamp(min = 1.0, max = 20.0))
            loss = (args.full_policy_loss_weight * policy_loss
                    + args.type_loss_weight * type_loss
                    + args.subaction_loss_weight * subaction_loss
                    + args.win_loss_weight * win_loss
                    + args.fan_loss_weight * fan_loss
                    + args.shanten_loss_weight * shanten_loss
                    + args.discard_rank_loss_weight * discard_rank_loss
                    + args.risk_loss_weight * risk_loss
                    + args.risk_opp_loss_weight * risk_opp_loss
                    + args.risk_severity_loss_weight * risk_severity_loss
                    + args.risk_severity_opp_loss_weight * risk_severity_opp_loss
                    + args.tenpai_opp_loss_weight * tenpai_opp_loss
                    + args.fan_route_loss_weight * fan_route_loss
                    + args.action_value_loss_weight * action_value_loss)
            if i % 128 == 0:
                elapsed = max(1e-6, time.perf_counter() - epoch_start)
                print('Iteration %d/%d'%(i, len(trainDataset) // args.batch_size + 1),
                      'samples_per_sec', int((i + 1) * args.batch_size / elapsed),
                      'policy_loss', policy_loss.item(), 'type_loss', type_loss.item(),
                      'subaction_loss', subaction_loss.item(),
                      'win_loss', win_loss.item(), 'fan_loss', fan_loss.item(),
                      'shanten_loss', shanten_loss.item(), 'discard_rank_loss', discard_rank_loss.item(),
                      'risk_loss', risk_loss.item(), 'risk_opp_loss', risk_opp_loss.item(),
                      'risk_severity_loss', risk_severity_loss.item(),
                      'risk_severity_opp_loss', risk_severity_opp_loss.item(),
                      'tenpai_opp_loss', tenpai_opp_loss.item(), 'fan_route_loss', fan_route_loss.item(),
                      'action_value_loss', action_value_loss.item(),
                      flush = True)
            optimizer.zero_grad(set_to_none = True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        print('Run validation:', flush = True)
        correct = 0
        policy_total = 0
        total_loss = 0
        risk_tp = risk_fp = risk_fn = 0
        tenpai_tp = tenpai_fp = tenpai_fn = 0
        risk_severity_abs = risk_severity_n = 0
        action_value_abs = action_value_n = 0
        fan_route_correct = fan_route_n = 0
        model.train(False)
        for i, d in enumerate(vloader):
            input_dict, target, wt, aux_target = move_batch(d, device)
            with torch.no_grad():
                with torch.autocast(device_type = 'cuda', dtype = amp_dtype, enabled = amp_enabled):
                    logits, aux_pred = model(input_dict, return_aux = True)
                logits = logits.float()
                aux_pred = {k: v.float() for k, v in aux_pred.items()}
                pred = logits.argmax(dim = 1)
                policy_mask = wt > 0
                if policy_mask.any():
                    total_loss += F.cross_entropy(logits[policy_mask], target[policy_mask], reduction = 'sum').item()
                    correct += torch.eq(pred[policy_mask], target[policy_mask]).sum().item()
                    policy_total += int(policy_mask.sum().item())
                play_legal = input_dict['action_mask'][:, 2:36].bool()
                predicted_risk = aux_pred['risk'].sigmoid()[play_legal] >= 0.5
                true_risk = aux_target['risk'].bool()[play_legal]
                risk_tp += int((predicted_risk & true_risk).sum().item())
                risk_fp += int((predicted_risk & ~true_risk).sum().item())
                risk_fn += int((~predicted_risk & true_risk).sum().item())
                predicted_tenpai = aux_pred['tenpai_opp'].sigmoid() >= 0.5
                true_tenpai = aux_target['tenpai_opp'].bool()
                tenpai_tp += int((predicted_tenpai & true_tenpai).sum().item())
                tenpai_fp += int((predicted_tenpai & ~true_tenpai).sum().item())
                tenpai_fn += int((~predicted_tenpai & true_tenpai).sum().item())
                risk_severity_abs += F.l1_loss(
                    aux_pred['risk_loss'].sigmoid()[play_legal],
                    aux_target['risk_loss'].float()[play_legal], reduction = 'sum').item()
                risk_severity_n += int(play_legal.sum().item())
                chosen_value = aux_pred['action_value'].gather(1, target.unsqueeze(1)).squeeze(1).tanh()
                action_value_abs += F.l1_loss(
                    chosen_value, aux_target['score'].float(), reduction = 'sum').item()
                action_value_n += int(target.shape[0])
                fan_route_correct += int((aux_pred['fan_route'].argmax(dim = 1)
                                          == aux_target['fan_route'].long()).sum().item())
                fan_route_n += int(target.shape[0])
        acc = correct / max(1, policy_total)
        val_loss = total_loss / max(1, policy_total)
        risk_precision = risk_tp / max(1, risk_tp + risk_fp)
        risk_recall = risk_tp / max(1, risk_tp + risk_fn)
        tenpai_precision = tenpai_tp / max(1, tenpai_tp + tenpai_fp)
        tenpai_recall = tenpai_tp / max(1, tenpai_tp + tenpai_fn)
        print('Validation policy_acc=%.4f loss=%.4f risk_precision=%.4f risk_recall=%.4f '
              'tenpai_precision=%.4f tenpai_recall=%.4f severity_mae=%.4f '
              'action_value_mae=%.4f fan_route_acc=%.4f' %
              (acc, val_loss, risk_precision, risk_recall,
               tenpai_precision, tenpai_recall,
               risk_severity_abs / max(1, risk_severity_n),
               action_value_abs / max(1, action_value_n),
               fan_route_correct / max(1, fan_route_n)), flush = True)
        cpu_state_dict = {k: v.detach().cpu() for k, v in checkpoint_model.state_dict().items()}
        # Save full checkpoint for resumption
        torch.save({
            'epoch'     : e,
            'model'     : cpu_state_dict,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'scaler'    : scaler.state_dict(),
            'best_acc'  : best_acc,
        }, os.path.join(checkpoint_dir, '%d.pkl' % e))
        if acc > best_acc:
            best_acc = acc
            # best.pkl is a plain state dict so __main__.py can load it directly
            torch.save(cpu_state_dict, os.path.join(checkpoint_dir, 'best.pkl'))
        print('Epoch', e + 1, 'Validate loss:', val_loss, 'Validate acc:', acc, 'Best acc:', best_acc, flush = True)
        if device.type == 'cuda':
            print('Epoch seconds %.1f, peak CUDA memory %.2f GiB' % (
                time.perf_counter() - epoch_start,
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)), flush = True)
