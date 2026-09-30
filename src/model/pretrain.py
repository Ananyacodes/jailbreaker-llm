import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch

from src.model.transformer import GPT, GPTConfig
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def warm_start(model_name: str, out_dir: str, copy_to_out_dir: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info('warm-starting from %s -- no from-scratch pretraining needed', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not copy_to_out_dir:
        log.info('base model is ready in the Hugging Face cache; no duplicate project copy created')
        return model_name
    os.makedirs(out_dir, exist_ok=True)
    tok_dir = os.path.join(out_dir, 'tokenizer')
    tokenizer.save_pretrained(tok_dir)
    model.save_pretrained(out_dir)
    log.info('cached warm-start base + tokenizer under %s', out_dir)
    return out_dir


def get_batch(data_path, block_size, batch_size, device):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return (x.to(device), y.to(device))


def get_autocast_ctx(dtype_str: str, device: str):
    if device == 'cpu':
        return nullcontext()
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    dtype = dtype_map.get(dtype_str, torch.float32)
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=dtype)


@torch.no_grad()
def estimate_val_loss(model, val_bin, block_size, batch_size, device, eval_iters, autocast_ctx):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_bin, block_size, batch_size, device)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def from_scratch(cfg: dict, resume: bool):
    mcfg = GPTConfig(**cfg['model'])
    pcfg = cfg['pretrain']
    device = pcfg['device'] if torch.cuda.is_available() else 'cpu'
    autocast_ctx = get_autocast_ctx(pcfg.get('dtype', 'float32'), device)
    model = GPT(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pcfg['learning_rate'], weight_decay=pcfg['weight_decay'])
    os.makedirs(pcfg['out_dir'], exist_ok=True)
    ckpt_path = os.path.join(pcfg['out_dir'], 'latest.pt')
    best_path = os.path.join(pcfg['out_dir'], 'best.pt')
    start_iter = 0
    best_val_loss = float('inf')
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        log.info('resumed from %s at iter %d', ckpt_path, start_iter)
    if pcfg.get('compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
    train_bin = os.path.join(pcfg['data_dir'], 'train.bin')
    val_bin = os.path.join(pcfg['data_dir'], 'val.bin')
    for it in range(start_iter, pcfg['max_iters']):
        lr = pcfg['learning_rate']
        if it < pcfg['warmup_iters']:
            lr = pcfg['learning_rate'] * it / max(1, pcfg['warmup_iters'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        x, y = get_batch(train_bin, mcfg.block_size, pcfg['batch_size'], device)
        with autocast_ctx:
            _, loss = model(x, y)
        loss = loss / pcfg['grad_accum_steps']
        loss.backward()
        if (it + 1) % pcfg['grad_accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg['grad_clip'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if it % pcfg['eval_interval'] == 0:
            val_loss = estimate_val_loss(model, val_bin, mcfg.block_size, pcfg['batch_size'], device, pcfg['eval_iters'], autocast_ctx)
            train_loss_val = loss.item() * pcfg['grad_accum_steps']
            log.info('iter %d: train_loss=%.4f val_loss=%.4f', it, train_loss_val, val_loss)
            ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': mcfg, 'iter': it, 'best_val_loss': best_val_loss}
            torch.save(ckpt, ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt['best_val_loss'] = best_val_loss
                torch.save(ckpt, best_path)
    log.info('from-scratch pretraining done, final checkpoint saved to %s', pcfg['out_dir'])


def main(cfg_path: str, resume: bool = False):
    cfg = load_config(cfg_path)
    pcfg = cfg['pretrain']
    if pcfg.get('warm_start_from'):
        warm_start(
            pcfg['warm_start_from'],
            pcfg.get('out_dir', 'checkpoints/base_model'),
            copy_to_out_dir=pcfg.get('copy_to_out_dir', False),
        )
    else:
        from_scratch(cfg, resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base_model.yaml')
    parser.add_argument('--resume', action='store_true', help='resume a from-scratch run from out_dir/latest.pt')
    args = parser.parse_args()
    main(args.config, args.resume)
