#!/usr/bin/env python3
"""
FlashLM v5.2 "Nova-Ignition" — Optimized for 2 CPU / 5GB RAM
==============================================================
Streamlined version that ACTUALLY runs on constrained hardware.

Key optimizations:
- Smaller model (~8M params instead of 36M)
- Simplified attention (single-head, local window)
- Efficient MoE with vectorized routing
- Removed expensive Differential Attention overhead
"""

import os
import sys
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION — Optimized for 2 CPU Reality
# ============================================================================
CONFIG = {
    # Model - Smaller but efficient
    'vocab': 4096,
    'd_model': 192,
    'n_layers': 6,
    'n_heads': 3,
    'd_head': 64,
    'd_ffn': 384,
    'n_experts': 4,
    'expert_dim': 384,

    # Training - Aggressive settings
    'seq_len': 128,
    'batch_size': 4,
    'grad_accum': 8,
    'lr': 5e-3,
    'min_lr': 5e-5,
    'warmup_steps': 50,
    'weight_decay': 0.05,
    'grad_clip': 1.0,
    'betas': (0.9, 0.95),

    # Schedule
    'total_hours': 2.0,
    'save_every': 200,
    'eval_every': 50,
    'log_every': 10,
    'gen_every': 100,

    # Data
    'data_dir': 'data_v52',
    'out_dir': 'out_v52',
    'max_train_tokens': 5_000_000,
}


# ============================================================================
# BITLINEAR 1.58b — Ternary Weights
# ============================================================================
class BitLinear(nn.Module):
    """1.58-bit Linear: Weights quantized to {-1, 0, +1}"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        scale = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        w_q = torch.round(self.weight / scale).clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


# ============================================================================
# EFFICIENT LOCAL ATTENTION — O(n * window) instead of O(n²)
# ============================================================================
class LocalAttention(nn.Module):
    """Sliding window attention - much faster on CPU"""
    def __init__(self, d_model, n_heads, d_head, window_size=32):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window = window_size
        self.total_dim = n_heads * d_head

        self.qkv = BitLinear(d_model, 3 * self.total_dim)
        self.out = BitLinear(self.total_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x)

        # Project QKV
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: (B, T, H, Dh) -> (B, H, T, Dh)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scale
        q = q * (self.d_head ** -0.5)

        # Efficient causal local attention
        out = torch.empty_like(q)
        for t in range(T):
            start = max(0, t - self.window + 1)
            # (B, H, 1, Dh) @ (B, H, Dh, L) -> (B, H, 1, L)
            scores = torch.matmul(q[:, :, t:t+1], k[:, :, start:t+1].transpose(-1, -2))
            weights = F.softmax(scores, dim=-1)
            out[:, :, t] = torch.matmul(weights, v[:, :, start:t+1]).squeeze(2)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


# ============================================================================
# SIMPLIFIED MoE — Vectorized Routing
# ============================================================================
class SimpleMoE(nn.Module):
    """Efficient MoE with vectorized operations"""
    def __init__(self, d_model, expert_dim, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model

        # Shared expert (always active)
        self.shared = nn.Sequential(
            BitLinear(d_model, expert_dim),
            nn.SiLU(),
            BitLinear(expert_dim, d_model)
        )

        # Routed experts - combined for efficiency
        self.expert_up = BitLinear(d_model, n_experts * expert_dim)
        self.expert_down = BitLinear(n_experts * expert_dim, d_model)

        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x)

        # Shared expert
        shared_out = self.shared(h)

        # Router
        router_logits = self.router(h)  # (B, T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, T, E)

        # All experts at once
        expert_hidden = self.expert_up(h)  # (B, T, E * expert_dim)
        expert_hidden = F.silu(expert_hidden)

        # Reshape and weight by router
        expert_hidden = expert_hidden.view(B, T, self.n_experts, -1)  # (B, T, E, dim)
        router_weights = router_probs.unsqueeze(-1)  # (B, T, E, 1)
        weighted = (expert_hidden * router_weights).view(B, T, -1)  # (B, T, E * dim)

        routed_out = self.expert_down(weighted)

        return shared_out + routed_out


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================
class NovaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ffn, n_experts, expert_dim):
        super().__init__()
        self.attn = LocalAttention(d_model, n_heads, d_head)
        self.ffn = SimpleMoE(d_model, expert_dim, n_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# NOVA-IGNITION LM
# ============================================================================
class NovaIgnitionLM(nn.Module):
    def __init__(self, vocab=4096, d_model=192, n_layers=6, n_heads=3,
                 d_head=64, d_ffn=384, n_experts=4, expert_dim=384):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}

        self.embed = nn.Embedding(vocab, d_model)
        self.embed_scale = d_model ** -0.5

        self.blocks = nn.ModuleList([
            NovaBlock(d_model, n_heads, d_head, d_ffn, n_experts, expert_dim)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = BitLinear(d_model, vocab)

        # Init
        nn.init.normal_(self.embed.weight, std=0.02)

        # Stats
        self._total_params = sum(p.numel() for p in self.parameters())
        self._bitlinear_params = sum(
            p.numel() for m in self.modules()
            if isinstance(m, BitLinear) for p in m.parameters()
        )

        print(f"\n{'═'*55}")
        print(f"🚀 FlashLM v5.2 'Nova-Ignition' (Optimized)")
        print(f"{'═'*55}")
        print(f"   Parameters:      {self._total_params:,}")
        print(f"   BitLinear:       {self._bitlinear_params:,} ({100*self._bitlinear_params/self._total_params:.0f}%)")
        print(f"   Est. RAM:        ~{self._total_params*4*2.5/1024/1024/1024:.2f} GB")
        print(f"{'═'*55}\n")

    def forward(self, x, targets=None):
        h = self.embed(x) * self.embed_scale
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -CONFIG['seq_len']:]
            logits = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================================
# ZERO-COPY DATASET
# ============================================================================
class ZeroCopyDataset(Dataset):
    def __init__(self, bin_path, seq_len, max_tokens=None):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        if max_tokens and len(self.data) > max_tokens:
            self.data = self.data[:max_tokens]
        self.n = (len(self.data) - 1) // seq_len
        print(f"   Dataset: {self.n:,} samples, {len(self.data):,} tokens")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = np.array(self.data[i : i + self.seq_len + 1])
        return (torch.from_numpy(chunk[:-1].astype(np.int64)),
                torch.from_numpy(chunk[1:].astype(np.int64)))


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(config):
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(exist_ok=True)

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    tok_path = data_dir / "tokenizer.json"

    if train_bin.exists() and val_bin.exists() and tok_path.exists():
        print(f"✅ Data already prepared")
        return str(tok_path)

    print(f"\n{'═'*55}")
    print(f"📦 PREPARING DATA")
    print(f"{'═'*55}")

    train_txt = data_dir / "stories.txt"

    if not train_txt.exists():
        print("📥 Downloading TinyStories...")
        import urllib.request
        import random

        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
        urllib.request.urlretrieve(url, train_txt)
        print(f"   Downloaded: {train_txt.stat().st_size / 1e6:.1f} MB")

        if train_txt.stat().st_size > 30_000_000:
            with open(train_txt, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
            if len(lines) > 20000:
                lines = random.sample(lines, 20000)
            with open(train_txt, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"   Reduced: {train_txt.stat().st_size / 1e6:.1f} MB")

    # Tokenizer
    print(f"\n🔤 Training tokenizer...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    if not tok_path.exists():
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=config['vocab'], min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        ))
        tokenizer.save(str(tok_path))
    else:
        tokenizer = Tokenizer.from_file(str(tok_path))

    # Tokenize
    if not train_bin.exists():
        print(f"🔢 Tokenizing...")
        with open(train_txt, 'r', encoding='utf-8') as f:
            stories = [s.strip() for s in f.read().split('\n\n') if s.strip()]

        tokens = []
        eos_id = tokenizer.token_to_id("<eos>") or 0
        for story in stories[:25000]:
            tokens.extend(tokenizer.encode(story).ids)
            tokens.append(eos_id)

        tokens = tokens[:config['max_train_tokens']]
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(str(train_bin))
        print(f"   Train: {len(arr):,} tokens")

        split = int(len(arr) * 0.9)
        arr[split:].tofile(str(val_bin))
        print(f"   Val: {len(arr) - split:,} tokens")

    print(f"{'═'*55}\n")
    return str(tok_path)


# ============================================================================
# LR SCHEDULE
# ============================================================================
def get_lr(step, config):
    if step < config['warmup_steps']:
        return config['lr'] * (step + 1) / config['warmup_steps']

    cycle = 300
    pos = (step - config['warmup_steps']) % cycle
    num = (step - config['warmup_steps']) // cycle
    max_lr = config['lr'] * (0.9 ** num)

    return config['min_lr'] + 0.5 * (max_lr - config['min_lr']) * (1 + math.cos(math.pi * pos / cycle))


# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate(model, val_data, seq_len, max_batches=20):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n = (len(val_data) - 1) // seq_len

    for _ in range(min(max_batches, n // 4)):
        batch_x, batch_y = [], []
        for _ in range(4):
            i = np.random.randint(0, n) * seq_len
            chunk = val_data[i:i + seq_len + 1]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])

        x = torch.tensor(np.stack(batch_x), dtype=torch.long)
        y = torch.tensor(np.stack(batch_y), dtype=torch.long)

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss = model(x, targets=y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    model.train()
    avg = total_loss / max(total_tokens, 1)
    return {'loss': avg, 'ppl': math.exp(min(avg, 20))}


# ============================================================================
# TRAINING
# ============================================================================
def train():
    config = CONFIG
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(exist_ok=True)

    # CPU optimization
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'

    # Data
    tok_path = prepare_data(config)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    val_data = np.fromfile(str(Path(config['data_dir']) / 'val.bin'), dtype=np.uint16)
    print(f"📊 Val: {len(val_data):,} tokens\n")

    train_ds = ZeroCopyDataset(
        str(Path(config['data_dir']) / 'train.bin'),
        config['seq_len'], config['max_train_tokens']
    )

    train_dl = DataLoader(train_ds, batch_size=config['batch_size'],
                          shuffle=True, num_workers=0, drop_last=True)

    # Model
    print("🏗️  Building model...")
    model = NovaIgnitionLM(
        vocab=config['vocab'], d_model=config['d_model'],
        n_layers=config['n_layers'], n_heads=config['n_heads'],
        d_head=config['d_head'], d_ffn=config['d_ffn'],
        n_experts=config['n_experts'], expert_dim=config['expert_dim'],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  betas=config['betas'], weight_decay=config['weight_decay'])

    # Resume
    step, tokens_seen, best_val, log_loss = 0, 0, float('inf'), 0.0
    ckpt_path = out_dir / 'latest.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, tokens_seen, best_val = ckpt['step'], ckpt['tokens'], ckpt.get('best_val', float('inf'))
        print(f"📂 Resumed: step {step}, {tokens_seen/1e6:.1f}M tokens\n")

    json.dump(config, open(out_dir / 'config.json', 'w'), indent=2)

    prompts = ["Once upon a time", "The little girl", "A dog named"]
    toks_per_step = config['batch_size'] * config['grad_accum'] * config['seq_len']

    print(f"{'═'*55}")
    print(f"🚀 TRAINING — {config['total_hours']}h | {toks_per_step:,} tok/step")
    print(f"{'═'*55}\n")

    t_start = time.time()
    train_iter = iter(train_dl)

    while True:
        elapsed = time.time() - t_start
        if elapsed / 3600 >= config['total_hours']:
            print(f"\n⏰ Time limit ({elapsed/3600:.2f}h)")
            break

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config['grad_accum']):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                loss = model(x, targets=y) / config['grad_accum']
            loss.backward()
            log_loss += loss.item()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        if step % 25 == 0:
            gc.collect()

        if step % config['log_every'] == 0:
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"Step {step:4d} │ Loss {log_loss/config['log_every']:.4f} │ "
                  f"LR {lr:.1e} │ {tps:,.0f} tok/s │ {tokens_seen/1e6:.2f}M")
            log_loss = 0.0

        if step % config['eval_every'] == 0:
            m = evaluate(model, val_data, config['seq_len'])
            if m['loss'] < best_val:
                best_val = m['loss']
                torch.save(model.state_dict(), out_dir / 'best.pt')
            print(f"  ✦ VAL │ Loss {m['loss']:.4f} │ PPL {m['ppl']:.1f}{' ★' if m['loss'] == best_val else ''}")

        if step % config['gen_every'] == 0 and step > 0:
            print(f"\n{'─'*40}")
            model.eval()
            for p in prompts[:2]:
                ids = tokenizer.encode(p).ids
                x = torch.tensor([ids], dtype=torch.long)
                out = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=30)
                print(f"  > {tokenizer.decode(out[0].tolist())[:120]}")
            model.train()
            print(f"{'─'*40}\n")

        if step % config['save_every'] == 0:
            torch.save({'step': step, 'tokens': tokens_seen, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'best_val': best_val}, out_dir / 'latest.pt')
            print(f"  💾 Saved checkpoint")

    # Final
    m = evaluate(model, val_data, config['seq_len'])
    torch.save(model.state_dict(), out_dir / 'final.pt')

    print(f"\n{'═'*55}")
    print(f"✅ DONE")
    print(f"   Steps:    {step:,}")
    print(f"   Tokens:   {tokens_seen/1e6:.2f}M")
    print(f"   Time:     {(time.time()-t_start)/3600:.2f}h")
    print(f"   Loss:     {m['loss']:.4f}")
    print(f"   PPL:      {m['ppl']:.1f}")
    print(f"{'═'*55}")

    # Final generations
    model.eval()
    for p in prompts:
        ids = tokenizer.encode(p).ids
        out = model.generate(torch.tensor([ids], dtype=torch.long), max_new_tokens=60)
        print(f"\n> {p}\n  {tokenizer.decode(out[0].tolist())}")


if __name__ == '__main__':
    train()
