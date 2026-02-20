"""
Credit to @changcheng967 for the original idea and this is an experiment inspired by it.

Updated Architecture:
------------------------------
Input Tokens
    ↓
Embedding (vocab_size=10000, dim)
    ↓
[ImliBlock × N_BLOCKS]
    ├─ GatedConvMixer
    │   ├─ BitLinear(dim, dim*2) [with internal RMSNorm + blockwise quantization]
    │   ├─ Conv1d (depthwise, causal, kernel=8)
    │   └─ BitLinear(dim, dim) [with internal RMSNorm + blockwise quantization]
    └─ TernaryGLU
        ├─ BitLinear(dim, hidden) [with internal RMSNorm]
        ├─ BitLinear(dim, hidden) [with internal RMSNorm]
        └─ BitLinear(hidden, dim) [with internal RMSNorm]
    ↓
RMSNorm
    ↓
Linear Head (tied)
    ↓
Logits
------------------------------

"""

import os
import sys
import math
import time
import json
import argparse
from collections import Counter
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="TeenyLM Training")
parser.add_argument("--small", action="store_true",
                    help="Small config: d=192, 6 blocks, ~4.3    M params")
parser.add_argument("--large", action="store_true",
                    help="Large config: d=384, 8 blocks, ~15.7M params")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--epochs", type=int, default=10,
                    help="Max epochs (default: 10)")
parser.add_argument("--hours", type=float, default=None,
                    help="Time limit in hours (optional, overrides epochs)")
parser.add_argument("--batch", type=int, default=None,
                    help="Batch size (auto-detected if not set)")
parser.add_argument("--lr", type=float, default=None,
                    help="Peak learning rate (auto-set per config)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save-dir", type=str, default="checkpoints_v2",
                    help="Directory for ckpts")
args = parser.parse_args()

# ============================================================
# Auto-detect hardware
# ============================================================
NUM_CORES = os.cpu_count() or 2
try:
    import psutil
    TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
except ImportError:
    try:
        TOTAL_RAM_GB = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except:
        TOTAL_RAM_GB = 4.0  # conservative default

if not args.small and not args.large:
    if NUM_CORES >= 8 and TOTAL_RAM_GB >= 32:
        args.large = True
    else:
        args.small = True

VOCAB_SIZE = 10000
KERNEL_SIZE = 8
BLOCK_SIZE = 256
SEED = args.seed
SAVE_DIR = Path(args.save_dir)
LOG_FILE = SAVE_DIR / "training.log"

if args.small:
    DIM = 192
    N_BLOCKS = 6
    GLU_HIDDEN = 512
    SEQ_LEN = 256
    BATCH_SIZE = args.batch or 128
    GRAD_ACCUM = 1
    LR_PEAK = args.lr or 4e-3
    CONFIG_NAME = "teeny-s"
else:
    DIM = 384
    N_BLOCKS = 8
    GLU_HIDDEN = 1024
    SEQ_LEN = 512
    BATCH_SIZE = args.batch or 128
    GRAD_ACCUM = 2
    LR_PEAK = args.lr or 3e-3
    CONFIG_NAME = "teeny-l"

LR_MIN = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_EPOCHS = args.epochs
TIME_LIMIT = args.hours * 3600 if args.hours else None
PATIENCE = 3
EVAL_EVERY = 1000
SAMPLE_EVERY = 2000
SAVE_EVERY = 5000
NUM_EVAL_BATCHES = 50
NUM_WORKERS = min(4, NUM_CORES)
VOCAB_BUILD_SAMPLES = 50000

# Progressive training: start with short sequences, gradually increase (buggy right now, will fix this later)
SEQ_LEN_SCHEDULE = [128, 256, 512]    # Epoch 0: 128, Epoch 1: 256, Epoch 2+: 512
CURRICULUM_EPOCHS_PER_STAGE = 1
MAX_SEQ_LEN = SEQ_LEN_SCHEDULE[-1]

def get_seq_len_for_epoch(epoch):
    stage = min(epoch // CURRICULUM_EPOCHS_PER_STAGE, len(SEQ_LEN_SCHEDULE) - 1)
    return SEQ_LEN_SCHEDULE[stage]

def get_current_curriculum_stage(epoch):
    return min(epoch // CURRICULUM_EPOCHS_PER_STAGE, len(SEQ_LEN_SCHEDULE) - 1)

# logging
def setup_logging():
    SAVE_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode='a'),
        ]
    )
    return logging.getLogger("teenylm")

class TeenyTokenizer:
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SPECIAL_TOKENS = 4

    def __init__(self, vocab_size=VOCAB_SIZE):
        import tiktoken
        self.base_enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = vocab_size
        self.id_to_base = {}
        self.base_to_id = {}
        self._built = False

    def build_from_texts(self, texts, max_texts=VOCAB_BUILD_SAMPLES):
        logger = logging.getLogger("teenylm")
        logger.info(f"Building vocab from {min(len(texts), max_texts)} texts...")
        counts = Counter()
        for i, text in enumerate(texts):
            if i >= max_texts:
                break
            counts.update(self.base_enc.encode(text))
        usable = self.vocab_size - self.SPECIAL_TOKENS
        most_common = counts.most_common(usable)
        for idx, (base_id, _) in enumerate(most_common):
            our_id = idx + self.SPECIAL_TOKENS
            self.id_to_base[our_id] = base_id
            self.base_to_id[base_id] = our_id
        self._built = True
        coverage = sum(c for _, c in most_common) / sum(counts.values())
        logger.info(f"Vocab: {self.vocab_size} tokens, coverage: {coverage*100:.1f}%")
        return self

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "id_to_base": {str(k): v for k, v in self.id_to_base.items()},
                "base_to_id": {str(k): v for k, v in self.base_to_id.items()},
            }, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.id_to_base = {int(k): v for k, v in data["id_to_base"].items()}
        self.base_to_id = {int(k): v for k, v in data["base_to_id"].items()}
        self._built = True
        return self

    def encode(self, text):
        return [self.base_to_id.get(t, self.UNK_ID) for t in self.base_enc.encode(text)]

    def decode(self, ids):
        base_ids = [self.id_to_base[t] for t in ids
                    if t >= self.SPECIAL_TOKENS and t in self.id_to_base]
        return self.base_enc.decode(base_ids)

# Dataset
class TinyStoriesDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = token_ids
        self.seq_len = seq_len
        self.n_sequences = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), \
               torch.tensor(chunk[1:], dtype=torch.long)

def load_and_tokenize(tokenizer, split="train"):
    from datasets import load_dataset
    logger = logging.getLogger("teenylm")
    logger.info(f"Loading TinyStories {split}...")
    ds = load_dataset("roneneldan/TinyStories", split=split)
    logger.info(f"Tokenizing {len(ds)} stories...")
    all_ids = []
    for i, sample in enumerate(ds):
        all_ids.extend(tokenizer.encode(sample["text"]))
        all_ids.append(tokenizer.EOS_ID)
        if (i + 1) % 100000 == 0:
            logger.info(f"  {i+1}/{len(ds)} ({len(all_ids)/1e6:.1f}M tokens)")
    logger.info(f"Done: {len(all_ids):,} tokens")
    return all_ids

# Model
def ternary_quantize(w, block_size=BLOCK_SIZE):
    if block_size <= 0 or w.numel() <= block_size:
        alpha = w.abs().mean()
        w_t = ((w / (alpha + 1e-8)).round().clamp(-1, 1)) * alpha
        return w + (w_t - w).detach()
    orig_shape = w.shape
    flat_w = w.flatten()
    numel = flat_w.numel()
    num_blocks = (numel + block_size - 1) // block_size
    pad_len = num_blocks * block_size - numel
    if pad_len > 0:
        flat_w = F.pad(flat_w, (0, pad_len))
    w_blocks = flat_w.view(num_blocks, block_size)
    alphas = w_blocks.abs().mean(dim=1, keepdim=True)
    w_t_blocks = ((w_blocks / (alphas + 1e-8)).round().clamp(-1, 1)) * alphas
    w_t = w_t_blocks.flatten()[:numel].view(orig_shape)
    return w + (w_t - w).detach()

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.norm = RMSNorm(in_features)
    def forward(self, x):
        return F.linear(self.norm(x), ternary_quantize(self.weight))

class GatedConvMixer(nn.Module):
    def __init__(self, dim, kernel_size=KERNEL_SIZE):
        super().__init__()
        self.up = BitLinear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size - 1, groups=dim, bias=False)
        self.down = BitLinear(dim, dim)
    def forward(self, x):
        B, T, D = x.shape
        gv = self.up(x)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        h = val.transpose(1, 2)
        h = self.conv(h)[:, :, :T]
        h = h.transpose(1, 2)
        return self.down(h * gate)

class TernaryGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.W_gate = BitLinear(dim, hidden)
        self.W_up = BitLinear(dim, hidden)
        self.W_down = BitLinear(hidden, dim)
    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))

class ImliBlock(nn.Module):
    def __init__(self, dim, kernel_size, glu_hidden):
        super().__init__()
        self.mixer = GatedConvMixer(dim, kernel_size)
        self.ffn = TernaryGLU(dim, glu_hidden)
    def forward(self, x):
        x = x + self.mixer(x)
        x = x + self.ffn(x)
        return x

class TeenyLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, n_blocks=N_BLOCKS,
                 kernel_size=KERNEL_SIZE, glu_hidden=GLU_HIDDEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            ImliBlock(dim, kernel_size, glu_hidden) for _ in range(n_blocks)
        ])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x, targets=None):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.final_norm(h))
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   targets.reshape(-1))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LR_PEAK * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(math.pi * progress))

# Eval
@torch.no_grad()
def evaluate(model, val_loader, max_batches=NUM_EVAL_BATCHES):
    model.eval()
    total_loss, count = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)

def generate_samples(model, tokenizer, max_tokens=150):
    model.eval()
    logger = logging.getLogger("teenylm")
    prompts = ["Once upon a time", "The little girl", "One day, a boy named"]
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=max_tokens)
        text = tokenizer.decode(out[0].tolist())
        logger.info(f"  > {text[:400]}")
        logger.info("")
    model.train()

def train():
    logger = setup_logging()
    SAVE_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    # Handle resume case
    start_epoch = 0
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            start_epoch = ckpt.get("epoch", 0)
        except:
            pass
    
    initial_seq_len = get_seq_len_for_epoch(start_epoch)
    
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(min(4, NUM_CORES))
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"  TeenyLM — {CONFIG_NAME}")
    logger.info(f"{'='*60}")
    logger.info(f"  CPU cores: {NUM_CORES}")
    logger.info(f"  RAM: {TOTAL_RAM_GB:.1f} GB")
    logger.info(f"  PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"  Config: d={DIM}, blocks={N_BLOCKS}, glu={GLU_HIDDEN}, "
                f"kernel={KERNEL_SIZE}")
    logger.info(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = "
                f"{BATCH_SIZE * GRAD_ACCUM} effective")
    logger.info(f"  Seq len: {initial_seq_len} (curriculum: {'→'.join(map(str, SEQ_LEN_SCHEDULE))}), LR: {LR_PEAK}")
    if TIME_LIMIT:
        logger.info(f"  Time limit: {args.hours}h")
    else:
        logger.info(f"  Max epochs: {MAX_EPOCHS}")
    logger.info(f"{'='*60}\n")

    tokenizer_path = SAVE_DIR / "tokenizer.json"
    tokenizer = TeenyTokenizer(VOCAB_SIZE)
    if tokenizer_path.exists():
        logger.info("Loading cached tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = [s["text"] for s in ds.select(range(min(VOCAB_BUILD_SAMPLES, len(ds))))]
        tokenizer.build_from_texts(texts)
        tokenizer.save(tokenizer_path)
        del texts, ds

    train_cache = SAVE_DIR / "train_tokens.pt"
    val_cache = SAVE_DIR / "val_tokens.pt"
    if train_cache.exists() and val_cache.exists():
        logger.info("Loading cached tokens...")
        train_ids = torch.load(train_cache, weights_only=False)
        val_ids = torch.load(val_cache, weights_only=False)
    else:
        train_ids = load_and_tokenize(tokenizer, "train")
        val_ids = load_and_tokenize(tokenizer, "validation")
        logger.info("Caching tokens to disk...")
        torch.save(train_ids, train_cache)
        torch.save(val_ids, val_cache)
    logger.info(f"Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    steps_per_epoch = 0  # Will be set after creating loaders
    if TIME_LIMIT:
        total_steps = steps_per_epoch * MAX_EPOCHS  # upper bound for LR schedule
    else:
        total_steps = steps_per_epoch * MAX_EPOCHS
    logger.info(f"Steps/epoch: {steps_per_epoch:,} | Max total: {total_steps:,}\n")

    model = TeenyLM(vocab_size=VOCAB_SIZE, dim=DIM, n_blocks=N_BLOCKS,
                      kernel_size=KERNEL_SIZE, glu_hidden=GLU_HIDDEN)
    total_p, train_p = model.count_params()
    logger.info(f"Model: {total_p:,} params ({total_p*4/1e6:.1f} MB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEAK,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    tokens_seen = 0
    curriculum_stage = 0
    current_seq_len = SEQ_LEN_SCHEDULE[0]

    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            tokens_seen = ckpt.get("tokens_seen", 0)
            # Restore curriculum state if available
            curriculum_stage = ckpt.get("curriculum_stage", 0)
            current_seq_len = ckpt.get("current_seq_len", SEQ_LEN_SCHEDULE[0])
            logger.info(f"  Resumed at step {global_step}, epoch {start_epoch}, "
                        f"best val loss {best_val_loss:.4f}")
            logger.info(f"  Restored curriculum stage: {curriculum_stage}, seq_len: {current_seq_len}")
        else:
            model.load_state_dict(ckpt, strict=False)
            logger.info(f"  Loaded weights only (no optimizer state)")

    if not args.resume or 'curriculum_stage' not in locals():
        current_seq_len = get_seq_len_for_epoch(start_epoch)
        curriculum_stage = get_current_curriculum_stage(start_epoch)
    
    def create_loaders_for_seq_len(seq_len):
        train_ds = TinyStoriesDataset(train_ids, seq_len)
        val_ds = TinyStoriesDataset(val_ids, seq_len)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
        return train_ld, val_ld, BATCH_SIZE
    
    train_loader, val_loader, current_batch_size = create_loaders_for_seq_len(current_seq_len)
    
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    if TIME_LIMIT:
        total_steps = steps_per_epoch * MAX_EPOCHS
    else:
        total_steps = steps_per_epoch * MAX_EPOCHS
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  Curriculum Learning Configuration")
    logger.info(f"{'='*60}")
    logger.info(f"  Starting seq_len: {current_seq_len}")
    logger.info(f"  Curriculum stage: {curriculum_stage}")
    logger.info(f"  Schedule: {SEQ_LEN_SCHEDULE}")
    logger.info(f"  Epochs per stage: {CURRICULUM_EPOCHS_PER_STAGE}")
    logger.info(f"  Current batch size: {current_batch_size}")
    logger.info(f"{'='*60}\n")

    model.train()
    patience_counter = 0
    t_start = time.time()
    last_log_time = t_start
    log_interval = 30

    logger.info("\nTraining started!\n")

    for epoch in range(start_epoch, MAX_EPOCHS):
        # ---- Curriculum Progression Check ----
        new_seq_len = get_seq_len_for_epoch(epoch)
        new_stage = get_current_curriculum_stage(epoch)
        
        if new_seq_len != current_seq_len:
            # Progress to next curriculum stage
            logger.info(f"\n{'='*60}")
            logger.info(f"  CURRICULUM PROGRESSION: Stage {curriculum_stage} → {new_stage}")
            logger.info(f"  Sequence length: {current_seq_len} → {new_seq_len}")
            logger.info(f"{'='*60}\n")
            
            current_seq_len = new_seq_len
            curriculum_stage = new_stage
            
            train_loader, val_loader, current_batch_size = create_loaders_for_seq_len(current_seq_len)
            
            steps_per_epoch = len(train_loader) // GRAD_ACCUM
            if TIME_LIMIT:
                total_steps = steps_per_epoch * MAX_EPOCHS
            else:
                total_steps = steps_per_epoch * MAX_EPOCHS
            
            logger.info(f"  New batch size: {current_batch_size}")
            logger.info(f"  New steps/epoch: {steps_per_epoch:,}")
            logger.info(f"  Total steps: {total_steps:,}\n")
        
        logger.info(f"--- Epoch {epoch+1}/{MAX_EPOCHS} (seq_len={current_seq_len}, batch={current_batch_size}) ---")
        optimizer.zero_grad()
        accum_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            if TIME_LIMIT and (time.time() - t_start) >= TIME_LIMIT:
                logger.info(f"\nTime limit reached ({args.hours}h)")
                break

            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                tokens_seen += BATCH_SIZE * GRAD_ACCUM * SEQ_LEN

                # Log
                now = time.time()
                if now - last_log_time >= log_interval:
                    elapsed = now - t_start
                    tok_s = tokens_seen / elapsed
                    eta = ""
                    if TIME_LIMIT:
                        remaining = TIME_LIMIT - elapsed
                        eta = f" | ETA {remaining/60:.0f}min"
                    logger.info(
                        f"  Step {global_step:>6d} | loss {accum_loss:.4f} | "
                        f"lr {lr:.6f} | {tok_s:.0f} tok/s | "
                        f"{tokens_seen/1e6:.1f}M tokens{eta}"
                    )
                    last_log_time = now
                accum_loss = 0.0

                # Eval
                if global_step % EVAL_EVERY == 0:
                    val_loss = evaluate(model, val_loader)
                    logger.info(f"\n  >>> Val loss: {val_loss:.4f} "
                                f"(best: {best_val_loss:.4f})\n")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(),
                                   SAVE_DIR / "teenylm_best.pt")
                        logger.info(f"  New best! Saved.")
                    else:
                        patience_counter += 1
                        logger.info(f"  No improvement ({patience_counter}/{PATIENCE})")
                    if patience_counter >= PATIENCE:
                        logger.info("Early stopping.")
                        break

                # Samples
                if global_step % SAMPLE_EVERY == 0:
                    logger.info("--- Samples ---")
                    generate_samples(model, tokenizer)

                # Checkpoint
                if global_step % SAVE_EVERY == 0:
                    ckpt_path = SAVE_DIR / f"teenylm_step_{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "tokens_seen": tokens_seen,
                        "curriculum_stage": curriculum_stage,
                        "current_seq_len": current_seq_len,
                        "config": {
                            "vocab_size": VOCAB_SIZE, "dim": DIM,
                            "n_blocks": N_BLOCKS, "kernel_size": KERNEL_SIZE,
                            "glu_hidden": GLU_HIDDEN, "seq_len": SEQ_LEN,
                            "seq_len_schedule": SEQ_LEN_SCHEDULE,
                        }
                    }, ckpt_path)
                    logger.info(f"  Checkpoint: {ckpt_path}")

        # Check breaks
        if TIME_LIMIT and (time.time() - t_start) >= TIME_LIMIT:
            break
        if patience_counter >= PATIENCE:
            break

        # End-of-epoch eval
        val_loss = evaluate(model, val_loader)
        logger.info(f"\n  Epoch {epoch+1} done | Val loss: {val_loss:.4f}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_DIR / "teenylm_best.pt")

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Config: {CONFIG_NAME}")
    logger.info(f"  Params: {total_p:,}")
    logger.info(f"  Steps: {global_step:,}")
    logger.info(f"  Tokens: {tokens_seen:,} ({tokens_seen/1e6:.1f}M)")
    logger.info(f"  Time: {elapsed/3600:.2f}h")
    logger.info(f"  Speed: {tokens_seen/elapsed:.0f} tok/s")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Saved: {SAVE_DIR / 'teenylm_best.pt'}")
    logger.info("=" * 60)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": global_step,
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "curriculum_stage": curriculum_stage,
        "current_seq_len": current_seq_len,
        "config": {
            "vocab_size": VOCAB_SIZE, "dim": DIM,
            "n_blocks": N_BLOCKS, "kernel_size": KERNEL_SIZE,
            "glu_hidden": GLU_HIDDEN, "seq_len": SEQ_LEN,
            "seq_len_schedule": SEQ_LEN_SCHEDULE,
        }
    }, SAVE_DIR / "teenylm_final.pt")

    logger.info("\n--- Final samples ---")
    generate_samples(model, tokenizer)
    logger.info("\nDone!")

if __name__ == "__main__":
    train()
