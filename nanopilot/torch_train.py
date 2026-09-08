#!/usr/bin/env python3
"""
Standalone PyTorch training script for Nanopilot CNN.

Reads tub data directly (catalog JSON + JPEG images), trains a PyTorch
LinearModel, and saves the best checkpoint as a .pth file.

Usage:
    python nanopilot/torch_train.py --tubs=data/tub_16_26-03-01 --model=models/pilot.pth
    python nanopilot/torch_train.py --tubs=data/tub_16_26-03-01,data/tub_17_26-03-01 --model=models/pilot.pth

    # Fine-tune an existing model on new data (fresh optimizer/epoch counter)
    python nanopilot/torch_train.py --tubs=data/new_tub --model=models/pilot_v2.pth \\
        --init-weights=models/pilot.pth --lr=0.0003

    # Resume an interrupted run (restores model + optimizer + epoch + best-val)
    python nanopilot/torch_train.py --tubs=data/tub_16_26-03-01 --model=models/pilot.pth \\
        --resume=models/pilot.ckpt

Options:
    --tubs           Comma-separated list of tub directories
    --model          Output .pth model path (bare state_dict, for deploy/inference)
    --myconfig       Config file override [default: drone_config_80mm.py]
    --init-weights   Warm-start model weights from a .pth or .ckpt (fine-tune)
    --resume         Resume full training state from a .ckpt (model+optimizer+epoch)
    --checkpoint     Override the .ckpt path [default: <model> with .ckpt extension]
    --no-checkpoint  Disable periodic .ckpt writing (still writes the best .pth)
    --lr             Override the learning rate
    --seed           Seed torch/numpy/random for reproducibility
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

# Training runs for many minutes and is routinely redirected to a log file, where
# Python block-buffers stdout: a run killed mid-training then loses its entire
# epoch history to the buffer (the log looks empty even though the .ckpt shows
# real progress). Line-buffer instead, so `python nanopilot/torch_train.py > log`
# behaves like `python -u` without every caller having to remember the flag.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

from dataset import TubDataset
from torch_model import LinearModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _atomic_save(obj, path):
    """Write to a temp file then os.replace, so an interrupt mid-write can't
    corrupt an existing checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _default_ckpt_path(model_path):
    base, _ = os.path.splitext(model_path)
    return base + ".ckpt"


def save_checkpoint(path, model, optimizer, epoch, best_val_loss,
                     epochs_no_improve, arch_meta):
    """Save full resumable training state (model + optimizer + counters)."""
    _atomic_save({
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "arch": arch_meta,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def _extract_model_state(loaded):
    """Accept either a bare state_dict (.pth) or a wrapped .ckpt dict."""
    if isinstance(loaded, dict) and "model_state" in loaded:
        return loaded["model_state"]
    return loaded


def load_weights_into(model, path, device):
    """Fine-tune warm-start: load weights only, with a clear error on
    architecture mismatch (e.g. a retired attention/control-feedback or
    3-output checkpoint)."""
    loaded = torch.load(path, map_location=device, weights_only=False)
    state = _extract_model_state(loaded)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(
            f"ERROR: weights in {path} are incompatible with the current "
            f"LinearModel (different architecture — e.g. a retired "
            f"attention/control-feedback or 3-output checkpoint). Details:\n{e}"
        )


def load_full_checkpoint(path, model, optimizer, device):
    """Resume: restore model + optimizer + counters (+ rng, best-effort).
    Returns (start_epoch, best_val_loss, epochs_no_improve)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ck, dict) and "model_state" in ck):
        raise SystemExit(f"ERROR: {path} is not a resumable .ckpt (no model_state).")
    model.load_state_dict(ck["model_state"], strict=True)
    optimizer.load_state_dict(ck["optimizer_state"])
    rng = ck.get("rng")
    if rng:
        try:
            torch.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception:
            pass  # rng restore is best-effort
    return ck["epoch"] + 1, ck["best_val_loss"], ck["epochs_no_improve"]


def train(
    cfg,
    tub_paths,
    model_path,
    max_epochs_override=None,
    max_samples=None,
    batch_size_override=None,
    train_split_override=None,
    print_model_summary_override=None,
    init_weights=None,
    resume=None,
    checkpoint_path=None,
    save_checkpoints=True,
    lr_override=None,
    seed=None,
):
    if init_weights and resume:
        print("ERROR: --init-weights and --resume are mutually exclusive.")
        return

    device = get_device()
    print(f"Device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Training/inference resolution is decoupled from the camera capture
    # resolution: the sim captures at IMAGE_W/H (640x480), but we downscale to a
    # small TRAIN_IMAGE_W/H for the CNN. Full-res frames make the data loader the
    # bottleneck and add nothing for a sim course. Keep this in sync with
    # TorchPilot's input_shape at runtime.
    image_h = getattr(cfg, 'TRAIN_IMAGE_H', 120)
    image_w = getattr(cfg, 'TRAIN_IMAGE_W', 160)
    image_d = getattr(cfg, 'IMAGE_DEPTH', 3)
    batch_size = batch_size_override or getattr(cfg, 'BATCH_SIZE', 128)
    max_epochs = max_epochs_override or getattr(cfg, 'MAX_EPOCHS', 100)
    lr = lr_override or getattr(cfg, 'LEARNING_RATE', 0.001)
    patience = getattr(cfg, 'EARLY_STOP_PATIENCE', 5)
    train_split = train_split_override or getattr(cfg, 'TRAIN_TEST_SPLIT', 0.8)
    seq_len = getattr(cfg, 'SEQUENCE_LENGTH', 7)

    # Define image transformations to match config resolution
    img_transform = transforms.Compose([
        transforms.Resize((image_h, image_w)),
        transforms.ToTensor(), # handles HWC -> CHW and [0, 1] scaling
    ])

    # Load dataset using the multi-modal TubDataset
    # Train the altitude axis in throttle-PWM space when configured. The raw
    # altitude command maps to PWM through sign(a)*|a|**DRONE_THROTTLE_SCALE,
    # a curve so steep at zero that raw-space regression error is uncorrelated
    # with thrust error. See TubDataset for the full reasoning. TorchPilot must
    # be constructed with the same value or the altitude axis is silently wrong.
    altitude_pwm_scale = (
        getattr(cfg, 'DRONE_THROTTLE_SCALE', None)
        if getattr(cfg, 'TRAIN_ALTITUDE_PWM_SPACE', False) else None
    )
    if altitude_pwm_scale:
        print(f"Altitude target: PWM space (scale={altitude_pwm_scale})")
    dataset = TubDataset(tub_paths, seq_len=seq_len, transform=img_transform,
                         altitude_pwm_scale=altitude_pwm_scale)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: No samples found. Check --tubs path.")
        return
    if dataset.missing_imu_records:
        print(
            "WARNING: "
            f"{dataset.missing_imu_records}/{len(dataset)} samples are missing "
            "imu/acl_* or imu/gyr_* fields; zeros will be used for those records."
        )
    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        if max_samples < 2:
            print("ERROR: --max-samples must be at least 2.")
            return
        last_idx = len(dataset) - 1
        if max_samples == 1:
            indices = [0]
        else:
            indices = [round(i * last_idx / (max_samples - 1)) for i in range(max_samples)]
        dataset = Subset(dataset, indices)
        print(f"Using deterministic subset: {len(dataset)} samples (--max-samples={max_samples})")

    # Train/val split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    if n_train == 0 or n_val == 0:
        print("ERROR: Need at least two samples for a train/val split.")
        return
    # Chronological Split: Train on the start of the logs, validate on the end.
    # This prevents temporal data leakage between adjacent frames.
    train_ds = Subset(dataset, range(0, n_train))
    val_ds = Subset(dataset, range(n_train, n_train + n_val))
    print(f"Train: {n_train}, Val: {n_val}")

    pin = device.type == 'cuda'
    # Parallel workers keep the (MPS/CUDA) device fed with decoded JPEGs instead
    # of decoding single-threaded on the main process.
    num_workers = getattr(cfg, 'TRAIN_NUM_WORKERS', 4)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin)
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Model
    model = LinearModel(
        input_shape=(image_d, image_h, image_w), 
        imu_shape=(seq_len, 6)
    ).to(device)
    if print_model_summary_override is None:
        print_model_summary = getattr(cfg, 'PRINT_MODEL_SUMMARY', True)
    else:
        print_model_summary = print_model_summary_override
    if print_model_summary:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # HuberLoss is more robust to outliers than MSE, reducing "twitchy" flight.
    # Per-axis weights on [yaw, pitch, roll, altitude]. Keep these near-uniform:
    # altitude in the current tubs is almost always ~0 (std ~0.02), so a large
    # altitude weight just makes the loss dominated by a trivially-predicted axis
    # and starves the axes that carry real visual signal (yaw/roll).
    loss_weights = torch.tensor(getattr(cfg, 'TRAIN_LOSS_WEIGHTS', [1.0, 1.0, 1.0, 1.0]), device=device)

    def criterion(pred, target):
        # Calculate per-element Huber loss
        loss = torch.nn.functional.huber_loss(pred, target, reduction='none', delta=1.0)
        # Multiply by weights and return mean
        return (loss * loss_weights).mean()

    arch_meta = {
        "altitude_pwm_scale": altitude_pwm_scale,
        "vel_dim": 3,
        "model_class": "LinearModel",
        "input_shape": [image_d, image_h, image_w],
        "seq_len": seq_len,
        "output_dim": 4,
    }

    # Training loop
    best_val_loss = float('inf')
    # Reference the patience counter measures against; tracked separately from
    # best_val_loss so a real-but-small improvement still saves weights.
    patience_ref = float('inf')
    epochs_no_improve = 0
    # Rolling window of recent val losses. The PATIENCE test runs on the median
    # of this window, never on a single epoch -- see the comment at the test.
    smooth_window = max(1, int(getattr(cfg, 'EARLY_STOP_SMOOTH_WINDOW', 3)))
    val_history = []
    start_epoch = 0
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    ckpt_path = checkpoint_path or _default_ckpt_path(model_path)
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)

    if resume:
        start_epoch, best_val_loss, epochs_no_improve = load_full_checkpoint(
            resume, model, optimizer, device)
        print(f"Resuming from {resume}: continuing at epoch {start_epoch + 1}, "
              f"best_val_loss={best_val_loss:.6f}")
        patience_ref = best_val_loss
    elif init_weights:
        load_weights_into(model, init_weights, device)
        print(f"Initialized weights from {init_weights} (fine-tune: fresh "
              f"optimizer, epoch counter reset)")

    if start_epoch >= max_epochs:
        print(f"Nothing to do: resumed at epoch {start_epoch + 1} but "
              f"--max-epochs={max_epochs}. Raise --max-epochs to continue training.")
        return

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for images, imus, vels, labels in train_loader:
            images, imus, vels, labels = (images.to(device), imus.to(device),
                                          vels.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(images, imus, vels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, imus, vels, labels in val_loader:
                images, imus, vels, labels = (images.to(device), imus.to(device),
                                              vels.to(device), labels.to(device))
                outputs = model(images, imus, vels)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= n_val

        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{max_epochs} — "
              f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f} "
              f"({elapsed:.1f}s)")

        # Keeping the BEST weights and deciding when to STOP are separate
        # questions, and conflating them silently deployed a worse model.
        #
        # The old code did both with `val_loss < best - MIN_DELTA`, where
        # MIN_DELTA is an ABSOLUTE 0.0005. That is fine at the ~0.01 losses of
        # human-piloted tubs, but an expert-distilled student reaches ~0.0003,
        # where the threshold is larger than the entire remaining loss and NO
        # improvement can ever clear it. Measured 2026-09-06 on tub_19: epochs
        # 5, 7, 9 and 10 all beat the running best and none were saved, so the
        # deployed model was epoch 8 (0.000654) rather than epoch 10
        # (0.000317), and patience then expired on those same real gains.
        #
        # So: save on any true improvement, and use a SCALE-AWARE threshold for
        # patience only, so early stopping behaves the same whether the loss
        # lives at 0.01 or 0.0003.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  ↳ Saved best model (val_loss: {val_loss:.6f})")

        # PATIENCE runs on the MEDIAN of the last `smooth_window` epochs, not on
        # this epoch's raw value. Validation loss on expert-distilled tubs swings
        # ~10x epoch to epoch while train_loss falls smoothly, so a single epoch
        # is a coin flip and the old raw test let ONE lucky early draw decide the
        # whole run. Measured 2026-09-07 on tub_22: epoch 2 drew 0.001186,
        # nothing beat it for 5 epochs, and training stopped at epoch 7 having
        # deployed the epoch-2 weights while train_loss was still falling. The
        # same code on the smaller slalom tub drew UNLUCKY early values, so
        # patience kept resetting and it ran 30 epochs to 0.000087 -- opposite
        # outcomes from noise alone. A median needs a MAJORITY of the window to
        # agree before it moves, so one outlier in either direction cannot end
        # a run or extend it.
        #
        # Best-weight SAVING above still uses the raw value and must keep doing
        # so: it is the 2026-08-24 fix, and smoothing it would mean deploying
        # weights from an epoch that was never actually the best.
        val_history.append(val_loss)
        if len(val_history) > smooth_window:
            val_history.pop(0)
        smoothed = statistics.median(val_history)

        if len(val_history) < smooth_window:
            # Not enough history to judge. Do NOT let these epochs seed
            # patience_ref -- that is exactly how one lucky early draw used to
            # capture the run.
            improved = True
            epochs_no_improve = 0
        else:
            if patience_ref == float('inf'):
                # First scored window: any finite loss is an improvement.
                # Computing the threshold here would give inf - inf = nan, and
                # every comparison against nan is False.
                min_delta = 0.0
            else:
                min_delta = max(
                    getattr(cfg, 'MIN_DELTA_ABS', 0.0),
                    abs(patience_ref) * getattr(cfg, 'MIN_DELTA_REL', 0.01),
                )
            improved = smoothed < patience_ref - min_delta
            if improved:
                patience_ref = smoothed
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if save_checkpoints:
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss,
                             epochs_no_improve, arch_meta)

        if not improved:
            if getattr(cfg, 'USE_EARLY_STOP', True) and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch + 1} epochs "
                      f"(median-of-{smooth_window} val has not improved for "
                      f"{patience}; last median {smoothed:.6f}, best {best_val_loss:.6f})")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch CNN for Nanopilot')
    parser.add_argument('--tubs', required=True, help='Comma-separated tub directories')
    parser.add_argument('--model', required=True, help='Output .pth model path')
    parser.add_argument('--myconfig', default='drone_config_80mm.py', help='Config file')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override MAX_EPOCHS')
    parser.add_argument('--max-samples', type=int, default=None, help='Use at most this many evenly spaced samples')
    parser.add_argument('--batch-size', type=int, default=None, help='Override BATCH_SIZE')
    parser.add_argument('--train-split', type=float, default=None, help='Override TRAIN_TEST_SPLIT, e.g. 0.8')
    parser.add_argument('--no-model-summary', action='store_true', help='Skip printing model layers')
    parser.add_argument('--init-weights', default=None,
                         help='Warm-start model weights from a .pth or .ckpt (fine-tune: fresh optimizer/epoch)')
    parser.add_argument('--resume', default=None,
                         help='Resume full training state (model+optimizer+epoch) from a .ckpt')
    parser.add_argument('--checkpoint', default=None,
                         help='Override the .ckpt path [default: <model> with .ckpt extension]')
    parser.add_argument('--no-checkpoint', action='store_true',
                         help='Disable periodic .ckpt writing (still writes the best .pth)')
    parser.add_argument('--lr', type=float, default=None, help='Override LEARNING_RATE')
    parser.add_argument('--seed', type=int, default=None, help='Seed torch/numpy/random for reproducibility')
    args = parser.parse_args()

    if args.init_weights and args.resume:
        parser.error('--init-weights and --resume are mutually exclusive')

    import donkeycar as dk
    cfg = dk.load_config(
        config_path=os.path.join(os.path.dirname(__file__), 'config.py'),
        myconfig=args.myconfig,
    )

    tub_paths = [t.strip() for t in args.tubs.split(',')]
    train(
        cfg,
        tub_paths,
        args.model,
        max_epochs_override=args.max_epochs,
        max_samples=args.max_samples,
        batch_size_override=args.batch_size,
        train_split_override=args.train_split,
        print_model_summary_override=False if args.no_model_summary else None,
        init_weights=args.init_weights,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        save_checkpoints=not args.no_checkpoint,
        lr_override=args.lr,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
