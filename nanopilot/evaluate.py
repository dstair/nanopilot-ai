#!/usr/bin/env python3
"""Evaluate Nanopilot PyTorch models against recorded tubs."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import TubDataset
from torch_model import LinearModel, ResidualBlock


AXES = ("Yaw", "Pitch", "Roll", "Altitude")
DEFAULT_BENCHMARK_TUBS = ("data/tub_209_26-05-09",)
BENCHMARK_ENV = "NANOPILOT_BENCHMARK_TUBS"


class LegacyImuFcLinearModel(torch.nn.Module):
    """LinearModel variant used by checkpoints saved before the IMU GRU change."""

    def __init__(self, input_shape=(3, 240, 320), imu_shape=(3, 6)):
        super().__init__()
        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 256, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.imu_fc = torch.nn.Sequential(
            torch.nn.Linear(imu_shape[0] * imu_shape[1], 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 32),
            torch.nn.GELU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 + 32, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 3),
        )

    def forward(self, img, imu, prev_ctrl=None):
        x = self.init_conv(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        imu_feat = self.imu_fc(torch.flatten(imu, 1))
        return self.fc(torch.cat((x, imu_feat), dim=1))


class LegacyImuGruLinearModel(torch.nn.Module):
    """LinearModel variant used by checkpoints after IMU GRU but before prev controls."""

    def __init__(self, input_shape=(3, 240, 320), imu_shape=(3, 6)):
        super().__init__()
        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 256, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.imu_hidden_dim = 32
        self.imu_gru = torch.nn.GRU(
            input_size=imu_shape[1],
            hidden_size=self.imu_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 + self.imu_hidden_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 3),
        )

    def forward(self, img, imu, prev_ctrl=None):
        x = self.init_conv(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        _, h_n = self.imu_gru(imu)
        imu_feat = h_n[-1]
        return self.fc(torch.cat((x, imu_feat), dim=1))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_corrcoef(preds, targets):
    correlations = []
    for i in range(targets.shape[1]):
        if np.std(preds[:, i]) == 0 or np.std(targets[:, i]) == 0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(preds[:, i], targets[:, i])[0, 1]))
    return np.asarray(correlations, dtype=np.float32)


def checkpoint_kind(state_dict):
    if any(k.startswith("imu_fc.") for k in state_dict):
        return "legacy_imu_fc"
    if any(k.startswith("ctrl_fc.") for k in state_dict) or any(
        k.startswith("fusion_attention.") for k in state_dict
    ):
        return "control_feedback"
    if any(k.startswith("imu_gru.") for k in state_dict):
        # Current vision+IMU model and the legacy GRU model share layer names;
        # they differ only in the output width (4 controls vs the old 3).
        out_dim = state_dict["fc.5.weight"].shape[0] if "fc.5.weight" in state_dict else 4
        return "legacy_imu_gru" if out_dim == 3 else "current"
    return "unknown"


def load_model(model_path, device, input_shape, imu_shape):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    kind = checkpoint_kind(state_dict)
    if kind == "legacy_imu_fc":
        model = LegacyImuFcLinearModel(input_shape=input_shape, imu_shape=imu_shape)
    elif kind == "legacy_imu_gru":
        model = LegacyImuGruLinearModel(input_shape=input_shape, imu_shape=imu_shape)
    elif kind == "control_feedback":
        raise RuntimeError(
            f"{model_path} is a retired control-feedback/attention checkpoint "
            "(has ctrl_fc/fusion_attention weights) and is not compatible with the "
            "current vision+IMU LinearModel. Retrain with torch_train.py."
        )
    else:
        model = LinearModel(input_shape=input_shape, imu_shape=imu_shape)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, kind


def parse_tub_list(value):
    return [t.strip() for t in value.split(",") if t.strip()]


def resolve_benchmark_tubs(args):
    if args.tubs:
        return parse_tub_list(args.tubs), "cli"
    if args.benchmark_tubs:
        return parse_tub_list(args.benchmark_tubs), "cli"
    env_tubs = os.getenv(BENCHMARK_ENV)
    if env_tubs:
        return parse_tub_list(env_tubs), f"env:{BENCHMARK_ENV}"
    return list(DEFAULT_BENCHMARK_TUBS), "default"


def parse_weights(value):
    try:
        weights = np.asarray([float(v.strip()) for v in value.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be comma-separated numbers") from exc
    if weights.shape != (4,):
        raise argparse.ArgumentTypeError("weights must contain exactly four values")
    if np.any(weights < 0) or np.sum(weights) == 0:
        raise argparse.ArgumentTypeError("weights must be nonnegative and not all zero")
    return weights


def weighted_score(mae, weights):
    return float(np.average(mae, weights=weights))


def evaluate_model(
    model_path,
    tub_paths,
    device,
    *,
    image_h=240,
    image_w=320,
    seq_len=3,
    batch_size=32,
):
    img_transform = transforms.Compose(
        [
            transforms.Resize((image_h, image_w)),
            transforms.ToTensor(),
        ]
    )
    dataset = TubDataset(tub_paths, seq_len=seq_len, transform=img_transform)
    if len(dataset) == 0:
        raise RuntimeError("No samples found in evaluation tubs")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, kind = load_model(
        model_path,
        device,
        input_shape=(3, image_h, image_w),
        imu_shape=(seq_len, 6),
    )

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, imus, targets in loader:
            outputs = model(images.to(device), imus.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mae = np.mean(np.abs(preds - targets), axis=0)
    rmse = np.sqrt(np.mean(np.square(preds - targets), axis=0))
    correlations = safe_corrcoef(preds, targets)
    jitter = np.mean(np.abs(np.diff(preds, axis=0)), axis=0) if len(preds) > 1 else np.zeros(len(AXES))
    return {
        "samples": len(dataset),
        "missing_imu_records": dataset.missing_imu_records,
        "checkpoint_kind": kind,
        "mae": mae,
        "rmse": rmse,
        "corr": correlations,
        "jitter": jitter,
    }


def printable_metrics(metrics, weights):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            result[key] = [float(v) if not np.isnan(v) else None for v in value]
        else:
            result[key] = value
    result["score"] = weighted_score(metrics["mae"], weights)
    return result


def print_single(label, metrics, weights):
    print(f"\n{label}")
    print(
        f"Samples: {metrics['samples']}  Missing IMU: {metrics['missing_imu_records']}  "
        f"Checkpoint: {metrics['checkpoint_kind']}  Score: {weighted_score(metrics['mae'], weights):.4f}"
    )
    print(f"{'Axis':<12} | {'MAE':>8} | {'RMSE':>8} | {'Corr':>8} | {'Jitter':>8}")
    print("-" * 58)
    for i, axis in enumerate(AXES):
        corr = metrics["corr"][i]
        corr_text = "nan" if np.isnan(corr) else f"{corr:.4f}"
        print(
            f"{axis:<12} | {metrics['mae'][i]:8.4f} | {metrics['rmse'][i]:8.4f} | "
            f"{corr_text:>8} | {metrics['jitter'][i]:8.4f}"
        )


def print_comparison(old_metrics, new_metrics, weights):
    old_score = weighted_score(old_metrics["mae"], weights)
    new_score = weighted_score(new_metrics["mae"], weights)
    print("\nComparison")
    print(f"{'Axis':<12} | {'Old MAE':>8} | {'New MAE':>8} | {'MAE Change':>10}")
    print("-" * 50)
    for i, axis in enumerate(AXES):
        old_mae = old_metrics["mae"][i]
        new_mae = new_metrics["mae"][i]
        change = ((old_mae - new_mae) / old_mae) * 100 if old_mae else float("nan")
        print(f"{axis:<12} | {old_mae:8.4f} | {new_mae:8.4f} | {change:+9.2f}%")
    score_change = ((old_score - new_score) / old_score) * 100 if old_score else float("nan")
    print("-" * 50)
    print(f"{'Weighted':<12} | {old_score:8.4f} | {new_score:8.4f} | {score_change:+9.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nanopilot .pth models")
    parser.add_argument(
        "--tubs",
        help=(
            "Comma-separated tub directories. If omitted, uses --benchmark-tubs, "
            f"${BENCHMARK_ENV}, or {','.join(DEFAULT_BENCHMARK_TUBS)}."
        ),
    )
    parser.add_argument("--benchmark-tubs", help="Comma-separated held-out benchmark tub directories")
    parser.add_argument("--model", help="Model path to evaluate")
    parser.add_argument("--old-model", help="Baseline model path")
    parser.add_argument("--new-model", help="Candidate model path")
    parser.add_argument("--image-h", type=int, default=120)
    parser.add_argument("--image-w", type=int, default=160)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=parse_weights("1,1,1,1"),
        help="Comma-separated MAE score weights for yaw,pitch,roll,altitude",
    )
    parser.add_argument("--json-output", help="Optional path for machine-readable metrics JSON")
    args = parser.parse_args()

    if not args.model and not (args.old_model and args.new_model):
        parser.error("provide --model or both --old-model and --new-model")

    tub_paths, tub_source = resolve_benchmark_tubs(args)
    device = get_device()
    print(f"Device: {device}")
    print(f"Benchmark ({tub_source}): {', '.join(tub_paths)}")
    print(f"Weights: {','.join(f'{w:g}' for w in args.weights)}")

    kwargs = {
        "image_h": args.image_h,
        "image_w": args.image_w,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
    }
    if args.model:
        label = os.path.basename(args.model)
        metrics = evaluate_model(args.model, tub_paths, device, **kwargs)
        print_single(label, metrics, args.weights)
        output = {"benchmark_tubs": tub_paths, "weights": args.weights.tolist(), "model": printable_metrics(metrics, args.weights)}
        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(output, f, indent=2)
        return

    old_metrics = evaluate_model(args.old_model, tub_paths, device, **kwargs)
    new_metrics = evaluate_model(args.new_model, tub_paths, device, **kwargs)
    print_single(f"Old: {args.old_model}", old_metrics, args.weights)
    print_single(f"New: {args.new_model}", new_metrics, args.weights)
    print_comparison(old_metrics, new_metrics, args.weights)
    output = {
        "benchmark_tubs": tub_paths,
        "weights": args.weights.tolist(),
        "old_model": printable_metrics(old_metrics, args.weights),
        "new_model": printable_metrics(new_metrics, args.weights),
    }
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
