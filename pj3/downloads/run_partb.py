"""Run every required CS180 Project 3 Part B experiment.

Examples
--------
python run_partb.py --group denoise
python run_partb.py --group time
python run_partb.py --group class
python run_partb.py --group no-scheduler

Each group checkpoints after every epoch, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from partb_models import (
    ClassConditionalFM,
    ClassConditionalUNet,
    TimeConditionalFM,
    TimeConditionalUNet,
    UnconditionalUNet,
    architecture_smoke_test,
)


SEED = 180
SIGMAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(images: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(images[:, None].float() / 255.0, labels)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )


def save_grid(
    images: torch.Tensor,
    path: Path,
    title: str,
    nrow: int,
    normalize: bool = False,
) -> None:
    images = images.detach().float().cpu()
    if normalize:
        lo = images.amin(dim=(1, 2, 3), keepdim=True)
        hi = images.amax(dim=(1, 2, 3), keepdim=True)
        images = (images - lo) / (hi - lo).clamp_min(1e-8)
    grid = make_grid(images.clamp(0, 1), nrow=nrow, padding=2, pad_value=0.15)
    rows = math.ceil(len(images) / nrow)
    fig, ax = plt.subplots(figsize=(max(7, nrow * 1.05), max(2.5, rows * 1.15)))
    ax.imshow(grid.permute(1, 2, 0), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss(losses: list[float], path: Path, title: str) -> None:
    array = np.asarray(losses, dtype=np.float64)
    window = min(100, len(array))
    smooth = np.convolve(array, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.plot(array, color="#73a9c2", alpha=0.28, linewidth=0.55, label="step")
    ax.plot(
        np.arange(window - 1, len(array)),
        smooth,
        color="#c46d39",
        linewidth=1.7,
        label=f"{window}-step mean",
    )
    ax.set(xlabel="Optimizer step", ylabel="Mean squared error", title=title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def checkpoint_path(out: Path, name: str, epoch: int) -> Path:
    return out / "checkpoints" / f"{name}_epoch{epoch}.pt"


def latest_checkpoint(out: Path, name: str, max_epoch: int) -> tuple[int, Path] | None:
    for epoch in range(max_epoch, 0, -1):
        path = checkpoint_path(out, name, epoch)
        if path.is_file():
            return epoch, path
    return None


def load_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    out: Path,
    name: str,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, list[float]]:
    found = latest_checkpoint(out, name, epochs)
    if not found:
        return 0, []
    epoch, path = found
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    print(f"Resuming {name} after epoch {epoch}: {path}")
    return epoch, list(state.get("losses", []))


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    out: Path,
    name: str,
    epoch: int,
    losses: list[float],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path = checkpoint_path(out, name, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "losses": losses,
        },
        path,
    )


def train_loop(
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epochs: int,
    out: Path,
    step_fn,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch_callback=None,
    accumulation_steps: int = 1,
) -> tuple[list[float], list[float], float]:
    start_epoch, losses = load_resume(model, optimizer, out, name, epochs, scheduler)
    epoch_losses = []
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        # Sampling callbacks switch the module to eval mode.  Restore training
        # mode explicitly at every epoch so BatchNorm keeps updating.
        model.train()
        running = 0.0
        count = 0
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            labels = labels.to(DEVICE, non_blocking=True)
            loss = step_fn(model, images, labels)
            (loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach())
            losses.append(value)
            running += value
            count += 1
        if scheduler is not None:
            scheduler.step()
        mean_loss = running / count
        epoch_losses.append(mean_loss)
        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{name}: epoch {epoch + 1}/{epochs}, loss={mean_loss:.6f}, "
            f"lr={lr:.6g}, {elapsed:.1f}s",
            flush=True,
        )
        save_checkpoint(model, optimizer, out, name, epoch + 1, losses, scheduler)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, model)
    return losses, epoch_losses, time.perf_counter() - started


def run_denoising(
    out: Path,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
) -> dict:
    group = out / "denoise"
    group.mkdir(parents=True, exist_ok=True)
    micro_batch = 64 if DEVICE.type == "cuda" else 128
    accumulation_steps = 256 // micro_batch
    loader = make_loader(train_images, train_labels, batch_size=micro_batch)

    sample = test_images[0:1, None].float() / 255.0
    noise_generator = torch.Generator().manual_seed(SEED)
    epsilon = torch.randn(sample.shape, generator=noise_generator)
    noise_levels = torch.cat([sample + sigma * epsilon for sigma in SIGMAS])
    save_grid(
        noise_levels,
        group / "noise_levels.png",
        "Forward noising process",
        nrow=7,
    )

    fixed_test = (test_images[:20, None].float() / 255.0).to(DEVICE)
    fixed_generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    fixed_eps = torch.randn(fixed_test.shape, generator=fixed_generator, device=DEVICE)

    model = UnconditionalUNet(1, 128).to(DEVICE, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    def sigma_step(net, images, _labels):
        noisy = images + 0.5 * torch.randn_like(images)
        return criterion(net(noisy), images)

    @torch.inference_mode()
    def sigma_callback(epoch, net):
        if epoch in (1, 5):
            net.eval()
            noisy = fixed_test + 0.5 * fixed_eps
            denoised = net(noisy.to(memory_format=torch.channels_last))
            paired = torch.stack((noisy, denoised), dim=1).flatten(0, 1)
            save_grid(
                paired,
                group / f"denoise_epoch{epoch}.png",
                f"Sigma=0.5: noisy input / denoised output after epoch {epoch}",
                nrow=10,
            )

    losses, epoch_losses, elapsed = train_loop(
        "denoise_sigma05",
        model,
        optimizer,
        loader,
        5,
        group,
        sigma_step,
        epoch_callback=sigma_callback,
        accumulation_steps=accumulation_steps,
    )
    save_loss(losses, group / "denoise_loss.png", "Single-step denoiser, sigma=0.5")

    model.eval()
    with torch.inference_mode():
        fixed_image = fixed_test[:1]
        fixed_noise = fixed_eps[:1]
        ood_inputs = torch.cat([fixed_image + sigma * fixed_noise for sigma in SIGMAS])
        ood_outputs = model(ood_inputs.to(memory_format=torch.channels_last))
        interleaved = torch.stack((ood_inputs, ood_outputs), dim=1).flatten(0, 1)
        save_grid(
            interleaved,
            group / "ood_noise_levels.png",
            "OOD noise levels: input / denoised output",
            nrow=7,
        )

    pure_model = UnconditionalUNet(1, 128).to(DEVICE, memory_format=torch.channels_last)
    pure_optimizer = torch.optim.Adam(pure_model.parameters(), lr=1e-4)
    pure_generator = torch.Generator(device=DEVICE).manual_seed(SEED + 2)
    pure_fixed = torch.randn(
        fixed_test.shape, generator=pure_generator, device=DEVICE
    )

    def pure_step(net, images, _labels):
        source = torch.randn_like(images)
        return criterion(net(source), images)

    @torch.inference_mode()
    def pure_callback(epoch, net):
        if epoch in (1, 5):
            net.eval()
            generated = net(pure_fixed.to(memory_format=torch.channels_last))
            save_grid(
                generated,
                group / f"pure_noise_epoch{epoch}.png",
                f"Single-step predictions from independent noise, epoch {epoch}",
                nrow=10,
            )

    pure_losses, pure_epoch_losses, pure_elapsed = train_loop(
        "denoise_pure_noise",
        pure_model,
        pure_optimizer,
        loader,
        5,
        group,
        pure_step,
        epoch_callback=pure_callback,
        accumulation_steps=accumulation_steps,
    )
    save_loss(
        pure_losses,
        group / "pure_noise_loss.png",
        "Single-step denoiser trained from independent noise",
    )
    return {
        "sigma_epoch_losses": epoch_losses,
        "pure_epoch_losses": pure_epoch_losses,
        "seconds": elapsed + pure_elapsed,
        "micro_batch": micro_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": micro_batch * accumulation_steps,
        "interpretation": (
            "Because the source noise is independent of the target digit, MSE rewards "
            "the conditional mean of the training distribution. The output therefore "
            "collapses toward a soft MNIST centroid rather than diverse digits."
        ),
    }


def run_time_conditioned(
    out: Path, train_images: torch.Tensor, train_labels: torch.Tensor
) -> dict:
    group = out / "time"
    group.mkdir(parents=True, exist_ok=True)
    loader = make_loader(train_images, train_labels, batch_size=64)
    model = TimeConditionalFM(TimeConditionalUNet(1, 10, 64), num_ts=50)
    model.to(DEVICE, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1.0 / 10)
    )

    def step(net, images, _labels):
        return net(images)

    @torch.inference_mode()
    def callback(epoch, net):
        if epoch in (1, 5, 10):
            samples = net.sample((28, 28), seed=SEED)
            save_grid(
                samples,
                group / f"time_epoch{epoch}.png",
                f"Time-conditioned samples, epoch {epoch}",
                nrow=10,
            )

    losses, epoch_losses, elapsed = train_loop(
        "time_conditioned",
        model,
        optimizer,
        loader,
        10,
        group,
        step,
        scheduler=scheduler,
        epoch_callback=callback,
    )
    save_loss(losses, group / "time_loss.png", "Time-conditioned flow matching")
    return {"epoch_losses": epoch_losses, "seconds": elapsed, "num_ts": 50}


def run_class_conditioned(
    out: Path,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    no_scheduler: bool,
) -> dict:
    group_name = "no_scheduler" if no_scheduler else "class"
    group = out / group_name
    group.mkdir(parents=True, exist_ok=True)
    loader = make_loader(train_images, train_labels, batch_size=64)
    model = ClassConditionalFM(
        ClassConditionalUNet(1, 10, 64), num_ts=300, p_uncond=0.1
    )
    model.to(DEVICE, memory_format=torch.channels_last)
    epochs = 15 if no_scheduler else 10
    learning_rate = 1e-3 if no_scheduler else 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if not no_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.1 ** (1.0 / 10)
        )

    def step(net, images, labels):
        return net(images, labels)

    labels = torch.arange(10).repeat_interleave(4)
    milestones = (1, 5, 10, 15) if no_scheduler else (1, 5, 10)

    @torch.inference_mode()
    def callback(epoch, net):
        if epoch in milestones:
            samples = net.sample(labels, guidance_scale=5.0, seed=SEED)
            save_grid(
                samples,
                group / f"{group_name}_epoch{epoch}.png",
                f"Class-conditioned CFG samples, epoch {epoch}",
                nrow=4,
            )

    losses, epoch_losses, elapsed = train_loop(
        group_name,
        model,
        optimizer,
        loader,
        epochs,
        group,
        step,
        scheduler=scheduler,
        epoch_callback=callback,
    )
    save_loss(
        losses,
        group / f"{group_name}_loss.png",
        "Class-conditioned flow matching"
        + (" without a scheduler" if no_scheduler else ""),
    )
    result = {
        "epoch_losses": epoch_losses,
        "seconds": elapsed,
        "num_ts": 300,
        "guidance_scale": 5.0,
        "p_uncond": 0.1,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "scheduler": None
        if no_scheduler
        else {"type": "ExponentialLR", "gamma": 0.1 ** (1.0 / 10)},
    }
    if no_scheduler:
        result["compensation"] = (
            "Replaced the decaying 1e-2 schedule with a conservative constant "
            "1e-3 learning rate and extended training from 10 to 15 epochs."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=("download", "denoise", "time", "class", "no-scheduler", "all"),
        default="all",
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/partb"))
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    seed_everything()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    print("Architecture smoke test:", architecture_smoke_test())
    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"torch={torch.__version__}, device={DEVICE}, name={device_name}, threads={args.threads}")

    train = MNIST(args.data, train=True, download=True, transform=ToTensor())
    test = MNIST(args.data, train=False, download=True, transform=ToTensor())
    if args.group == "download":
        print(f"MNIST ready: train={len(train)}, test={len(test)}")
        return

    metrics_path = args.out / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    metrics.update(
        {
            "seed": SEED,
            "torch": torch.__version__,
            "device": str(DEVICE),
            "device_name": device_name,
            "train_examples": len(train),
            "test_examples": len(test),
            "architecture": "official Fall 2026 UNet diagram",
        }
    )

    groups = (
        ("denoise", "time", "class", "no-scheduler")
        if args.group == "all"
        else (args.group,)
    )
    for group in groups:
        if group == "denoise":
            metrics[group] = run_denoising(
                args.out, train.data, train.targets, test.data
            )
        elif group == "time":
            metrics[group] = run_time_conditioned(
                args.out, train.data, train.targets
            )
        elif group == "class":
            metrics[group] = run_class_conditioned(
                args.out, train.data, train.targets, no_scheduler=False
            )
        elif group == "no-scheduler":
            metrics[group] = run_class_conditioned(
                args.out, train.data, train.targets, no_scheduler=True
            )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
