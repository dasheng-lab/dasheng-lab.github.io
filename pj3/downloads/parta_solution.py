"""Completed Part A algorithms for the official PixNerd notebook.

This module is intentionally model-agnostic: the expensive PixNerd adapter is
provided by the official notebook as ``predict_velocity``. All integration states
remain float32 and all sampling runs under inference mode.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torchvision.transforms.functional import gaussian_blur


Tensor = torch.Tensor
VelocityFn = Callable[[Tensor, float], Tensor]


def interpolate(image: Tensor, noise: Tensor, t: float) -> Tensor:
    """Optimal-transport interpolation: t=0 noise, t=1 clean image."""
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    return (1.0 - float(t)) * noise + float(t) * image


@torch.inference_mode()
def one_step_denoise(
    x: Tensor,
    t: float,
    condition: Tensor,
    predict_velocity,
) -> Tensor:
    """Extrapolate once from x_t to the clean endpoint."""
    return x + (1.0 - float(t)) * predict_velocity(x, t, condition)


@torch.inference_mode()
def euler_sample(
    x_start: Tensor,
    t_start: float,
    step_size: float,
    velocity: VelocityFn,
) -> tuple[Tensor, list[tuple[float, Tensor]]]:
    """Integrate dx/dt=v(x,t) to exactly t=1 with explicit Euler steps.

    The trajectory stores the initial state, every fifth update, and the final
    state. The last integration step is shortened when needed.
    """
    if not 0.0 <= float(t_start) <= 1.0:
        raise ValueError("t_start must lie in [0, 1]")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    x = x_start.detach().float().clone()
    t = float(t_start)
    trajectory = [(t, x.detach().clone())]
    update = 0
    while t < 1.0 - 1e-10:
        h = min(float(step_size), 1.0 - t)
        x = x + h * velocity(x, t)
        t = min(1.0, t + h)
        update += 1
        if update % 5 == 0 or t >= 1.0 - 1e-10:
            trajectory.append((1.0 if t >= 1.0 - 1e-10 else t, x.detach().clone()))
    return x, trajectory


@torch.inference_mode()
def cfg_velocity(
    x: Tensor,
    t: float,
    condition: Tensor,
    w: float,
    predict_velocity,
    null_condition: Tensor,
) -> Tensor:
    """Part A CFG convention: v_c + w(v_c-v_u)."""
    v_cond = predict_velocity(x, t, condition)
    if float(w) == 0.0:
        return v_cond
    v_uncond = predict_velocity(x, t, null_condition)
    return v_cond + float(w) * (v_cond - v_uncond)


@torch.inference_mode()
def edit(
    image: Tensor,
    noise: Tensor,
    t_start: float,
    condition: Tensor,
    predict_velocity,
    null_condition: Tensor,
    w: float = 7.0,
    step_size: float = 0.02,
) -> Tensor:
    x_start = interpolate(image, noise, t_start)
    result, _ = euler_sample(
        x_start,
        t_start,
        step_size,
        lambda x, t: cfg_velocity(
            x, t, condition, w, predict_velocity, null_condition
        ),
    )
    return result


def rotate180(x: Tensor) -> Tensor:
    return torch.flip(x, (-2, -1))


@torch.inference_mode()
def visual_anagrams(
    initial_noise: Tensor,
    condition_a: Tensor,
    condition_b: Tensor,
    predict_velocity,
    null_condition: Tensor,
    w: float = 7.0,
    step_size: float = 0.02,
) -> Tensor:
    def velocity(x: Tensor, t: float) -> Tensor:
        v_a = cfg_velocity(x, t, condition_a, w, predict_velocity, null_condition)
        flipped = rotate180(x)
        v_b = cfg_velocity(
            flipped, t, condition_b, w, predict_velocity, null_condition
        )
        return 0.5 * (v_a + rotate180(v_b))

    result, _ = euler_sample(initial_noise, 0.0, step_size, velocity)
    return result


def lowpass(x: Tensor, sigma: float = 16.0) -> Tensor:
    """Assignment-recommended Gaussian: kernel 265, sigma 16 at 512px."""
    kernel = 2 * round(16.5 * x.shape[-1] / 64) + 1
    return gaussian_blur(x, [kernel, kernel], [sigma, sigma])


@torch.inference_mode()
def make_hybrids(
    initial_noise: Tensor,
    condition_low: Tensor,
    condition_high: Tensor,
    predict_velocity,
    null_condition: Tensor,
    w: float = 7.0,
    step_size: float = 0.02,
) -> Tensor:
    def velocity(x: Tensor, t: float) -> Tensor:
        v_low = cfg_velocity(
            x, t, condition_low, w, predict_velocity, null_condition
        )
        v_high = cfg_velocity(
            x, t, condition_high, w, predict_velocity, null_condition
        )
        return lowpass(v_low) + (v_high - lowpass(v_high))

    result, _ = euler_sample(initial_noise, 0.0, step_size, velocity)
    return result


@torch.inference_mode()
def velocity_diagnostics(
    x_t: Tensor,
    t: float,
    clean_image: Tensor,
    epsilon: Tensor,
    condition: Tensor,
    predict_velocity,
) -> tuple[Tensor, Tensor, Tensor]:
    predicted_velocity = predict_velocity(x_t, t, condition)
    ground_truth_velocity = clean_image - epsilon
    predicted_update = (1.0 - float(t)) * predicted_velocity
    target_update = (1.0 - float(t)) * ground_truth_velocity
    error_map = ((predicted_update - target_update).square().mean(dim=1)).sqrt()
    return predicted_update, target_update, error_map


def smoke_test() -> None:
    image = torch.full((1, 3, 4, 4), 2.0)
    noise = torch.zeros_like(image)
    assert torch.equal(interpolate(image, noise, 0), noise)
    assert torch.equal(interpolate(image, noise, 1), image)

    calls = []

    def constant(x: Tensor, t: float) -> Tensor:
        calls.append(t)
        return torch.ones_like(x)

    result, states = euler_sample(noise, 0.93, 0.02, constant)
    assert len(calls) == 4 and states[-1][0] == 1.0
    assert torch.allclose(result, torch.full_like(noise, 0.07), atol=1e-6)
    assert torch.equal(rotate180(rotate180(image)), image)

    def conditional(x: Tensor, _t: float, c: Tensor) -> Tensor:
        return torch.ones_like(x) * float(c)

    guided = cfg_velocity(noise, 0.5, torch.tensor(2.0), 0, conditional, torch.tensor(1.0))
    assert torch.equal(guided, torch.full_like(noise, 2.0))


if __name__ == "__main__":
    smoke_test()
    print("Part A algorithm smoke tests passed.")
