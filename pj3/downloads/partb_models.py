"""Networks and flow-matching algorithms for CS180 Project 3 Part B.

The tensor shapes and layer parameters follow the official Fall 2026 diagrams.
Time runs from noise at t=0 to data at t=1, and every flow model predicts
velocity x_1 - x_0.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AvgPool2d(kernel_size=7), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=7, stride=7, padding=0
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            Conv(in_channels, out_channels), Conv(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            DownConv(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            UpConv(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnconditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_hiddens: int = 128):
        super().__init__()
        d = num_hiddens
        self.init = ConvBlock(in_channels, d)
        self.down1 = DownBlock(d, d)
        self.down2 = DownBlock(d, 2 * d)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2 * d)
        self.up1 = UpBlock(4 * d, d)
        self.up2 = UpBlock(2 * d, d)
        self.final = nn.Sequential(
            ConvBlock(2 * d, d),
            nn.Conv2d(d, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        x0 = self.init(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        z = self.unflatten(self.flatten(x2))
        u1 = self.up1(torch.cat((x2, z), dim=1))
        u2 = self.up2(torch.cat((x1, u1), dim=1))
        return self.final(torch.cat((x0, u2), dim=1))


class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConditionalBackbone(nn.Module):
    """Shared official UNet backbone with two conditioning injection points."""

    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()
        d = num_hiddens
        self.d = d
        self.init = ConvBlock(in_channels, d)
        self.down1 = DownBlock(d, d)
        self.down2 = DownBlock(d, 2 * d)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2 * d)
        self.up1 = UpBlock(4 * d, d)
        self.up2 = UpBlock(2 * d, d)
        self.final = nn.Sequential(
            ConvBlock(2 * d, d),
            nn.Conv2d(d, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.init(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        z = self.unflatten(self.flatten(x2))
        return x0, x1, x2, z

    def decode(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        z: torch.Tensor,
        second_condition: torch.Tensor,
    ) -> torch.Tensor:
        u1 = self.up1(torch.cat((x2, z), dim=1))
        u1 = u1 * second_condition
        u2 = self.up2(torch.cat((x1, u1), dim=1))
        return self.final(torch.cat((x0, u2), dim=1))


class TimeConditionalUNet(nn.Module):
    def __init__(
        self, in_channels: int = 1, num_classes: int = 10, num_hiddens: int = 64
    ):
        super().__init__()
        del num_classes  # Kept for compatibility with the official starter signature.
        d = num_hiddens
        self.backbone = _ConditionalBackbone(in_channels, d)
        self.fc1_t = FCBlock(1, 2 * d)
        self.fc2_t = FCBlock(1, d)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        t = t.float().reshape(-1, 1)
        t1 = self.fc1_t(t).unsqueeze(-1).unsqueeze(-1)
        t2 = self.fc2_t(t).unsqueeze(-1).unsqueeze(-1)
        x0, x1, x2, z = self.backbone.encode(x)
        z = z * t1
        return self.backbone.decode(x0, x1, x2, z, t2)


class ClassConditionalUNet(nn.Module):
    def __init__(
        self, in_channels: int = 1, num_classes: int = 10, num_hiddens: int = 64
    ):
        super().__init__()
        d = num_hiddens
        self.num_classes = num_classes
        self.backbone = _ConditionalBackbone(in_channels, d)
        self.fc1_t = FCBlock(1, 2 * d)
        self.fc2_t = FCBlock(1, d)
        self.fc1_c = FCBlock(num_classes, 2 * d)
        self.fc2_c = FCBlock(num_classes, d)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        t = t.float().reshape(-1, 1)
        one_hot = F.one_hot(c.long(), self.num_classes).float()
        if mask is not None:
            one_hot = one_hot * mask.float().reshape(-1, 1)

        t1 = self.fc1_t(t).unsqueeze(-1).unsqueeze(-1)
        t2 = self.fc2_t(t).unsqueeze(-1).unsqueeze(-1)
        c1 = self.fc1_c(one_hot).unsqueeze(-1).unsqueeze(-1)
        c2 = self.fc2_c(one_hot).unsqueeze(-1).unsqueeze(-1)

        x0, x1, x2, z = self.backbone.encode(x)
        z = c1 * z + t1
        u1 = self.backbone.up1(torch.cat((x2, z), dim=1))
        u1 = c2 * u1 + t2
        u2 = self.backbone.up2(torch.cat((x1, u1), dim=1))
        return self.backbone.final(torch.cat((x0, u2), dim=1))


def time_fm_forward(
    unet: TimeConditionalUNet, x_1: torch.Tensor, num_ts: int
) -> torch.Tensor:
    unet.train()
    del num_ts  # The training algorithm samples continuous t ~ Uniform([0, 1]).
    batch = x_1.shape[0]
    x_0 = torch.randn_like(x_1)
    t = torch.rand(batch, device=x_1.device)
    t_image = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_image) * x_0 + t_image * x_1
    target = x_1 - x_0
    return F.mse_loss(unet(x_t, t), target)


@torch.inference_mode()
def time_fm_sample(
    unet: TimeConditionalUNet,
    img_wh: tuple[int, int],
    num_ts: int,
    seed: int = 0,
    num_samples: int = 40,
) -> torch.Tensor:
    unet.eval()
    device = next(unet.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((num_samples, 1, *img_wh), generator=generator, device=device)
    step = 1.0 / num_ts
    for index in range(num_ts):
        t = torch.full((num_samples,), index / num_ts, device=device)
        x = x + step * unet(x, t)
    return x


def class_fm_forward(
    unet: ClassConditionalUNet,
    x_1: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
    num_ts: int,
) -> torch.Tensor:
    unet.train()
    del num_ts  # num_ts controls Euler sampling, not the training distribution.
    batch = x_1.shape[0]
    x_0 = torch.randn_like(x_1)
    t = torch.rand(batch, device=x_1.device)
    t_image = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_image) * x_0 + t_image * x_1
    target = x_1 - x_0
    keep_mask = (torch.rand(batch, device=x_1.device) >= p_uncond).float()
    return F.mse_loss(unet(x_t, c, t, keep_mask), target)


@torch.inference_mode()
def class_fm_sample(
    unet: ClassConditionalUNet,
    c: torch.Tensor,
    img_wh: tuple[int, int],
    num_ts: int,
    guidance_scale: float = 5.0,
    seed: int = 0,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    unet.eval()
    device = next(unet.parameters()).device
    c = c.to(device)
    batch = len(c)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((batch, 1, *img_wh), generator=generator, device=device)
    step = 1.0 / num_ts
    cache = []
    keep = torch.ones(batch, device=device)
    drop = torch.zeros(batch, device=device)
    cache_every = max(1, num_ts // 10)
    for index in range(num_ts):
        t = torch.full((batch,), index / num_ts, device=device)
        u_uncond = unet(x, c, t, drop)
        u_cond = unet(x, c, t, keep)
        velocity = u_uncond + guidance_scale * (u_cond - u_uncond)
        x = x + step * velocity
        if return_cache and (index % cache_every == 0 or index == num_ts - 1):
            cache.append(x.detach().cpu())
    if return_cache:
        return x, torch.stack(cache, dim=1)
    return x


class TimeConditionalFM(nn.Module):
    def __init__(self, unet: TimeConditionalUNet, num_ts: int = 50):
        super().__init__()
        self.unet = unet
        self.num_ts = num_ts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return time_fm_forward(self.unet, x, self.num_ts)

    def sample(self, img_wh: tuple[int, int], seed: int = 0) -> torch.Tensor:
        return time_fm_sample(self.unet, img_wh, self.num_ts, seed)


class ClassConditionalFM(nn.Module):
    def __init__(
        self,
        unet: ClassConditionalUNet,
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.num_ts = num_ts
        self.p_uncond = p_uncond

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return class_fm_forward(self.unet, x, c, self.p_uncond, self.num_ts)

    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int] = (28, 28),
        guidance_scale: float = 5.0,
        seed: int = 0,
    ) -> torch.Tensor:
        return class_fm_sample(
            self.unet, c, img_wh, self.num_ts, guidance_scale, seed
        )


def architecture_smoke_test() -> dict[str, tuple[int, ...]]:
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0.25, 0.75])
    c = torch.tensor([2, 7])
    outputs = {
        "unconditional": tuple(UnconditionalUNet(1, 8)(x).shape),
        "time": tuple(TimeConditionalUNet(1, 10, 8)(x, t).shape),
        "class": tuple(ClassConditionalUNet(1, 10, 8)(x, c, t).shape),
    }
    expected = (2, 1, 28, 28)
    if any(shape != expected for shape in outputs.values()):
        raise AssertionError(outputs)
    return outputs


if __name__ == "__main__":
    print(architecture_smoke_test())
