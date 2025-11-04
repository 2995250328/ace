#!/usr/bin/env python3
"""Compare intrinsic fusion strategies on a synthetic regression task."""

import argparse
import math
import statistics

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "PyTorch is required for this benchmark. Please install torch before running the comparison."
    ) from exc

from ace_network import IntrinsicFusion


def _build_synthetic_batch(batch_size, feature_dim, height, width, strength, device):
    features = torch.randn(batch_size, feature_dim, height, width, device=device)
    encoding = torch.randn_like(features)
    target = features + strength * encoding
    return features, encoding, target


def evaluate_mode(mode, *, steps, lr, feature_dim, height, width, batch_size, strength, seed, device):
    torch.manual_seed(seed)

    fusion = IntrinsicFusion(feature_dim, mode=mode)
    fusion.train()

    features, encoding, target = _build_synthetic_batch(batch_size, feature_dim, height, width, strength, device)

    params = [p for p in fusion.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr) if params else None

    for _ in range(steps):
        output = fusion(features, encoding)
        loss = F.mse_loss(output, target)
        if optimizer is None:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fusion.eval()
    with torch.no_grad():
        final = fusion(features, encoding)
        final_loss = F.mse_loss(final, target).item()

    return {
        "loss": final_loss,
        "psnr": -10.0 * math.log10(final_loss) if final_loss > 0 else float("inf"),
        "params": sum(p.numel() for p in fusion.parameters() if p.requires_grad),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='*', default=list(IntrinsicFusion.AVAILABLE_MODES),
                        help='fusion modes to benchmark')
    parser.add_argument('--steps', type=int, default=200, help='training steps per mode')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for trainable modes')
    parser.add_argument('--feature_dim', type=int, default=64, help='feature dimension used in the synthetic task')
    parser.add_argument('--height', type=int, default=8, help='height of the synthetic feature maps')
    parser.add_argument('--width', type=int, default=8, help='width of the synthetic feature maps')
    parser.add_argument('--batch_size', type=int, default=32, help='synthetic batch size')
    parser.add_argument('--strength', type=float, default=0.3, help='relative contribution of ray encodings in the synthetic target')
    parser.add_argument('--seed', type=int, default=0, help='base random seed')
    parser.add_argument('--num_seeds', type=int, default=3, help='number of runs for statistics')
    parser.add_argument('--device', type=str, default='cpu', help='device used for the synthetic experiment')
    args = parser.parse_args()

    results = {}
    for mode in args.modes:
        mode_losses = []
        mode_psnr = []
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            metrics = evaluate_mode(
                mode,
                steps=args.steps,
                lr=args.lr,
                feature_dim=args.feature_dim,
                height=args.height,
                width=args.width,
                batch_size=args.batch_size,
                strength=args.strength,
                seed=seed,
                device=args.device,
            )
            mode_losses.append(metrics['loss'])
            mode_psnr.append(metrics['psnr'])
            params = metrics['params']

        results[mode] = {
            'loss_mean': statistics.mean(mode_losses),
            'loss_std': statistics.pstdev(mode_losses) if len(mode_losses) > 1 else 0.0,
            'psnr_mean': statistics.mean(mode_psnr),
            'psnr_std': statistics.pstdev(mode_psnr) if len(mode_psnr) > 1 else 0.0,
            'params': params,
        }

    header = f"{'Mode':<15}{'Params':>10}{'Loss (mean±std)':>25}{'PSNR (dB)':>20}"
    print(header)
    print('-' * len(header))
    for mode, metrics in results.items():
        loss_mean = metrics['loss_mean']
        loss_std = metrics['loss_std']
        psnr_mean = metrics['psnr_mean']
        psnr_std = metrics['psnr_std']
        params = metrics['params']
        print(f"{mode:<15}{params:>10}{loss_mean:>15.6f}±{loss_std:<7.6f}{psnr_mean:>12.2f}±{psnr_std:<7.2f}")


if __name__ == '__main__':
    main()
