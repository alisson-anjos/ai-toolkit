"""Composable differentiable video objectives with optional frozen TorchScript evaluators.

ArcFace evaluator contract: RGB BCHW in [-1, 1], 112x112 face crops -> BxD.
Pose evaluator contract: RGB BCHW in [0, 1], 256x256 -> BxKxHxW heatmaps.
The pose model must expose differentiable heatmaps, not detected coordinates.
"""
import math
from pathlib import Path
import torch
import torch.nn.functional as F

KINDS = {'pixel_l1', 'arcface', 'pose_heatmap'}


def validate_auxiliary_losses(config):
    if config is None:
        return []
    if not isinstance(config, list):
        raise ValueError('auxiliary_losses must be a list')
    result, seen = [], set()
    for item in config:
        if not isinstance(item, dict) or item.get('type') not in KINDS:
            raise ValueError('Unknown H3 auxiliary loss; choose pixel_l1, arcface, or pose_heatmap')
        kind = item['type']
        if kind in seen:
            raise ValueError(f'Duplicate auxiliary loss: {kind}')
        seen.add(kind)
        weight = item.get('weight', 0.1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'{kind} weight must be finite and >= 0')
        if weight == 0:
            continue
        if kind != 'pixel_l1' and not Path(item.get('model_path', '')).is_file():
            raise ValueError(f'{kind} requires a local TorchScript model_path file')
        crop = item.get('crop', [0, 0, 1, 1])
        if len(crop) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in crop):
            raise ValueError('Auxiliary crop must be [left, top, right, bottom] fractions')
        left, top, right, bottom = crop
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError('Auxiliary crop must be inside [0, 1] with nonzero area')
        result.append({**item, 'weight': float(weight), 'crop': crop})
    return result


class VideoAuxiliaryLosses:
    def __init__(self, config, max_sigma=0.5, latent_frames=2, sample_frames=1):
        self.config = validate_auxiliary_losses(config)
        if not isinstance(max_sigma, (int, float)) or not math.isfinite(max_sigma) or not 0 < max_sigma <= 1:
            raise ValueError('auxiliary_loss_max_sigma must be in (0, 1]')
        if isinstance(latent_frames, bool) or not isinstance(latent_frames, int) or latent_frames < 0 or (latent_frames > 1 and (latent_frames - 2) % 5):
            raise ValueError('auxiliary_loss_latent_frames must be 0 (full), 1, or 5n+2')
        if isinstance(sample_frames, bool) or not isinstance(sample_frames, int) or sample_frames < 1:
            raise ValueError('auxiliary_loss_sample_frames must be >= 1')
        self.max_sigma, self.latent_frames, self.sample_frames = max_sigma, latent_frames, sample_frames
        self.models = {}

    def evaluator(self, spec, device):
        key = (spec['type'], spec['model_path'], str(device))
        if key not in self.models:
            model = torch.jit.load(spec['model_path'], map_location=device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models[key] = model
        return self.models[key]

    def __call__(self, prediction, noisy_latents, clean_latents, sigma, decode):
        if not self.config:
            return None, {}
        sigma = sigma.detach().float().reshape(prediction.shape[0], -1)[:, 0]
        active = sigma <= self.max_sigma
        if not bool(active.any()):
            return prediction.sum() * 0, {'loss/aux_active': 0.0}
        prediction, noisy_latents, clean_latents = prediction[active], noisy_latents[active], clean_latents[active]
        sigma = sigma[active].to(prediction.device).view(-1, 1, 1, 1, 1)
        x0 = noisy_latents.detach().float() - sigma * prediction.float()
        if self.latent_frames:
            x0 = x0[:, :, :self.latent_frames]
            clean_latents = clean_latents[:, :, :self.latent_frames]
        # The VAE is frozen, but the prediction decode must record gradients.
        predicted_video = decode(x0)
        with torch.no_grad():
            target_video = decode(clean_latents.detach())
        if not predicted_video.requires_grad and prediction.requires_grad:
            raise RuntimeError('Auxiliary losses require a differentiable VAE decode')
        count = min(self.sample_frames, predicted_video.shape[2])
        indices = torch.linspace(0, predicted_video.shape[2] - 1, count, device=predicted_video.device).round().long()
        def frames(video):
            return video[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, 3, video.shape[-2], video.shape[-1]).float()
        predicted, target = frames(predicted_video), frames(target_video).to(predicted_video.device)
        total = predicted.sum() * 0
        logs = {'loss/aux_active': float(active.float().mean())}
        for spec in self.config:
            kind = spec['type']
            left, top, right, bottom = spec['crop']
            h, w = predicted.shape[-2:]
            x0, y0 = int(left*w), int(top*h)
            x1, y1 = max(x0+1, int(right*w)), max(y0+1, int(bottom*h))
            p, t = predicted[:, :, y0:y1, x0:x1], target[:, :, y0:y1, x0:x1]
            if kind == 'pixel_l1':
                term = F.l1_loss(p, t)
            else:
                size = 112 if kind == 'arcface' else 256
                p = F.interpolate(p, (size, size), mode='bilinear', align_corners=False)
                t = F.interpolate(t, (size, size), mode='bilinear', align_corners=False)
                if kind == 'pose_heatmap':
                    p, t = (p + 1) / 2, (t + 1) / 2
                model = self.evaluator(spec, p.device)
                features = model(p)
                with torch.no_grad():
                    target_features = model(t)
                expected_rank = 2 if kind == 'arcface' else 4
                if not isinstance(features, torch.Tensor) or features.ndim != expected_rank:
                    raise ValueError(f'{kind} evaluator must return a rank-{expected_rank} tensor')
                if not features.requires_grad and p.requires_grad:
                    raise RuntimeError(f'{kind} evaluator detached its output; it cannot train the model')
                if kind == 'arcface':
                    term = (1 - F.cosine_similarity(features.float(), target_features.float(), dim=-1)).mean()
                else:
                    term = F.mse_loss(features.float(), target_features.float())
            if not torch.isfinite(term):
                raise RuntimeError(f'Non-finite auxiliary loss: {kind}')
            total = total + spec['weight'] * term
            logs[f'loss/aux_{kind}'] = float(term.detach())
            logs[f'loss/aux_{kind}_weighted'] = float((spec['weight'] * term).detach())
        # Keep low-sigma selection from increasing the effective batch weight.
        return total * active.float().mean().to(total.device), logs
