import math
import random
import torch
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.h3_restoration_slider import residual_target


class H3RestorationSliderTrainer(DiffusionTrainer):
    """One frozen baseline + one student pass, with identical guide/noise/text.

    Only this new extension enables the objective. Ordinary H3 trainers and
    live runs are unaffected. Outer SDTrainer owns backward/accumulation.
    """
    def __init__(self, process_id, job, config, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        if not self.model_config.arch.startswith('minimax_h3'):
            raise ValueError('h3_restoration_slider requires MiniMax H3')
        cfg = self.config.get('slider', {})
        self.slider_min = float(cfg.get('min_scale', 0.25))
        self.slider_cap = float(cfg.get('residual_cap', 1.0))
        self.slider_anchor = float(cfg.get('anchor_probability', 0.1))
        if not (0 < self.slider_min <= 1 and 0 <= self.slider_anchor <= 1
                and math.isfinite(self.slider_cap) and self.slider_cap > 0):
            raise ValueError('Invalid restoration slider bounds')
        if self.train_config.do_guidance_loss:
            raise ValueError('Disable do_guidance_loss for supervised restoration slider')
        self.do_guided_loss = True

    def get_guided_loss(self, noisy_latents, conditional_embeds, timesteps,
                        pred_kwargs, batch, noise, **kwargs):
        if self.network is None:
            raise ValueError('Restoration slider needs a LoRA network')
        if batch.control_tensor_list is None and batch.control_tensor is None:
            # Cached H3 refs can be carried directly on the file items.
            if not all(getattr(item, 'control_video_paths', None) for item in batch.file_items):
                raise ValueError('Every slider target needs its matched degraded guide')
        if getattr(batch, 'audio_latents', None) is not None:
            raise ValueError('Pilot slider is video-only; disable audio')
        active = self.network.is_active
        was_training = self.sd.unet.training
        try:
            self.network.is_active = False
            self.sd.unet.eval()
            with torch.no_grad():
                base = self.sd.predict_noise(
                    latents=noisy_latents, conditional_embeddings=conditional_embeds,
                    timestep=timesteps, guidance_scale=1.0,
                    guidance_embedding_scale=1.0, batch=batch, **pred_kwargs)
        finally:
            self.network.is_active = active
            self.sd.unet.train(was_training)
        clean_flow = self.sd.get_loss_target(noise=noise, batch=batch)
        # Anchors train zero *effect at positive scale*, rather than disabling
        # LoRA (which would give zero useful preservation gradients).
        anchor = random.random() < self.slider_anchor
        scale = random.uniform(self.slider_min, 1.0)
        target = residual_target(base, clean_flow, 0.0 if anchor else scale,
                                 self.slider_cap)
        self.network.set_multiplier(scale)
        prediction = self.sd.predict_noise(
            latents=noisy_latents, conditional_embeddings=conditional_embeds,
            timestep=timesteps, guidance_scale=1.0,
            guidance_embedding_scale=1.0, batch=batch, **pred_kwargs)
        loss = torch.nn.functional.mse_loss(prediction.float(), target)
        self.additional_logs['slider/scale'] = scale
        self.additional_logs['slider/anchor'] = float(anchor)
        return loss

    def hook_train_loop(self, batch):
        try:
            return super().hook_train_loop(batch)
        finally:
            # Keep the chosen scale alive through checkpoint recomputation and
            # backward; restore only after the complete optimizer step.
            if self.network is not None:
                self.network.set_multiplier(1.0)
