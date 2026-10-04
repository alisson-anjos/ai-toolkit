'use client';
import type { JobConfig } from '@/types';
import { FormGroup, NumberInput, SelectInput, TextInput, Checkbox } from '@/components/formInputs';

// Online RL for MiniMax H3 LoRAs (toolkit/bfs_nft.py) and the scene loss (SDTrainer, train.scene_loss).
// See docs/H3_RL_TRAINING.md.
type Props = { jobConfig: JobConfig; setJobConfig: (value: any, key: string) => void };

export const NFT_DEFAULTS = {
  group: 6,
  steps: 10,
  train_fraction: 0.3,
  mix_beta: 0.1,
  adv_clip_max: 5.0,
  ref_kl_coef: 1e-4,
  adaptive_weight_min: 1e-5,
  decay_schedule: 'delayed_linear_to_0_999',
  update_interval: 2,
  weights: { id: 1.0, char: 1.0, bg: 1.0, light: 1.0, pose: 1.0, lips: 1.0 },
  src_datasets: [] as string[],
  reward_python: '',
  reward_models: {} as Record<string, string>,
  keep_rollouts: 2,
};

const SCENE_DEFAULTS = { person_weight: 2.0, bg_weight: 1.5, lowfreq_weight: 0.5, lowfreq_kernel: 4 };

const REWARDS: { key: keyof typeof NFT_DEFAULTS.weights; label: string; hint: string }[] = [
  { key: 'id', label: 'Identity (ArcFace)', hint: 'face of the result vs the reference (or the real target)' },
  { key: 'char', label: 'Character (DINOv2)', hint: 'whole-subject look vs the reference: works from behind, for anime and creatures' },
  { key: 'bg', label: 'Background', hint: 'PSNR outside the person mask vs the input / target' },
  { key: 'light', label: 'Lighting', hint: 'low-frequency colour (target) or face shading vs the original face' },
  { key: 'pose', label: 'Pose', hint: 'body keypoint similarity (YOLOv8-pose) frame by frame' },
  { key: 'lips', label: 'Lip sync', hint: 'mouth-opening curve correlation; neutral when nobody speaks' },
];

export default function H3RLOptions({ jobConfig, setJobConfig }: Props) {
  const train: any = jobConfig.config.process[0].train;
  const nft = train.bfs_nft as typeof NFT_DEFAULTS | undefined;
  const scene = train.scene_loss as typeof SCENE_DEFAULTS | undefined;
  const set = (key: string, value: any) => setJobConfig(value, `config.process[0].train.bfs_nft.${key}`);
  const setScene = (key: string, value: any) => setJobConfig(value, `config.process[0].train.scene_loss.${key}`);
  const models = nft?.reward_models ?? {};

  return (
    <div className="space-y-4 pt-1">
      <FormGroup label="Online RL (DiffusionNFT)">
        <Checkbox
          label="Train with rewards instead of the regression loss"
          checked={!!nft}
          onChange={value => setJobConfig(value ? { ...NFT_DEFAULTS } : undefined, 'config.process[0].train.bfs_nft')}
        />
        <p className="text-sm text-gray-400 mt-2">
          Each step generates a group of videos for one dataset item with the EMA ("old") copy of the LoRA, scores them,
          and pulls the LoRA toward the above-average samples and away from the below-average ones (DiffusionNFT, loss
          adapted from verl-omni). The model never regresses on the target. Start from a LoRA that already does the task
          (Network → pretrained LoRA path); use batch size 1. Disable sampling: rollouts are saved every 25 steps in
          output/&lt;name&gt;/nft_rollouts.
        </p>
      </FormGroup>

      {nft && (
        <>
          <FormGroup label="Rollouts & update">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              <NumberInput label="Group size (videos per item)" min={2} value={nft.group ?? 6}
                onChange={v => set('group', v ?? 6)} />
              <NumberInput label="Rollout sampling steps" min={2} value={nft.steps ?? 10}
                onChange={v => set('steps', v ?? 10)} />
              <NumberInput label="Trained timestep fraction" min={0.05} max={1} value={nft.train_fraction ?? 0.3}
                onChange={v => set('train_fraction', v ?? 0.3)} />
              <NumberInput label="Rollouts kept in the video (best/worst)" min={0} max={2} value={nft.keep_rollouts ?? 2}
                onChange={v => set('keep_rollouts', v ?? 2)} />
              <NumberInput label="Mix beta" min={0.001} max={1} value={nft.mix_beta ?? 0.1}
                onChange={v => set('mix_beta', v ?? 0.1)} />
              <NumberInput label="Advantage clip" min={0.1} value={nft.adv_clip_max ?? 5}
                onChange={v => set('adv_clip_max', v ?? 5)} />
              <NumberInput label="KL to base (LoRA off)" min={0} value={nft.ref_kl_coef ?? 1e-4}
                onChange={v => set('ref_kl_coef', v ?? 0)} />
              <NumberInput label="Old-policy update interval" min={1} value={nft.update_interval ?? 2}
                onChange={v => set('update_interval', v ?? 2)} />
              <SelectInput label="Old-policy EMA schedule" value={nft.decay_schedule ?? 'delayed_linear_to_0_999'}
                options={[
                  { value: 'delayed_linear_to_0_999', label: 'Delayed linear → 0.999 (verl-omni H3)' },
                  { value: 'linear_to_0_5', label: 'Linear → 0.5' },
                  { value: 'copy', label: 'Copy (no EMA)' },
                ]}
                onChange={v => set('decay_schedule', v)} />
            </div>
            <p className="text-sm text-gray-400 mt-2">
              Cost per step ≈ group × rollout steps forwards + group × fraction × steps × 3 forwards (one with grad), plus
              the reward pass on the CPU. 6 / 10 / 0.3 at 512×288×39 frames ≈ 4 min on one 96 GB GPU.
            </p>
          </FormGroup>

          <FormGroup label="Reward weights">
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {REWARDS.map(r => (
                <NumberInput key={r.key} label={r.label} min={0}
                  value={(nft.weights ?? NFT_DEFAULTS.weights)[r.key] ?? 0}
                  onChange={v => set('weights', { ...NFT_DEFAULTS.weights, ...(nft.weights ?? {}), [r.key]: v ?? 0 })} />
              ))}
            </div>
            <ul className="text-sm text-gray-400 mt-2 list-disc pl-5">
              {REWARDS.map(r => <li key={r.key}><b>{r.label}</b>: {r.hint}</li>)}
            </ul>
            <p className="text-sm text-gray-400 mt-1">
              Every reward is z-normalised inside the group before the weighted sum, so no metric wins by its scale. 0
              switches one off.
            </p>
          </FormGroup>

          <FormGroup label="Ground-truth-free datasets">
            <TextInput label="Source-mode datasets (comma-separated path fragments)"
              value={(nft.src_datasets ?? []).join(', ')} placeholder="eg. rl_v2"
              onChange={v => set('src_datasets', v.split(',').map(s => s.trim()).filter(Boolean))} />
            <p className="text-sm text-gray-400 mt-2">
              Items whose path contains one of these are scored WITHOUT a ground truth: the target is the input clip
              itself (same file as the guide) and &lt;dataset&gt;/refs/&lt;name&gt;.png is the new person. Identity is
              measured against that picture (copying the input is penalised), background / pose / lips against the input,
              lighting from the face shading of the original person. Other items use their target as the ground truth.
            </p>
          </FormGroup>

          <FormGroup label="Reward models">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <TextInput label="Reward python (needs insightface + ultralytics)" value={nft.reward_python ?? ''}
                placeholder="empty = the trainer's python" onChange={v => set('reward_python', v)} />
              <TextInput label="InsightFace root (buffalo_l)" value={models.insightface_root ?? ''}
                placeholder="~/.insightface" onChange={v => set('reward_models', { ...models, insightface_root: v })} />
              <TextInput label="YOLO pose model" value={models.pose ?? ''} placeholder="yolov8m-pose.pt"
                onChange={v => set('reward_models', { ...models, pose: v })} />
              <TextInput label="YOLO person segmentation model" value={models.seg ?? ''} placeholder="yolov8m-seg.pt"
                onChange={v => set('reward_models', { ...models, seg: v })} />
              <TextInput label="DINOv2 model (character reward)" value={models.dino ?? ''} placeholder="facebook/dinov2-base"
                onChange={v => set('reward_models', { ...models, dino: v })} />
            </div>
          </FormGroup>
        </>
      )}

      <FormGroup label="Scene loss (regression training)">
        <Checkbox
          label="Weight the loss by the person mask + low-frequency lighting term"
          checked={!!scene}
          onChange={value => setJobConfig(value ? { ...SCENE_DEFAULTS } : undefined, 'config.process[0].train.scene_loss')}
        />
        {scene && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 mt-2">
            <NumberInput label="Person weight" min={0} value={scene.person_weight ?? 2}
              onChange={v => setScene('person_weight', v ?? 2)} />
            <NumberInput label="Background weight" min={0} value={scene.bg_weight ?? 1.5}
              onChange={v => setScene('bg_weight', v ?? 1.5)} />
            <NumberInput label="Low-frequency weight" min={0} value={scene.lowfreq_weight ?? 0.5}
              onChange={v => setScene('lowfreq_weight', v ?? 0.5)} />
            <NumberInput label="Low-frequency kernel (latent cells)" min={1} value={scene.lowfreq_kernel ?? 4}
              onChange={v => setScene('lowfreq_kernel', v ?? 4)} />
          </div>
        )}
        <p className="text-sm text-gray-400 mt-2">
          Needs &lt;targets&gt;/_person_masks/&lt;name&gt;.npy (tools/h3_rl/make_person_masks.py). Items without a mask
          train normally. Ignored while RL is on.
        </p>
      </FormGroup>
    </div>
  );
}
