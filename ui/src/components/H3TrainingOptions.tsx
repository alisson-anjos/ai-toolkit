'use client';
import type { JobConfig } from '@/types';
import { FormGroup, NumberInput, SelectInput, TextInput } from '@/components/formInputs';

type Props = { jobConfig: JobConfig; setJobConfig: (value: any, key: string) => void };
type AuxiliaryLoss = { type: string; weight: number; model_path?: string; crop?: number[] };
const lossOptions = [
  { value: 'pixel_l1', label: 'Pixel L1 — target reconstruction' },
  { value: 'arcface', label: 'ArcFace — face identity similarity' },
  { value: 'pose_heatmap', label: 'Pose — differentiable heatmaps' },
];

export default function H3TrainingOptions({ jobConfig, setJobConfig }: Props) {
  const kwargs = jobConfig.config.process[0].model.model_kwargs ?? {};
  const losses: AuxiliaryLoss[] = kwargs.auxiliary_losses ?? [];
  const set = (key: string, value: any) => setJobConfig(value, `config.process[0].model.model_kwargs.${key}`);
  const dropoutEnabled = (kwargs.guide_latent_only || kwargs.control_latent_only) && !kwargs.dopsd && !kwargs.image_refs_as_video;
  const updateLoss = (type: string, updates: Partial<AuxiliaryLoss>) => {
    set('auxiliary_losses', losses.map(loss => loss.type === type ? { ...loss, ...updates } : loss));
  };
  return (
    <div className="space-y-4 pt-3">
      <FormGroup label="Reference & Guide Dropout">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <NumberInput label="Native Reference Dropout" min={0} max={1}
            value={kwargs.reference_dropout ?? 0} disabled={!dropoutEnabled || !kwargs.control_latent_only}
            onChange={value => set('reference_dropout', value ?? 0)} />
          <NumberInput label="Image / Video Guide Dropout" min={0} max={1}
            value={kwargs.guide_dropout ?? 0} disabled={!dropoutEnabled}
            onChange={value => set('guide_dropout', value ?? 0)} />
        </div>
        <p className="text-sm text-gray-400 mt-2">
          Each reference is dropped independently with this probability. 0 keeps all; 1 drops all.
          Image dropout requires All controls as latents (caption-only VLM). Video dropout also works with
          Latent video guides + native image references. Source and mask videos share the video probability.
          Use batch size 1 when dataset items have different reference counts or aspect ratios.
        </p>
      </FormGroup>
      <FormGroup label="Auxiliary Losses">
        <SelectInput label="Additional Training Objectives" multiple value={losses.map(loss => loss.type)}
          options={lossOptions} onChange={types => set('auxiliary_losses', types.map(type =>
            losses.find(loss => loss.type === type) ?? { type, weight: 0.1 }))} />
        <p className="text-sm text-gray-400 mt-2">
          Combine objectives with the standard training loss. ArcFace and pose require local differentiable
          TorchScript evaluator files. These objectives add VAE decoding and increase memory use.
        </p>
        {losses.map(loss => (
          <div key={loss.type} className="mt-3 rounded-md border border-gray-700 p-3 space-y-2">
            <NumberInput label={`${lossOptions.find(option => option.value === loss.type)?.label ?? loss.type} Weight`}
              min={0} value={loss.weight} onChange={weight => updateLoss(loss.type, { weight: weight ?? 0 })} />
            {loss.type !== 'pixel_l1' && (
              <TextInput label={loss.type === 'arcface' ? 'ArcFace TorchScript File' : 'Pose Heatmap TorchScript File'}
                value={loss.model_path ?? ''} required placeholder="/path/to/evaluator.pt"
                onChange={model_path => updateLoss(loss.type, { model_path })} />
            )}
            {loss.type === 'arcface' && (
              <>
                <p className="text-sm text-gray-400">Face crop in target frame coordinates (0–1). Use a crop containing the face; no automatic face detection is performed.</p>
                <div className="grid grid-cols-2 gap-2">
                  {['Left', 'Top', 'Right', 'Bottom'].map((label, index) => (
                    <NumberInput key={label} label={`Face Crop ${label}`} min={0} max={1}
                      value={(loss.crop ?? [0, 0, 1, 1])[index]}
                      onChange={value => {
                        const crop = [...(loss.crop ?? [0, 0, 1, 1])]; crop[index] = value ?? 0;
                        updateLoss(loss.type, { crop });
                      }} />
                  ))}
                </div>
              </>
            )}
            {loss.type === 'pose_heatmap' && (
              <p className="text-sm text-gray-400">Evaluator input: RGB [0, 1], 256×256. Output: differentiable joint heatmaps. An OpenPose detector returning discrete keypoints cannot be used here.</p>
            )}
          </div>
        ))}
        {losses.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
            <NumberInput label="Apply Below Noise Sigma" min={0.01} max={1}
              value={kwargs.auxiliary_loss_max_sigma ?? 0.5} onChange={value => set('auxiliary_loss_max_sigma', value ?? 0.5)} />
            <SelectInput label="Decoded Loss Clip" value={String(kwargs.auxiliary_loss_latent_frames ?? 2)}
              options={[
                { value: '2', label: 'First 5 video frames (default)' },
                { value: '7', label: 'First 22 video frames' },
                { value: '12', label: 'First 39 video frames' },
                { value: '0', label: 'Full clip (more memory)' },
              ]} onChange={value => set('auxiliary_loss_latent_frames', Number(value))} />
            <NumberInput label="Sampled Frames Per Loss" min={1}
              value={kwargs.auxiliary_loss_sample_frames ?? 1} onChange={value => set('auxiliary_loss_sample_frames', value ?? 1)} />
          </div>
        )}
      </FormGroup>
    </div>
  );
}
