'use client';
import { SelectInput } from '@/components/formInputs';

type Props = {
  channel: number;
  value?: 'guide' | 'reference';
  onChange: (value: string) => void;
};

export default function H3ControlRoleSelect({ channel, value, onChange }: Props) {
  const role = value ?? (channel === 1 ? 'guide' : 'reference');
  return (
    <div className="pt-2">
      <SelectInput label={`Channel ${channel} Role`} value={role} onChange={onChange}
        options={[
          { value: 'guide', label: 'Guide latent — aligned with target' },
          { value: 'reference', label: 'Native reference — identity / appearance' },
        ]} />
      <p className="pt-1 text-xs text-gray-400">
        {role === 'guide'
          ? 'Image or video encoded by the VAE, aligned to the target, and reduced by Guide Downscale Factor. Bypasses the VLM.'
          : 'Uses the native reference layout and VLM conditioning. Guide Downscale Factor does not apply.'}
      </p>
    </div>
  );
}
