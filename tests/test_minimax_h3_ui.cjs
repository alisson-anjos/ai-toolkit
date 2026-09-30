// Exercise the same transpilation/evaluation contract as the extension loader.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('../ui/node_modules/typescript');
const root = path.resolve(__dirname, '..');
const modules = new Map();
function load(relative) {
  if (modules.has(relative)) return modules.get(relative);
  const filename = path.join(root, relative);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  const customRequire = name => {
    if (name === 'next/link') return { __esModule: true, default: () => null };
    if (name === '@/components/formInputs') return { FormGroup: 'group', NumberInput: 'number', TextInput: 'text', SelectInput: 'select' };
    if (name.startsWith('@/')) {
      const base = 'ui/src/' + name.slice(2);
      return load(fs.existsSync(path.join(root, base + '.ts')) ? base + '.ts' : base + '.tsx');
    }
    if (name === './options') return {};
    return require(path.join(root, 'ui/node_modules', name));
  };
  new Function('require', 'module', 'exports', code)(customRequire, module, module.exports);
  modules.set(relative, module.exports);
  return module.exports;
}
const arches = load('extensions_built_in/diffusion_models/ui.tsx').AI_TOOLKIT_UI_MODELS;
const arch = arches.find(a => a.name === 'minimax_h3_ref2va');
const mode = arch.customModelSelectOptions.find(o => o.label === 'Control Conditioning');
const factor = arch.customModelSelectOptions.find(o => o.label === 'Guide Downscale Factor');
const distillation = arch.customModelSelectOptions.find(o => o.label === 'Distillation Handling Method');
const presentation = arch.customModelSelectOptions.find(o => o.label === 'Image Reference Presentation');
const config = {config: {process: [{model: {arch: arch.name, model_kwargs: {partition: 'ref2va_pruned'}}, train: {},
  datasets: [{folder_path: '/target', control_path_1: '/source', control_path_2: '/face', control_path_3: '/mask'}],
  sample: {samples: [{prompt: 'edit white region', ctrl_img_1: 'source.mp4', ctrl_img_2: 'face.png', ctrl_img_3: 'mask.mp4'}]}}]}};
function set(value, key) {
  const keys = key.replace(/\[(\d+)\]/g, '.$1').split('.');
  const last = keys.pop(); let object = config;
  for (const k of keys) { if (object[k] === undefined) object[k] = {}; object = object[k]; }
  object[last] = value;
}
const originalData = JSON.stringify(config.config.process[0].datasets);
const originalSamples = JSON.stringify(config.config.process[0].sample);
assert.equal(mode.getValue(config), 'reference');
assert.equal(factor.disabled(config), true);
mode.onChange('guide', config, set);
factor.onChange('2', config, set);
assert.equal(mode.getValue(config), 'guide');
assert.equal(factor.getValue(config), '2');
assert.equal(factor.disabled(config), false);
assert.deepEqual(config.config.process[0].model.model_kwargs, {
  partition: 'ref2va_pruned', align_video_refs: true, control_latent_only: false,
  guide_latent_only: true, reference_downscale_factor: 2, reference_dropout: 0,
});
mode.onChange('all_latent', config, set);
assert.equal(config.config.process[0].model.model_kwargs.control_latent_only, true);
mode.onChange('aligned_vlm', config, set);
assert.equal(config.config.process[0].model.model_kwargs.guide_latent_only, false);
assert.equal(mode.getValue(config), 'aligned_vlm');
assert.equal(presentation.disabled(config), true);
mode.onChange('reference', config, set);
assert.equal(presentation.disabled(config), false);
presentation.onChange('video', config, set);
assert.equal(factor.getValue(config), '1');
assert.equal(factor.disabled(config), true);
presentation.onChange('picture', config, set);
mode.onChange('guide', config, set);
factor.onChange('2', config, set);
distillation.onChange('dopsd', config, set);
assert.equal(factor.getValue(config), '1');
assert.equal(mode.getValue(config), 'reference');
assert.equal(mode.disabled(config), true);
distillation.onChange('ta', config, set);
assert.equal(mode.disabled(config), false);
mode.onChange('guide', config, set);
factor.onChange('2', config, set);
mode.onChange('reference', config, set);
assert.equal(factor.getValue(config), '1');
assert.equal(config.config.process[0].model.model_kwargs.control_latent_only, false);
assert.equal(config.config.process[0].model.model_kwargs.guide_latent_only, false);
assert.equal(JSON.stringify(config.config.process[0].datasets), originalData);
assert.equal(JSON.stringify(config.config.process[0].sample), originalSamples);
// Arch switches use the real helper and must preserve native reference channels
// between ref2va and another multi-control model, while clearing guide kwargs.
const utils = load('ui/src/app/jobs/new/utils.ts');
const other = {name: 'other', defaults: {}, additionalSections: ['datasets.multi_control_paths', 'sample.multi_ctrl_imgs']};
mode.onChange('guide', config, set);
factor.onChange('2', config, set);
utils.handleModelArchChange([arch, other], arch.name, 'other', config, set);
for (const key of ['align_video_refs', 'control_latent_only', 'guide_latent_only', 'reference_downscale_factor']) {
  assert.equal(config.config.process[0].model.model_kwargs[key], undefined);
}
assert.equal(config.config.process[0].datasets[0].control_path_1, '/source');
assert.equal(config.config.process[0].datasets[0].control_path_2, '/face');
assert.equal(config.config.process[0].datasets[0].control_path_3, '/mask');
assert.equal(config.config.process[0].sample.samples[0].ctrl_img_3, 'mask.mp4');
console.log('UI mode/factor transitions, native channels, and arch reset passed.');

const component = load('ui/src/components/H3TrainingOptions.tsx').default;
function findLabel(tree, label) {
  if (!tree) return null;
  if (Array.isArray(tree)) { for (const child of tree) { const found = findLabel(child, label); if (found) return found; } return null; }
  if (tree.props?.label === label) return tree;
  return findLabel(tree.props?.children, label);
}
const render = () => component({jobConfig: config, setJobConfig: set});
mode.onChange('all_latent', config, set);
let view = render();
findLabel(view, 'Image Reference Dropout').props.onChange(0.3);
findLabel(view, 'Video Guide Dropout').props.onChange(0.1);
assert.equal(config.config.process[0].model.model_kwargs.reference_dropout, 0.3);
assert.equal(config.config.process[0].model.model_kwargs.guide_dropout, 0.1);
findLabel(view, 'Additional Training Objectives').props.onChange(['pixel_l1', 'arcface', 'pose_heatmap']);
view = render();
findLabel(view, 'ArcFace TorchScript File').props.onChange('/models/arcface.pt');
view = render();
findLabel(view, 'Pose Heatmap TorchScript File').props.onChange('/models/pose.pt');
view = render();
findLabel(view, 'Face Crop Left').props.onChange(0.2);
view = render();
findLabel(view, 'ArcFace — face identity similarity Weight').props.onChange(0.4);
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses.length, 3);
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses[1].weight, 0.4);
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses[1].model_path, '/models/arcface.pt');
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses[1].crop[0], 0.2);
view = render();
findLabel(view, 'Additional Training Objectives').props.onChange(['arcface', 'pose_heatmap']);
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses[0].weight, 0.4);
assert.equal(config.config.process[0].model.model_kwargs.auxiliary_losses[0].model_path, '/models/arcface.pt');
mode.onChange('reference', config, set);
assert.equal(config.config.process[0].model.model_kwargs.reference_dropout, 0);
assert.equal(findLabel(render(), 'Image Reference Dropout').props.disabled, true);
console.log('UI dropout, combined custom losses, weights, evaluator paths, and face crop passed.');
