import pytest

from toolkit.config_modules import SampleConfig, SampleItem


@pytest.mark.parametrize('kwargs', [{}, {'neg': False}, {'neg': None}, {'neg': ''}])
def test_omitted_and_legacy_negative_prompts_are_text(kwargs):
    config = SampleConfig(**kwargs)
    assert config.neg == ''
    assert SampleItem(config, prompt='test').neg == ''


def test_per_sample_negative_prompt_can_clear_or_override_default():
    config = SampleConfig(neg='blur')
    assert SampleItem(config, prompt='test').neg == 'blur'
    assert SampleItem(config, prompt='test', neg=False).neg == ''
    assert SampleItem(config, prompt='test', neg='noise').neg == 'noise'


@pytest.mark.parametrize('value', [True, 1, []])
def test_invalid_negative_prompts_fail_before_model_load(value):
    with pytest.raises(ValueError, match='sample neg'):
        SampleConfig(neg=value)
