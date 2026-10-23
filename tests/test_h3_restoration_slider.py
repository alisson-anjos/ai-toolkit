import pytest
import torch
from toolkit.h3_restoration_slider import residual_target


def test_scale_endpoints_and_flow_sign():
    base=torch.tensor([[0.1, 0.2]])
    noise=torch.tensor([[0.5,0.6]])
    clean=torch.tensor([[0.4,0.3]])
    flow=noise-clean
    assert torch.equal(residual_target(base,flow,0),base)
    assert torch.allclose(residual_target(base,flow,1),flow)
    assert torch.allclose(residual_target(base,flow,.5),(base+flow)/2)


def test_bound_is_per_sample_and_targets_detached():
    base=torch.zeros(2,3,requires_grad=True)
    clean=torch.tensor([[100.,100.,100.],[.1,.1,.1]],requires_grad=True)
    target=residual_target(base,clean,1,residual_cap=.5)
    assert torch.allclose(target[0],torch.full((3,),.5))
    assert torch.allclose(target[1],torch.full((3,),.1))
    assert not target.requires_grad


@pytest.mark.parametrize('scale',[-1,1.1,float('nan')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):residual_target(torch.zeros(1,2),torch.ones(1,2),scale)


def test_example_uses_one_output_namespace():
    import yaml
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'config/examples/h3_flashvsr_sharpness_slider.yaml').read_text())
    name=cfg['config']['name']
    def check(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='name':assert v==name
                else:check(v)
        elif isinstance(x,list):
            for v in x:check(v)
    check(cfg)
