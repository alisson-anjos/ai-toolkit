import pytest
import torch
from torch.utils.checkpoint import checkpoint
from toolkit.activation_offloading import saved_tensor_offloading

@pytest.mark.parametrize('use_checkpoint', [False, True])
def test_cpu_storage_preserves_cuda_backward_and_frees_saved_storage(use_checkpoint):
    if not torch.cuda.is_available():
        pytest.skip('CUDA verification')
    torch.manual_seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(256,512),torch.nn.SiLU(),torch.nn.Linear(512,64)).cuda()
    x=torch.randn(128,256,device='cuda',requires_grad=True)
    results=[]
    for enabled in [False,True]:
        model.zero_grad(set_to_none=True);x.grad=None
        with saved_tensor_offloading(enabled):
            y=checkpoint(model,x,use_reentrant=False) if use_checkpoint else model(x)
            y.square().mean().backward()
        results.append([x.grad.detach().clone()]+[p.grad.detach().clone() for p in model.parameters()])
    for a,b in zip(*results):torch.testing.assert_close(a,b)
    context=saved_tensor_offloading(True)
    original_device,stored=context.pack_hook(x)
    assert stored.device.type=='cpu'
    assert stored.is_pinned()
    torch.testing.assert_close(context.unpack_hook((original_device,stored)),x)
