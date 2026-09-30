from types import SimpleNamespace

from toolkit.training_oom import batch_geometry_summary, optimizer_updates_during_backward


def test_oom_reports_actual_files_and_geometry():
    item = SimpleNamespace(path='/data/video.mp4', crop_width=1280, crop_height=768, num_frames=73)
    batch = SimpleNamespace(file_items=[item], num_frames=5)
    report = batch_geometry_summary([None, batch])
    assert '/data/video.mp4' in report
    assert '1280x768' in report
    assert 'frames=73' in report


def test_backward_updates_are_detected_through_accelerator_wrapper():
    fused = SimpleNamespace(updates_during_backward=True)
    assert optimizer_updates_during_backward(SimpleNamespace(optimizer=fused))
    assert not optimizer_updates_during_backward(SimpleNamespace(updates_during_backward=False))
    wrapper = SimpleNamespace()
    wrapper.optimizer = wrapper
    assert not optimizer_updates_during_backward(wrapper)
