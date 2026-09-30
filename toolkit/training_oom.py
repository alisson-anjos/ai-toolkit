"""Small, dependency-free helpers for actionable training OOM reports."""


def batch_geometry_summary(batches):
    records = []
    for batch in batches:
        if batch is None:
            continue
        for item in getattr(batch, 'file_items', None) or []:
            records.append(
                f"file={getattr(item, 'path', '<unknown>')} "
                f"canvas={getattr(item, 'crop_width', '?')}x"
                f"{getattr(item, 'crop_height', '?')} "
                f"frames={getattr(item, 'num_frames', getattr(batch, 'num_frames', '?'))}"
            )
    return '; '.join(records) or 'batch geometry unavailable'


def optimizer_updates_during_backward(optimizer):
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        if getattr(optimizer, 'updates_during_backward', False):
            return True
        optimizer = getattr(optimizer, 'optimizer', None)
    return False
