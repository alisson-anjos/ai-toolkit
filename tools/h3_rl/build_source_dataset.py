"""Ground-truth-free ("source mode") dataset for H3 RL: real clips + pictures of OTHER people.

Layout written to OUT (point the dataset at OUT/targets, control_path_1 OUT/guides with role guide, control_path_2
OUT/refs with role reference, and list a fragment of OUT in train.bfs_nft.src_datasets):
    targets/<name>.mp4   the real clip (the trainer only takes its shape; rewards compare against it)
    guides/<name>.mp4    the same clip (the aligned guide = the video to edit)
    refs/<name>.png      a random picture from REFS: the person on grey, light neutralised (so the model must relight)
    targets/<name>.txt   the caption, from --caption (a template file); {desc} is replaced by --descriptions[name]
                         when given (e.g. "the young woman in a white T-shirt"), otherwise by "the person"

Optional: --seconds S cuts S seconds from the middle of each clip at 24 fps (long side --long-side).
Then run make_person_masks.py on OUT/targets.

    python tools/h3_rl/build_source_dataset.py --videos /clips --refs /people --caption caption.txt --out /data/rl
"""
import argparse, glob, json, os, random, subprocess
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True, help="folder of real .mp4 clips")
ap.add_argument("--refs", required=True, help="folder of pictures of other people (jpg/png)")
ap.add_argument("--caption", required=True, help="caption template file ({desc} = the person being replaced)")
ap.add_argument("--out", required=True)
ap.add_argument("--descriptions", default=None, help="optional JSON {clip name: description of its person}")
ap.add_argument("--seconds", type=float, default=0.0, help="cut this many seconds from the middle (0 = whole clip)")
ap.add_argument("--long-side", type=int, default=640)
ap.add_argument("--seg-model", default="yolov8m-seg.pt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
seg = YOLO(a.seg_model)
template = open(a.caption).read()
desc = json.load(open(a.descriptions)) if a.descriptions else {}
refs = sorted(p for e in ("jpg", "jpeg", "png", "webp") for p in glob.glob(f"{a.refs}/*.{e}"))
for d in ("targets", "guides", "refs", "_grey"):
    os.makedirs(f"{a.out}/{d}", exist_ok=True)


def grey(src):
    """The biggest person on #808080 (feathered YOLO mask)."""
    img = cv2.imread(src)
    r = seg(img, verbose=False, conf=0.3, classes=[0])[0]
    m = np.zeros(img.shape[:2], np.float32)
    if r.masks is not None:
        mm = r.masks.data.cpu().numpy()
        m = cv2.resize((mm[int(np.argmax([(x > 0.5).sum() for x in mm]))] > 0.5).astype(np.float32), img.shape[1::-1])
    m = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]
    return (img * m + 128 * (1 - m)).astype(np.uint8)


def neutral(img):
    """Grey-world + normalised exposure + small jitter on the person, so the reference carries no scene light."""
    x = img.astype(np.float32)
    person = np.abs(x - 128).sum(-1) > 6
    if person.sum() < 100:
        return img
    px = x[person] * (x[person].mean() / np.maximum(x[person].mean(0), 1))
    px = px * (118.0 / max(px.mean(), 1))
    jit = np.array([random.uniform(0.93, 1.07) for _ in range(3)]) * random.uniform(0.9, 1.1)
    px = (px - px.mean()) * random.uniform(0.9, 1.15) + px.mean()
    x[person] = np.clip(px * jit, 0, 255)
    return x.astype(np.uint8)


n = 0
for v in sorted(glob.glob(f"{a.videos}/*.mp4")):
    name = os.path.splitext(os.path.basename(v))[0]
    dst = f"{a.out}/targets/{name}.mp4"
    if not os.path.exists(dst):
        if a.seconds > 0:
            c = cv2.VideoCapture(v)
            dur = (c.get(7) or 0) / (c.get(5) or 24)
            start = max(0.0, dur / 2 - a.seconds / 2)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", v, "-t",
                            str(a.seconds), "-vf", f"fps=24,scale='if(gt(iw,ih),{a.long_side},-2)':'if(gt(iw,ih),-2,{a.long_side})'",
                            "-an", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", dst], check=True)
        else:
            os.symlink(os.path.abspath(v), dst)
    g = f"{a.out}/guides/{name}.mp4"
    if not os.path.lexists(g):
        os.symlink(os.path.abspath(dst), g)
    ref = random.choice(refs)
    gp = f"{a.out}/_grey/{os.path.basename(ref)}.png"
    if not os.path.exists(gp):
        cv2.imwrite(gp, grey(ref))
    cv2.imwrite(f"{a.out}/refs/{name}.png", neutral(cv2.imread(gp)))
    open(f"{a.out}/targets/{name}.txt", "w").write(template.replace("{desc}", desc.get(name, "the person")))
    n += 1
print("items", n, "->", a.out)
