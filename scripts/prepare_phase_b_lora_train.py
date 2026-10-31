#!/usr/bin/env python3
"""Prepare Phase B mask-free Mocha selective character-swap dataset for ai-toolkit.

Strictly uses Mocha same-background datasets (NO Genjutsu):
- `v1.1/bodyswap_inv_v1/v2_mocha_inv` (80 clips) paired with `mocha_orig_all` target descriptions
- `v1.1/bodyswap_inv_v1/full_mocha_inv` (20 clips) + `full_mocha_orig` (20 clips)
- `v1.1/bodyswap_inv_v1/mocha` + `v1.1/bodyswap_inv_v1/mocha_orig` bidirectional pairs
- `v2/motion_binding_v6_301/original_guides` (14 unmasked 2-performer clips)
- 60 two-reference (`<Picture 1>` and `<Picture 2>`) selective Mocha pairs where the prompt
  and Molmo2 visual conditioning must select BOTH the target person in `<Video 1>` and the
  designated reference (`<Subject 1>` in `<Picture 1>` vs `<Subject 2>` in `<Picture 2>`).
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


_SUBJ_DEF_RE = re.compile(
    r"<Subject 1>\s+is\s+(?:the exact entity specified by\s+)?(.+?)\s+(?:shown\s+)?in\s+<Picture 1>",
    re.DOTALL,
)


def extract_subject_desc(caption: str):
    m = _SUBJ_DEF_RE.search(caption)
    if not m:
        return None
    desc = " ".join(m.group(1).split()).strip(" ,.")
    if not desc.lower().startswith(("the ", "a ", "an ")):
        desc = "the " + desc
    return desc


def upgrade_mocha_caption_single_ref(target_caption: str, guide_person_desc: str) -> str:
    """Replace generic 'the guide subject' / 'the main subject of the guide video' with the
    exact visual description of the person inside <Video 1> being replaced."""
    text = " ".join(target_caption.split())
    if not guide_person_desc:
        return text

    text = text.replace(
        "The guide video provides the scene and the performance:",
        "The guide video <Video 1> provides the scene and the performance:",
    )
    text = text.replace(
        "only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.",
        f"only {guide_person_desc} in <Video 1> is replaced: their face, hair, body and clothing become <Subject 1> (keep any other people and the background in <Video 1> untouched).",
    )
    text = text.replace(
        "Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and performance of the guide video.",
        f"Replace only {guide_person_desc} in <Video 1> with <Subject 1> from <Picture 1>, keeping the scene, camera and the complete motion and performance of <Video 1>.",
    )
    text = text.replace(
        "<Subject 1> takes the exact place of the guide subject and moves",
        f"<Subject 1> takes the exact place of {guide_person_desc} in <Video 1> and moves",
    )
    # Also support full_mocha_inv style phrasing if present
    text = text.replace(
        "replace the target subject in <Video 1> with <Subject 1>",
        f"replace only {guide_person_desc} in <Video 1> with <Subject 1>",
    )
    return text


def build_mocha_two_ref_caption(
    new_person_desc: str,
    guide_person_desc: str,
    other_ref_desc: str,
    active_slot: int,
) -> str:
    if active_slot == 1:
        s1_desc, s2_desc = new_person_desc, other_ref_desc
        active_subj, active_pic, unused_subj, unused_pic = (
            "<Subject 1>",
            "<Picture 1>",
            "<Subject 2>",
            "<Picture 2>",
        )
    else:
        s1_desc, s2_desc = other_ref_desc, new_person_desc
        active_subj, active_pic, unused_subj, unused_pic = (
            "<Subject 2>",
            "<Picture 2>",
            "<Subject 1>",
            "<Picture 1>",
        )

    return (
        f"body_swap: subject_definitions: <Subject 1> is {s1_desc} shown in <Picture 1>. "
        f"<Subject 2> is {s2_desc} shown in <Picture 2>. "
        f"The guide video <Video 1> provides the scene and the performance: its background, environment, "
        f"lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip "
        f"movement and facial expression, frame by frame; only {guide_person_desc} in <Video 1> is replaced: "
        f"their face, hair, body and clothing become {active_subj} from {active_pic}, while {unused_subj} "
        f"({unused_pic}) is not used and any other people in <Video 1> remain untouched.  "
        f"summary: [video generation + reference] Replace only {guide_person_desc} in <Video 1> with "
        f"{active_subj} ({active_pic}), keeping the scene, camera and the complete motion and performance "
        f"of <Video 1>.  "
        f"detailed_description: [Shot 1] {active_subj} takes the exact place of {guide_person_desc} in "
        f"<Video 1> and moves with the same timing, body pose, gestures, head motion, gaze, speech and "
        f"facial expressions, while keeping the complete identity from {active_pic}: facial structure, "
        f"hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, "
        f"colors and camera of <Video 1> are preserved. Prevent identity drift, clothing changes, texture "
        f"crawling and flicker."
    )


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_media(folder: Path, stem: str):
    if not folder.exists():
        return None
    for ext in (".mp4", ".png", ".jpg", ".jpeg", ".webp"):
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def resolve_guide_and_ref_dirs(sub_dir: Path):
    g_dir = sub_dir / "guides" if (sub_dir / "guides").exists() else sub_dir / "control_0"
    r_dir = sub_dir / "refs" if (sub_dir / "refs").exists() else sub_dir / "control_1"
    return g_dir, r_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-cache-dir",
        default="/workspace/ai-toolkit/datasets/_hf_mocha_raw",
    )
    parser.add_argument(
        "--dataset-out",
        default="/workspace/ai-toolkit/datasets/phase_b_mocha_selective",
    )
    parser.add_argument(
        "--max-extra-mocha-pairs",
        type=int,
        default=80,
        help="Number of bidirectional pairs from v1.1/bodyswap_inv_v1/mocha + mocha_orig",
    )
    args = parser.parse_args()

    print("[1/2] Downloading Mocha subsets from Alissonerdx/bodyswap-video-train...")
    # First download all text files in mocha & mocha_orig so we can pick stems that exist in both
    txt_patterns = [
        "v1.1/bodyswap_inv_v1/v2_mocha_inv/**",
        "v1.1/bodyswap_inv_v1/full_mocha_inv/**",
        "v1.1/bodyswap_inv_v1/full_mocha_orig/**",
        "v1.1/bodyswap_inv_v1/mocha_orig_all/targets/*.txt",
        "v1.1/bodyswap_inv_v1/mocha/targets/*.txt",
        "v1.1/bodyswap_inv_v1/mocha_orig/targets/*.txt",
        "v2/motion_binding_v6_301/original_guides/**",
    ]
    repo_dir = Path(
        snapshot_download(
            repo_id="Alissonerdx/bodyswap-video-train",
            repo_type="dataset",
            local_dir=args.hf_cache_dir,
            allow_patterns=txt_patterns,
        )
    )

    # Identify common stems between mocha and mocha_orig (not already in v2_mocha_inv)
    v2_stems = {p.stem for p in (repo_dir / "v1.1/bodyswap_inv_v1/v2_mocha_inv/targets").glob("*.txt")}
    m_stems = {p.stem for p in (repo_dir / "v1.1/bodyswap_inv_v1/mocha/targets").glob("*.txt")}
    mo_stems = {p.stem for p in (repo_dir / "v1.1/bodyswap_inv_v1/mocha_orig/targets").glob("*.txt")}
    paired_stems = sorted((m_stems & mo_stems) - v2_stems)[: args.max_extra_mocha_pairs]

    if paired_stems:
        media_patterns = []
        for s in paired_stems:
            media_patterns.append(f"v1.1/bodyswap_inv_v1/mocha/**/{s}.*")
            media_patterns.append(f"v1.1/bodyswap_inv_v1/mocha_orig/**/{s}.*")
        print(f"      Downloading {len(paired_stems)} additional bidirectional Mocha pairs...")
        snapshot_download(
            repo_id="Alissonerdx/bodyswap-video-train",
            repo_type="dataset",
            local_dir=args.hf_cache_dir,
            allow_patterns=media_patterns,
        )

    out_base = Path(args.dataset_out)
    if out_base.exists():
        shutil.rmtree(out_base)
    refs1_dir = out_base / "refs_1"
    refs2_dir = out_base / "refs_2"
    for d in (
        refs1_dir / "targets",
        refs1_dir / "control_0",
        refs1_dir / "control_1",
        refs2_dir / "targets",
        refs2_dir / "control_0",
        refs2_dir / "control_1",
        refs2_dir / "control_2",
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Build lookup of all original person descriptions from mocha_orig_all and mocha_orig
    orig_desc_map = {}
    for tdir in (
        repo_dir / "v1.1/bodyswap_inv_v1/mocha_orig_all/targets",
        repo_dir / "v1.1/bodyswap_inv_v1/mocha_orig/targets",
    ):
        if tdir.exists():
            for p in tdir.glob("*.txt"):
                d = extract_subject_desc(p.read_text(encoding="utf-8"))
                if d:
                    orig_desc_map[p.stem] = d

    inv_desc_map = {}
    for tdir in (
        repo_dir / "v1.1/bodyswap_inv_v1/v2_mocha_inv/targets",
        repo_dir / "v1.1/bodyswap_inv_v1/mocha/targets",
    ):
        if tdir.exists():
            for p in tdir.glob("*.txt"):
                d = extract_subject_desc(p.read_text(encoding="utf-8"))
                if d:
                    inv_desc_map[p.stem] = d

    counts = {
        "v2_mocha_inv": 0,
        "full_mocha_inv_orig": 0,
        "mocha_fwd": 0,
        "mocha_rev": 0,
        "motion_binding_2ref": 0,
        "mocha_selective_2ref": 0,
    }
    pool_for_2ref = []

    # 1. v2_mocha_inv (80 clips)
    sub = repo_dir / "v1.1/bodyswap_inv_v1/v2_mocha_inv"
    g_dir, r_dir = resolve_guide_and_ref_dirs(sub)
    for txt_path in sorted((sub / "targets").glob("*.txt")):
        stem = txt_path.stem
        t_vid = find_media(sub / "targets", stem)
        c0_vid = find_media(g_dir, stem)
        c1_img = find_media(r_dir, stem)
        if not (t_vid and c0_vid and c1_img):
            continue
        raw_cap = txt_path.read_text(encoding="utf-8")
        guide_person = orig_desc_map.get(stem, "the main subject of the guide video")
        new_person = extract_subject_desc(raw_cap) or "the person"
        up_cap = upgrade_mocha_caption_single_ref(raw_cap, guide_person)

        out_stem = f"v2mocha_{stem}"
        link_or_copy(t_vid, refs1_dir / "targets" / f"{out_stem}{t_vid.suffix}")
        link_or_copy(c0_vid, refs1_dir / "control_0" / f"{out_stem}{c0_vid.suffix}")
        link_or_copy(c1_img, refs1_dir / "control_1" / f"{out_stem}{c1_img.suffix}")
        (refs1_dir / "targets" / f"{out_stem}.txt").write_text(up_cap + "\n", encoding="utf-8")
        counts["v2_mocha_inv"] += 1
        if stem in orig_desc_map:
            pool_for_2ref.append((out_stem, t_vid, c0_vid, c1_img, new_person, guide_person))

    # 2. full_mocha_inv (20 clips) + full_mocha_orig (20 clips)
    for tag, sub_rel, is_inv in (
        ("fullinv", "v1.1/bodyswap_inv_v1/full_mocha_inv", True),
        ("fullorig", "v1.1/bodyswap_inv_v1/full_mocha_orig", False),
    ):
        sub = repo_dir / sub_rel
        if not (sub / "targets").exists():
            continue
        g_dir, r_dir = resolve_guide_and_ref_dirs(sub)
        for txt_path in sorted((sub / "targets").glob("*.txt")):
            stem = txt_path.stem
            t_vid = find_media(sub / "targets", stem)
            c0_vid = find_media(g_dir, stem)
            c1_img = find_media(r_dir, stem)
            if not (t_vid and c0_vid and c1_img):
                continue
            raw_cap = txt_path.read_text(encoding="utf-8")
            guide_person = (orig_desc_map if is_inv else inv_desc_map).get(stem)
            up_cap = upgrade_mocha_caption_single_ref(raw_cap, guide_person) if guide_person else raw_cap.strip()
            out_stem = f"{tag}_{stem}"
            link_or_copy(t_vid, refs1_dir / "targets" / f"{out_stem}{t_vid.suffix}")
            link_or_copy(c0_vid, refs1_dir / "control_0" / f"{out_stem}{c0_vid.suffix}")
            link_or_copy(c1_img, refs1_dir / "control_1" / f"{out_stem}{c1_img.suffix}")
            (refs1_dir / "targets" / f"{out_stem}.txt").write_text(up_cap + "\n", encoding="utf-8")
            counts["full_mocha_inv_orig"] += 1

    # 3. Paired bidirectional clips from mocha (A -> B) and mocha_orig (B -> A)
    sub_m = repo_dir / "v1.1/bodyswap_inv_v1/mocha"
    sub_mo = repo_dir / "v1.1/bodyswap_inv_v1/mocha_orig"
    gm_dir, rm_dir = resolve_guide_and_ref_dirs(sub_m)
    gmo_dir, rmo_dir = resolve_guide_and_ref_dirs(sub_mo)
    for stem in paired_stems:
        # Forward: mocha (guide has orig person, target has new person)
        t_vid = find_media(sub_m / "targets", stem)
        c0_vid = find_media(gm_dir, stem)
        c1_img = find_media(rm_dir, stem)
        txt_path = sub_m / "targets" / f"{stem}.txt"
        if t_vid and c0_vid and c1_img and txt_path.exists():
            raw_cap = txt_path.read_text(encoding="utf-8")
            guide_person = orig_desc_map.get(stem)
            new_person = inv_desc_map.get(stem, "the person")
            up_cap = upgrade_mocha_caption_single_ref(raw_cap, guide_person)
            out_stem = f"mochafwd_{stem}"
            link_or_copy(t_vid, refs1_dir / "targets" / f"{out_stem}{t_vid.suffix}")
            link_or_copy(c0_vid, refs1_dir / "control_0" / f"{out_stem}{c0_vid.suffix}")
            link_or_copy(c1_img, refs1_dir / "control_1" / f"{out_stem}{c1_img.suffix}")
            (refs1_dir / "targets" / f"{out_stem}.txt").write_text(up_cap + "\n", encoding="utf-8")
            counts["mocha_fwd"] += 1

        # Reverse: mocha_orig (guide has new person, target has orig person)
        t_vid_r = find_media(sub_mo / "targets", stem)
        c0_vid_r = find_media(gmo_dir, stem)
        c1_img_r = find_media(rmo_dir, stem)
        txt_path_r = sub_mo / "targets" / f"{stem}.txt"
        if t_vid_r and c0_vid_r and c1_img_r and txt_path_r.exists():
            raw_cap_r = txt_path_r.read_text(encoding="utf-8")
            guide_person_r = inv_desc_map.get(stem)
            up_cap_r = upgrade_mocha_caption_single_ref(raw_cap_r, guide_person_r)
            out_stem_r = f"mocharev_{stem}"
            link_or_copy(t_vid_r, refs1_dir / "targets" / f"{out_stem_r}{t_vid_r.suffix}")
            link_or_copy(c0_vid_r, refs1_dir / "control_0" / f"{out_stem_r}{c0_vid_r.suffix}")
            link_or_copy(c1_img_r, refs1_dir / "control_1" / f"{out_stem_r}{c1_img_r.suffix}")
            (refs1_dir / "targets" / f"{out_stem_r}.txt").write_text(up_cap_r + "\n", encoding="utf-8")
            counts["mocha_rev"] += 1

    # 4. Multi-ref (2 references): motion_binding_v6_301/original_guides (14 unmasked same-bg clips)
    mbind_dir = repo_dir / "v2/motion_binding_v6_301/original_guides"
    if (mbind_dir / "targets").exists():
        for txt_path in sorted((mbind_dir / "targets").glob("*.txt")):
            stem = txt_path.stem
            t_vid = find_media(mbind_dir / "targets", stem)
            c0_vid = find_media(mbind_dir / "control_0", stem)
            c1_img = find_media(mbind_dir / "control_1", stem)
            c2_img = find_media(mbind_dir / "control_2", stem)
            if not (t_vid and c0_vid and c1_img and c2_img):
                continue
            out_stem = f"mbind2_{stem}"
            link_or_copy(t_vid, refs2_dir / "targets" / f"{out_stem}{t_vid.suffix}")
            link_or_copy(c0_vid, refs2_dir / "control_0" / f"{out_stem}{c0_vid.suffix}")
            link_or_copy(c1_img, refs2_dir / "control_1" / f"{out_stem}{c1_img.suffix}")
            link_or_copy(c2_img, refs2_dir / "control_2" / f"{out_stem}{c2_img.suffix}")
            link_or_copy(txt_path, refs2_dir / "targets" / f"{out_stem}.txt")
            counts["motion_binding_2ref"] += 1

    # 5. Build 60 two-reference selective Mocha pairs (<Picture 1> vs <Picture 2>)
    n_pool = len(pool_for_2ref)
    for i in range(min(60, n_pool)):
        stem, t_vid, c0_vid, c1_img, new_person, guide_person = pool_for_2ref[i]
        other_item = pool_for_2ref[(i + 19) % n_pool]
        other_img = other_item[3]
        other_person_desc = other_item[4]
        active_slot = 1 if (i % 2 == 0) else 2
        cap_2ref = build_mocha_two_ref_caption(
            new_person_desc=new_person,
            guide_person_desc=guide_person,
            other_ref_desc=other_person_desc,
            active_slot=active_slot,
        )
        out_stem = f"mocha2ref_s{active_slot}_{stem}"
        link_or_copy(t_vid, refs2_dir / "targets" / f"{out_stem}{t_vid.suffix}")
        link_or_copy(c0_vid, refs2_dir / "control_0" / f"{out_stem}{c0_vid.suffix}")
        if active_slot == 1:
            link_or_copy(c1_img, refs2_dir / "control_1" / f"{out_stem}{c1_img.suffix}")
            link_or_copy(other_img, refs2_dir / "control_2" / f"{out_stem}{other_img.suffix}")
        else:
            link_or_copy(other_img, refs2_dir / "control_1" / f"{out_stem}{other_img.suffix}")
            link_or_copy(c1_img, refs2_dir / "control_2" / f"{out_stem}{c1_img.suffix}")
        (refs2_dir / "targets" / f"{out_stem}.txt").write_text(cap_2ref + "\n", encoding="utf-8")
        counts["mocha_selective_2ref"] += 1

    print("[2/2] Mocha dataset assembly complete:")
    for k, v in counts.items():
        print(f"      - {k}: {v}")
    total_r1 = len(list((refs1_dir / "targets").glob("*.txt")))
    total_r2 = len(list((refs2_dir / "targets").glob("*.txt")))
    print(f"      TOTAL refs_1: {total_r1} clips | TOTAL refs_2: {total_r2} clips")


if __name__ == "__main__":
    main()
