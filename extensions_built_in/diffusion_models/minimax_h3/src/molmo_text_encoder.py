"""Molmo2-8B + calibrated H3 projection conditioner for MiniMax-H3 in ai-toolkit.

Maps Molmo2-8B hidden states (tap 24) through a calibrated ClipProj-style
ridge + residual MLP projection into MiniMax-H3's 5120-dim conditioning space.
Supports Phase B multimodal conditioning where reference images (<Picture k>)
and sampled guide video frames (<Video k> with timestamps) enter Molmo2's
visual prefix before the caption suffix.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from .packing import TEXT_TAG
from .text_encoder import VideoRef


_WORKER_CODE = r'''
import json
import sys
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def prepare_inputs(processor, ids, visuals, max_video_frames=8):
    images, prefix = [], []
    picture_count = video_count = source_count = 0
    for item in visuals:
        data = np.load(item["path"], allow_pickle=False)
        if item["type"] in ("image", "source_person"):
            images.append(data[0] if data.ndim == 4 else data)
            if item["type"] == "source_person":
                source_count += 1
                prefix.append(f"<SourcePerson {source_count}> (original person to replace): <|image|>\n")
            else:
                picture_count += 1
                prefix.append(f"<Picture {picture_count}>: <|image|>\n")
        elif item["type"] == "video":
            video_count += 1
            timestamps = item["timestamps"]
            if len(timestamps) != len(data):
                raise ValueError("Video timestamps must match the sampled frames")
            indices = np.linspace(0, len(data) - 1, min(len(data), max_video_frames), dtype=int)
            for i in indices:
                images.append(data[i])
                prefix.append(f"<Video {video_count}> frame at {timestamps[i]:.3f}s: <|image|>\n")
    kwargs = {}
    if images:
        kwargs["images"] = images
    batch = processor(text="".join(prefix), return_tensors="pt", **kwargs)
    suffix = torch.tensor([ids], dtype=torch.long)
    offset = batch["input_ids"].shape[1]
    batch["input_ids"] = torch.cat([batch["input_ids"], suffix], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(suffix)], dim=1)
    if "token_type_ids" in batch:
        batch["token_type_ids"] = torch.cat([batch["token_type_ids"], torch.zeros_like(suffix)], dim=1)
    assert batch["input_ids"][0, offset:].tolist() == ids
    return batch, offset


def main():
    model_path = sys.argv[1]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print("READY", flush=True)
    processor = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "offload":
                model.to("cpu")
                torch.cuda.empty_cache()
                print("OK offload", flush=True)
                continue
            offset = 0
            if request.get("visuals"):
                if processor is None:
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                inputs, offset = prepare_inputs(
                    processor,
                    request["ids"],
                    request["visuals"],
                    request.get("max_video_frames", 8),
                )
                inputs = inputs.to("cuda")
            else:
                tokens = torch.tensor([request["ids"]], device="cuda")
                inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
            model.to("cuda")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            hidden = out.hidden_states[int(request["tap"])][0, offset:]
            if hidden.shape[0] != len(request["ids"]):
                raise RuntimeError("Visual/text alignment failed")
            np.save(request["out"], hidden.float().cpu().numpy().astype(np.float16))
            del out, hidden, inputs
            if request.get("offload_after"):
                model.to("cpu")
                torch.cuda.empty_cache()
            print("OK " + request["out"], flush=True)
        except Exception as exc:
            print("ERR " + str(exc).replace("\n", " ")[:2000], flush=True)


if __name__ == "__main__":
    main()
'''


def load_projection(path: str) -> dict:
    with safe_open(path, "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata() or {}
    tensors["tap"] = int(meta.get("tap", 24))
    return tensors


def build_residual(proj: dict, device: torch.device) -> Optional[torch.nn.Sequential]:
    layers = sorted(
        {int(k.split(".")[1]) for k in proj if isinstance(k, str) and k.startswith("mlp.")}
    )
    if not layers:
        return None
    modules = []
    for n, i in enumerate(layers):
        w = proj[f"mlp.{i}.weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=f"mlp.{i}.bias" in proj)
        lin.weight.data = w.to(device=device, dtype=torch.float32)
        if lin.bias is not None:
            lin.bias.data = proj[f"mlp.{i}.bias"].to(device=device, dtype=torch.float32)
        modules.append(lin)
        if n < len(layers) - 1:
            modules.append(torch.nn.GELU())
    net = torch.nn.Sequential(*modules).to(device)
    net.train(False)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class _MolmoWorkerProcess:
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, python_bin: str, model_path: str) -> "_MolmoWorkerProcess":
        with cls._lock:
            inst = cls._instances.get((python_bin, model_path))
            if inst is None or inst.proc.poll() is not None:
                inst = cls._instances[(python_bin, model_path)] = cls(python_bin, model_path)
            return inst

    @classmethod
    def stop_all(cls):
        with cls._lock:
            for key, inst in list(cls._instances.items()):
                try:
                    if inst.proc.poll() is None:
                        inst.proc.terminate()
                        inst.proc.wait(timeout=5)
                except Exception:
                    pass
            cls._instances.clear()

    def __init__(self, python_bin: str, model_path: str):
        self.proc = subprocess.Popen(
            [python_bin, "-c", _WORKER_CODE, model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        for line in self.proc.stdout:
            if line.startswith("READY"):
                break
        else:
            raise RuntimeError("Molmo2 worker exited before READY")
        self.lock = threading.Lock()

    def hidden(self, ids: List[int], tap: int, visuals=None, max_video_frames: int = 8) -> torch.Tensor:
        with self.lock, tempfile.TemporaryDirectory(prefix="aitk-molmo-") as folder:
            items = []
            for n, item in enumerate(visuals or []):
                path = os.path.join(folder, f"visual_{n}.npy")
                np.save(path, item["array"])
                items.append(
                    {
                        "type": item["type"],
                        "path": path,
                        "timestamps": item.get("timestamps", []),
                    }
                )
            out = os.path.join(folder, "h.npy")
            req = {
                "ids": ids,
                "tap": tap,
                "out": out,
                "visuals": items,
                "offload_after": False,
                "max_video_frames": max_video_frames,
            }
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline().strip()
            if not reply.startswith("OK "):
                raise RuntimeError("Molmo2 worker failed: " + reply)
            return torch.from_numpy(np.load(out))


class MolmoH3TextEncoder(torch.nn.Module):
    """Drop-in text_encoder module for MinimaxH3Model backed by Molmo2-8B + H3 projection."""

    def __init__(
        self,
        molmo_model_path: str,
        molmo_projection_path: str,
        molmo_python_path: Optional[str] = None,
        phase_b_vision: bool = True,
        max_video_frames: int = 8,
    ):
        super().__init__()
        from tokenizers import Tokenizer

        if not os.path.isfile(os.path.join(molmo_model_path, "tokenizer.json")):
            raise FileNotFoundError(
                f"Molmo2 tokenizer.json not found in {molmo_model_path}"
            )
        if not os.path.isfile(molmo_projection_path):
            raise FileNotFoundError(
                f"Molmo2 H3 projection not found: {molmo_projection_path}"
            )
        self.molmo_model_path = molmo_model_path
        self.molmo_projection_path = molmo_projection_path
        self.molmo_python_path = molmo_python_path or sys.executable
        self.phase_b_vision = bool(phase_b_vision)
        self.max_video_frames = int(max_video_frames)
        self.tokenizer = Tokenizer.from_file(os.path.join(molmo_model_path, "tokenizer.json"))
        self.proj = load_projection(molmo_projection_path)
        self.register_buffer("_dev_buf", torch.zeros(1, dtype=torch.float32), persistent=False)
        self._proj_cache = {}
        self.aitk_is_quantized = True

    @property
    def device(self) -> torch.device:
        return self._dev_buf.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dev_buf.dtype

    def aitk_post_load(self, **kwargs):
        return self

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        if self._dev_buf.device.type == "cpu":
            _MolmoWorkerProcess.stop_all()
        return res

    def unload_worker(self):
        _MolmoWorkerProcess.stop_all()

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        keyframes: Optional[List] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: Optional[int] = None,
    ):
        if device is None:
            device = self.device if self.device.type != "cpu" else torch.device("cuda")
        if dtype is None:
            dtype = torch.bfloat16

        # Sort keyframes so <Picture k> images come before <Video k> clips,
        # matching ComfyUI / Phase B calibration order.
        ordered_keyframes = []
        if keyframes:
            images = [k for k in keyframes if not isinstance(k, VideoRef)]
            videos = [k for k in keyframes if isinstance(k, VideoRef)]
            ordered_keyframes = images + videos

        visuals = []
        if self.phase_b_vision and ordered_keyframes:
            for item in ordered_keyframes:
                if isinstance(item, VideoRef):
                    arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in item.frames])
                    visuals.append(
                        {
                            "type": "video",
                            "array": arr,
                            "timestamps": [float(t) for t in item.timestamps],
                        }
                    )
                elif isinstance(item, Image.Image):
                    arr = np.asarray(item.convert("RGB"), dtype=np.uint8)
                    visuals.append({"type": "image", "array": arr})

        ids: List[int] = []
        if not visuals and ordered_keyframes:
            pic_idx = vid_idx = 0
            for item in ordered_keyframes:
                if isinstance(item, VideoRef):
                    vid_idx += 1
                    ids.extend(self.tokenizer.encode(f"<Video {vid_idx}>: ", add_special_tokens=False).ids)
                else:
                    pic_idx += 1
                    ids.extend(self.tokenizer.encode(f"<Picture {pic_idx}>: ", add_special_tokens=False).ids)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if max_length is not None and max_length > 0:
            prompt_ids = prompt_ids[:max_length]
        ids.extend(prompt_ids)
        if not ids:
            if "sink_out" in self.proj:
                embeds = self.proj["sink_out"].reshape(1, -1).to(device=device, dtype=dtype)
                tags = torch.full((1,), TEXT_TAG, dtype=torch.long)
                return embeds, tags
            ids = [151643]

        worker = _MolmoWorkerProcess.get(self.molmo_python_path, self.molmo_model_path)
        h = worker.hidden(
            ids=ids,
            tap=self.proj["tap"],
            visuals=visuals if visuals else None,
            max_video_frames=self.max_video_frames,
        )

        if self._proj_cache.get("device") != device:
            self._proj_cache = {
                "device": device,
                "p": {
                    k: self.proj[k].to(device=device, dtype=torch.float32)
                    for k in ("W", "mean_in", "std_in", "mean_out", "std_out")
                    if k in self.proj
                },
                "mlp": build_residual(self.proj, device),
            }
        p = self._proj_cache["p"]
        xn = (h.to(device=device, dtype=torch.float32) - p["mean_in"]) / p["std_in"]
        yn = xn @ p["W"] if "W" in p else None
        if self._proj_cache["mlp"] is not None:
            net = self._proj_cache["mlp"]
            r = net(xn.to(net[0].weight.dtype)).float()
            yn = r if yn is None else yn + r
        cond = yn * p["std_out"] + p["mean_out"]
        if "sink_out" in self.proj and cond.shape[0] > 0:
            sink = self.proj["sink_out"].to(device=cond.device, dtype=cond.dtype)
            if visuals:
                cond = torch.cat([sink.reshape(1, -1), cond], dim=0)
            else:
                cond[0] = sink

        embeds = cond.to(device=device, dtype=dtype)
        tags = torch.full((embeds.shape[0],), TEXT_TAG, dtype=torch.long)
        return embeds, tags
