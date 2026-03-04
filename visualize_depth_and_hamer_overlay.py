import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_depth_files(depth_dir: Path) -> List[Path]:
    # pred_depth_000123.npy
    files = sorted(depth_dir.glob("pred_depth_*.npy"))
    if files:
        return files
    # fallback: any .npy, sorted by trailing number if present
    any_npy = list(depth_dir.glob("*.npy"))
    pat = re.compile(r"(\d+)(?=\.npy$)")
    any_npy.sort(key=lambda p: int(pat.search(p.name).group(1)) if pat.search(p.name) else 10**18)
    return any_npy


def depth_to_vis(
    depth: np.ndarray,
    *,
    near_percentile: float = 1.0,
    far_percentile: float = 99.0,
    blur_ksize: int = 3,
) -> np.ndarray:
    d = depth.astype(np.float32, copy=True)
    mask = np.isfinite(d)
    if not np.any(mask):
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    vmin = float(np.percentile(d[mask], near_percentile))
    vmax = float(np.percentile(d[mask], far_percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-3

    d = np.clip(d, vmin, vmax)
    d_norm = (d - vmin) / (vmax - vmin)
    d_8u = (d_norm * 255).astype(np.uint8)

    if blur_ksize and blur_ksize > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        d_8u = cv2.GaussianBlur(d_8u, (k, k), 0)

    return cv2.applyColorMap(d_8u, cv2.COLORMAP_JET)


def _load_wrist2d(path: Optional[Path]) -> Dict[str, Tuple[float, float]]:
    if not path:
        return {}
    if not path.exists():
        return {}
    raw = _read_json(path)
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                out[str(k)] = (float(v[0]), float(v[1]))
            except Exception:
                continue
    return out


@dataclass
class VideoWriterCfg:
    fps: float
    size: Tuple[int, int]  # (w, h)


def _make_writer(path: Path, cfg: VideoWriterCfg) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = cfg.size
    return cv2.VideoWriter(str(path), fourcc, float(cfg.fps), (int(w), int(h)))


def _infer_video_path(root: Path, video_name: str) -> Optional[Path]:
    # Prefer depth/<video>_src.mp4 if present
    cand = root / "depth" / f"{video_name}_src.mp4"
    if cand.exists():
        return cand
    # Any mp4 under depth
    mp4s = list((root / "depth").glob("*.mp4")) if (root / "depth").exists() else []
    if mp4s:
        return sorted(mp4s)[0]
    # Any mp4 under root
    mp4s = list(root.glob("*.mp4"))
    if mp4s:
        return sorted(mp4s)[0]
    return None


def _draw_wrist_points(
    frame_bgr: np.ndarray,
    *,
    frame_key: str,
    wrist_L: Dict[str, Tuple[float, float]],
    wrist_R: Dict[str, Tuple[float, float]],
    radius: int = 6,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if frame_key in wrist_L:
        u, v = wrist_L[frame_key]
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        cv2.circle(out, (ui, vi), radius, (0, 0, 255), thickness=-1)  # left: red
        cv2.circle(out, (ui, vi), radius + 2, (255, 255, 255), thickness=1)
    if frame_key in wrist_R:
        u, v = wrist_R[frame_key]
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        cv2.circle(out, (ui, vi), radius, (0, 255, 0), thickness=-1)  # right: green
        cv2.circle(out, (ui, vi), radius + 2, (255, 255, 255), thickness=1)
    return out


def _hstack_resize(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    if ha != h:
        a = cv2.resize(a, (int(wa * (h / ha)), h))
    if hb != h:
        b = cv2.resize(b, (int(wb * (h / hb)), h))
    return np.concatenate([a, b], axis=1)


def run(
    *,
    root: Path,
    video_name: str,
    out_dir: Path,
    fps: Optional[float],
    near_percentile: float,
    far_percentile: float,
    blur_ksize: int,
    radius: int,
) -> None:
    depth_dir = root / "depth"
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth dir not found: {depth_dir}")

    depth_files = _sorted_depth_files(depth_dir)
    if not depth_files:
        raise FileNotFoundError(f"no depth npy found under: {depth_dir}")

    video_path = _infer_video_path(root, video_name)
    if not video_path:
        raise FileNotFoundError(
            f"cannot find source video; expected {root/'depth'/f'{video_name}_src.mp4'} or any .mp4"
        )

    wrist_L = _load_wrist2d(root / "hand2d_px" / f"{video_name}_left.json")
    wrist_R = _load_wrist2d(root / "hand2d_px" / f"{video_name}_right.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = float(fps) if fps is not None else (float(src_fps) if src_fps > 0 else 10.0)

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError(f"failed to read first frame from: {video_path}")
    H, W = frame0.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)
    depth_mp4 = out_dir / f"{video_name}_depth_vis.mp4"
    overlay_mp4 = out_dir / f"{video_name}_hamer_overlay.mp4"
    side_mp4 = out_dir / f"{video_name}_side_by_side.mp4"

    writer_depth = _make_writer(depth_mp4, VideoWriterCfg(fps=out_fps, size=(W, H)))
    writer_overlay = _make_writer(overlay_mp4, VideoWriterCfg(fps=out_fps, size=(W, H)))

    # Side-by-side width depends on resize, compute using first depth frame
    depth0 = np.load(depth_files[0]).astype(np.float32)
    depth0_vis = depth_to_vis(depth0, near_percentile=near_percentile, far_percentile=far_percentile, blur_ksize=blur_ksize)
    if depth0_vis.shape[:2] != (H, W):
        depth0_vis = cv2.resize(depth0_vis, (W, H))
    side0 = _hstack_resize(frame0, depth0_vis)
    side_h, side_w = side0.shape[:2]
    writer_side = _make_writer(side_mp4, VideoWriterCfg(fps=out_fps, size=(side_w, side_h)))

    # rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(depth_files)), len(depth_files))
    for i in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        depth = np.load(depth_files[i]).astype(np.float32)
        dvis = depth_to_vis(depth, near_percentile=near_percentile, far_percentile=far_percentile, blur_ksize=blur_ksize)
        if dvis.shape[:2] != (H, W):
            dvis = cv2.resize(dvis, (W, H))

        # EgoLoc wrist JSON is 1-based in existing scripts
        frame_key = str(i + 1)
        overlay = _draw_wrist_points(frame, frame_key=frame_key, wrist_L=wrist_L, wrist_R=wrist_R, radius=radius)

        writer_depth.write(dvis)
        writer_overlay.write(overlay)
        side = _hstack_resize(overlay, dvis)
        writer_side.write(side)

    writer_depth.release()
    writer_overlay.release()
    writer_side.release()
    cap.release()

    print(f"[SAVE] depth video: {depth_mp4}")
    print(f"[SAVE] overlay video: {overlay_mp4}")
    print(f"[SAVE] side-by-side: {side_mp4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="video output root, e.g. .../video32")
    ap.add_argument("--video_name", type=str, default=None, help="default: basename(root)")
    ap.add_argument("--out_dir", type=str, default=None, help="default: <root>/vis_depth_hamer")
    ap.add_argument("--fps", type=float, default=None, help="override output fps (default: src video fps)")
    ap.add_argument("--near_p", type=float, default=1.0)
    ap.add_argument("--far_p", type=float, default=99.0)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--radius", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    video_name = args.video_name or root.name
    out_dir = Path(args.out_dir) if args.out_dir else (root / "vis_depth_hamer")

    run(
        root=root,
        video_name=video_name,
        out_dir=out_dir,
        fps=args.fps,
        near_percentile=args.near_p,
        far_percentile=args.far_p,
        blur_ksize=args.blur,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()

