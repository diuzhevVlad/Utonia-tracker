import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import utonia
from matplotlib import cm

try:
    import flash_attn
except ImportError:
    flash_attn = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_bin_paths(path: str) -> list[Path]:
    path = Path(path)
    velodyne = path / "velodyne" if (path / "velodyne").is_dir() else path
    bins = sorted(velodyne.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin files found in {velodyne}")
    return bins


def load_xyz(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy()


def load_model():
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia")
    else:
        model = utonia.load(
            "utonia",
            repo_id="Pointcept/Utonia",
            custom_config={"enc_patch_size": [1024] * 5, "enable_flash": False},
        )
    return model.to(DEVICE).eval()


def prepare_point(coord: np.ndarray):
    point = {
        "coord": coord.copy(),
        "color": np.zeros_like(coord),
        "normal": np.zeros_like(coord),
    }
    return utonia.transform.default(0.2, apply_z_positive=False)(point)


def infer_feat(model, coord: np.ndarray):
    point = prepare_point(coord)
    with torch.inference_mode():
        for key, value in point.items():
            if isinstance(value, torch.Tensor) and DEVICE == "cuda":
                point[key] = value.cuda(non_blocking=True)
        point = model(point)
        for _ in range(2):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point.feat[point.inverse], coord


def heat_color(sim: torch.Tensor) -> np.ndarray:
    sim = ((sim + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(1)
    return cm.plasma(sim.squeeze(1).cpu().numpy())[:, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="SemanticKITTI sequence directory")
    parser.add_argument("--query-x", type=float, default=-7.18)
    parser.add_argument("--query-y", type=float, default=0.652)
    parser.add_argument("--query-z", type=float, default=-1.154)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--normalize-in-frame",
        action="store_false",
        help="Rescale similarity in each frame so the frame minimum becomes 0 and maximum becomes 1.",
    )
    args = parser.parse_args()

    utonia.utils.set_seed(6985480)
    model = load_model()
    bin_paths = resolve_bin_paths(args.path)

    first_coord = load_xyz(bin_paths[0])
    first_feat, _ = infer_feat(model, first_coord)
    query_coord = np.array([args.query_x, args.query_y, args.query_z], dtype=np.float32)
    query_index = np.argmin(np.sum((first_coord - query_coord) ** 2, axis=1))
    query_feat = F.normalize(first_feat[query_index], dim=0)

    rr.init("utonia_similarity", spawn=True)
    rr.log("query", rr.Points3D(first_coord[[query_index]], colors=np.array([[0, 255, 0]], dtype=np.uint8)))

    for bin_path in bin_paths:
        coord = load_xyz(bin_path)
        feat, _ = infer_feat(model, coord)
        sim = F.normalize(feat, dim=1) @ query_feat
        if args.normalize_in_frame:
            sim = (sim - sim.min()) / (sim.max() - sim.min()).clamp_min(1e-6)
        if args.threshold is not None:
            sim = (sim > args.threshold).float()
        colors = (heat_color(sim) * 255).astype(np.uint8)
        rr.log("points", rr.Points3D(coord, colors=colors))


if __name__ == "__main__":
    main()
