import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import utonia

try:
    import flash_attn
except ImportError:
    flash_attn = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_bin_paths(path: str, frame: int | None) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    velodyne = path / "velodyne" if (path / "velodyne").is_dir() else path
    if frame is None:
        bins = sorted(velodyne.glob("*.bin"))
        if not bins:
            raise FileNotFoundError(f"No .bin files found in {velodyne}")
        return bins
    bin_path = velodyne / f"{frame:06d}.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    return [bin_path]


def load_xyz(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3].copy()


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


def project_color(proj: torch.Tensor) -> torch.Tensor:
    return proj[:, :3] * 0.4 + proj[:, 3:6] * 0.2 + proj[:, 9:12] * 0.4


def prepare_point(coord: np.ndarray):
    point = {
        "coord": coord.copy(),
        "color": np.zeros_like(coord),
        "normal": np.zeros_like(coord),
    }
    return utonia.transform.default(0.2, apply_z_positive=False)(point)


def infer_feat(model, coord: np.ndarray) -> torch.Tensor:
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
        return point.feat[point.inverse]


def sample_paths(bin_paths: list[Path], count: int) -> list[Path]:
    count = min(count, len(bin_paths))
    if count == len(bin_paths):
        return bin_paths
    idx = np.linspace(0, len(bin_paths) - 1, num=count, dtype=int)
    return [bin_paths[i] for i in idx]


def fit_pca(feats: list[torch.Tensor]):
    feat = torch.cat(feats, dim=0)
    sample_idx = torch.linspace(0, feat.shape[0] - 1, steps=min(100000, feat.shape[0])).long()
    feat = feat[sample_idx]
    mean = feat.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(feat - mean, q=12, niter=5, center=False)
    color = project_color((feat - mean) @ v)
    lo = torch.quantile(color, 0.01, dim=0)
    hi = torch.quantile(color, 0.99, dim=0)
    return mean, v, lo, hi


def colorize(feat: torch.Tensor, mean, v, lo, hi) -> np.ndarray:
    mean = mean.to(feat.device)
    v = v.to(feat.device)
    lo = lo.to(feat.device)
    hi = hi.to(feat.device)
    color = project_color((feat - mean) @ v)
    color = (color - lo) / (hi - lo + 1e-6)
    return color.clamp(0.0, 1.0).cpu().numpy()


def fit_features(model, bin_paths: list[Path]) -> list[torch.Tensor]:
    feats = []
    for path in bin_paths:
        feats.append(infer_feat(model, load_xyz(path)).detach().cpu())
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="SemanticKITTI .bin file or sequence directory")
    parser.add_argument("--frame", type=int, help="Frame id, e.g. 300 -> 000300.bin")
    parser.add_argument("--fit-frames", type=int, default=16)
    args = parser.parse_args()

    bin_paths = resolve_bin_paths(args.path, args.frame)
    utonia.utils.set_seed(6985480)
    model = load_model()
    fit_paths = sample_paths(bin_paths, args.fit_frames)
    mean, v, lo, hi = fit_pca(fit_features(model, fit_paths))

    rr.init("utonia_pca", spawn=True)
    for i, bin_path in enumerate(bin_paths):
        coord = load_xyz(bin_path)
        colors = colorize(infer_feat(model, coord), mean, v, lo, hi)
        rr.log(
            "points",
            rr.Points3D(coord, colors=(colors * 255).astype(np.uint8)),
        )


if __name__ == "__main__":
    main()
