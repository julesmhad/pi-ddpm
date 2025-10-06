import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tif
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

__all__ = [
    "SimMetadata",
    "SimPatchDataset",
    "load_sim_stacks",
    "batch_iterator",
]


@dataclass
class SimMetadata:
    pixel_size_nm_xy: Optional[float]
    pixel_size_nm_z: Optional[float]
    exposure_ms: Optional[float]
    objective_na: Optional[float]
    refractive_index: Optional[float]
    ordering_map: Dict[int, Tuple[int, int]]
    otf_id: Optional[str]
    file_id: str
    angles: int
    phases: int


class SimPatchDataset:
    """Container for SIM patches with metadata and helper API."""

    def __init__(
        self,
        raw: np.ndarray,
        targets: Optional[np.ndarray],
        psf: np.ndarray,
        patch_meta: List[Dict[str, Any]],
        stack_meta: SimMetadata,
        patch_mode: str,
    ) -> None:
        self.x = raw.astype(np.float32)
        self.y = targets.astype(np.float32) if targets is not None else None
        self.k = psf.astype(np.complex64)
        self._patch_meta = patch_meta
        self._stack_meta = stack_meta
        self.patch_mode = patch_mode

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = {**asdict(self._stack_meta), **self._patch_meta[idx]}
        payload: Dict[str, Any] = {
            "raw": self.x[idx],
            "meta": meta,
            "psf": self.k[idx],
        }
        if self.y is not None:
            payload["recon_gt"] = self.y[idx]
        else:
            payload["recon_gt"] = None
        return payload


def _parse_resolution_tag(tag_value: Sequence[int]) -> Optional[float]:
    if tag_value is None:
        return None
    try:
        num, den = tag_value
        if den == 0:
            return None
        return float(den) / float(num)
    except Exception:
        return None


def _parse_ome_metadata(meta: Optional[str]) -> Dict[str, Any]:
    if meta is None:
        return {}
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(meta)
    except ET.ParseError:
        return {}

    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return {}

    def _extract_float(attr: str) -> Optional[float]:
        value = pixels.attrib.get(attr)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    physical_x = _extract_float("PhysicalSizeX")
    physical_y = _extract_float("PhysicalSizeY")
    physical_z = _extract_float("PhysicalSizeZ")

    objective = root.find(".//ome:Objective", ns)
    na = None
    refr_index = None
    if objective is not None:
        na_attr = objective.attrib.get("LensNA")
        ri_attr = objective.attrib.get("Immersion")
        if na_attr is not None:
            try:
                na = float(na_attr)
            except ValueError:
                na = None
        if ri_attr is not None:
            try:
                refr_index = float(ri_attr)
            except ValueError:
                refr_index = None

    exposure = None
    channel = root.find(".//ome:Channel", ns)
    if channel is not None:
        exp_attr = channel.attrib.get("ExposureTime")
        if exp_attr is not None:
            try:
                exposure = float(exp_attr)
            except ValueError:
                exposure = None

    return {
        "pixel_size_x_um": physical_x,
        "pixel_size_y_um": physical_y,
        "pixel_size_z_um": physical_z,
        "objective_na": na,
        "refractive_index": refr_index,
        "exposure_ms": exposure,
    }


def _extract_metadata(raw_path: str) -> Dict[str, Any]:
    with tif.TiffFile(raw_path) as tif_file:
        ome_dict = _parse_ome_metadata(tif_file.ome_metadata)
        page = tif_file.pages[0]
        x_res = _parse_resolution_tag(getattr(page.tags.get("XResolution"), "value", None))
        y_res = _parse_resolution_tag(getattr(page.tags.get("YResolution"), "value", None))

    meta: Dict[str, Any] = {
        "pixel_size_nm_xy": None,
        "pixel_size_nm_z": None,
        "exposure_ms": ome_dict.get("exposure_ms"),
        "objective_na": ome_dict.get("objective_na"),
        "refractive_index": ome_dict.get("refractive_index"),
    }

    # Prefer OME metadata; fallback to resolution tags if necessary.
    pix_um_x = ome_dict.get("pixel_size_x_um") or (x_res if x_res else None)
    pix_um_y = ome_dict.get("pixel_size_y_um") or (y_res if y_res else None)
    if pix_um_x is not None and pix_um_y is not None:
        meta["pixel_size_nm_xy"] = float(np.mean([pix_um_x, pix_um_y]) * 1000.0)
    elif pix_um_x is not None:
        meta["pixel_size_nm_xy"] = float(pix_um_x * 1000.0)
    elif pix_um_y is not None:
        meta["pixel_size_nm_xy"] = float(pix_um_y * 1000.0)

    pix_um_z = ome_dict.get("pixel_size_z_um")
    if pix_um_z is not None:
        meta["pixel_size_nm_z"] = float(pix_um_z * 1000.0)

    return meta


def _reshape_frames(frames: np.ndarray, angles: int, phases: int) -> Tuple[np.ndarray, str, Dict[int, Tuple[int, int]], float]:
    frames = np.asarray(frames)
    total = frames.shape[0]
    per_plane = angles * phases
    if total % per_plane != 0:
        raise ValueError(
            f"Raw stack does not contain an integer number of SIM planes: {total} frames vs {per_plane} per plane"
        )

    z_planes = total // per_plane
    frames = frames.reshape(z_planes, per_plane, *frames.shape[1:])

    def angle_major(plane: np.ndarray) -> np.ndarray:
        return plane.reshape(angles, phases, *plane.shape[1:])

    def phase_major(plane: np.ndarray) -> np.ndarray:
        reshaped = plane.reshape(phases, angles, *plane.shape[1:])
        return np.transpose(reshaped, (1, 0, 2, 3))

    candidates = {
        "angle_major": angle_major,
        "phase_major": phase_major,
    }

    def candidate_score(volume: np.ndarray) -> float:
        # volume -> (Z, A, P, Y, X)
        vol = volume.astype(np.float32)
        vol_flat = vol.reshape(vol.shape[0], vol.shape[1], vol.shape[2], -1)
        scores = []
        for z in range(vol_flat.shape[0]):
            for angle in range(vol_flat.shape[1]):
                angle_frames = vol_flat[z, angle]
                mean_frames = angle_frames - angle_frames.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(mean_frames, axis=1) + 1e-8
                corr = 0.0
                pairs = 0
                for i in range(len(mean_frames)):
                    for j in range(i + 1, len(mean_frames)):
                        corr += float(np.dot(mean_frames[i], mean_frames[j]) / (norms[i] * norms[j]))
                        pairs += 1
                if pairs:
                    scores.append(corr / pairs)
        if not scores:
            return 0.0
        return float(np.mean(scores))

    best_label = ""
    best_volume: Optional[np.ndarray] = None
    best_score = -np.inf
    scores: Dict[str, float] = {}
    for label, fn in candidates.items():
        vol = np.stack([fn(frames[z]) for z in range(z_planes)], axis=0)
        sc = candidate_score(vol)
        scores[label] = sc
        if sc > best_score:
            best_score = sc
            best_label = label
            best_volume = vol

    if best_volume is None:
        raise RuntimeError("Unable to determine ordering for SIM frames")

    score_margin = best_score - max(v for k, v in scores.items() if k != best_label)
    if score_margin < 1e-3:
        raise RuntimeError("SIM ordering inference ambiguous; score gap too small")

    if best_label == "angle_major":
        ordering = {idx: (idx // phases, idx % phases) for idx in range(per_plane)}
    else:
        ordering = {idx: (idx % angles, idx // angles) for idx in range(per_plane)}

    return best_volume.astype(np.float32), best_label, ordering, score_margin


def _register_plane(frames: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    registered = frames.copy()
    reference = frames[0, 0]
    for angle in range(frames.shape[0]):
        for phase in range(frames.shape[1]):
            if angle == 0 and phase == 0:
                continue
            shift, _, _ = phase_cross_correlation(reference, frames[angle, phase], upsample_factor=10)
            if float(np.linalg.norm(shift)) > threshold:
                registered[angle, phase] = ndi_shift(frames[angle, phase], shift, order=1, mode="reflect")
    return registered


def _apply_registration(volume: np.ndarray, threshold: float) -> np.ndarray:
    registered = np.empty_like(volume)
    for z in range(volume.shape[0]):
        registered[z] = _register_plane(volume[z], threshold=threshold)
    return registered


def _normalize_volume(volume: np.ndarray, percentile: Tuple[float, float]) -> np.ndarray:
    low, high = percentile
    normed = np.empty_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        plane = volume[z]
        lo = np.percentile(plane, low)
        hi = np.percentile(plane, high)
        if hi <= lo:
            normed[z] = np.zeros_like(plane, dtype=np.float32)
            continue
        plane_norm = (plane - lo) / (hi - lo)
        normed[z] = np.clip(plane_norm, 0.0, 1.0) * 2.0 - 1.0
    return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=-1.0)


def _normalize_gt(gt: np.ndarray, percentile: Tuple[float, float]) -> np.ndarray:
    if gt is None:
        return gt
    low, high = percentile
    normed = np.empty_like(gt, dtype=np.float32)
    for z in range(gt.shape[0]):
        plane = gt[z]
        lo = np.percentile(plane, low)
        hi = np.percentile(plane, high)
        if hi <= lo:
            normed[z] = np.zeros_like(plane, dtype=np.float32)
            continue
        plane_norm = (plane - lo) / (hi - lo)
        normed[z] = np.clip(plane_norm, 0.0, 1.0) * 2.0 - 1.0
    return np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=-1.0)


def _ensure_spatial_alignment(raw_xy: Tuple[int, int], gt: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if gt is None:
        return None
    from skimage.transform import resize

    target_y, target_x = raw_xy
    aligned = np.empty((gt.shape[0], target_y, target_x), dtype=np.float32)
    for z in range(gt.shape[0]):
        aligned[z] = resize(
            gt[z],
            (target_y, target_x),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
    return aligned.astype(np.float32)


def _make_indices(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    last = length - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def _make_gaussian_kernel(size: Sequence[int], sigma: float) -> np.ndarray:
    size_arr = np.array(size, dtype=int)
    grids = np.meshgrid(*[np.linspace(-1.0, 1.0, s) for s in size_arr], indexing="ij")
    squared = sum(g ** 2 for g in grids)
    sigma_scaled = (sigma / (size_arr / 2.0)) ** 2
    sigma_term = sum(sigma_scaled)
    kernel = np.exp(-squared / (2.0 * sigma_term))
    kernel /= kernel.sum() + 1e-8
    return kernel.astype(np.float32)


def _reject_patch(patch: np.ndarray, mean_thr: float, var_thr: float) -> bool:
    mean_val = float(np.mean((patch + 1.0) * 0.5))
    var_val = float(np.var((patch + 1.0) * 0.5))
    if mean_val < mean_thr:
        return True
    if var_val < var_thr:
        return True
    return False


def _extract_2d_patches(
    volume: np.ndarray,
    gt: Optional[np.ndarray],
    patch_hw: Tuple[int, int],
    stride_hw: Tuple[int, int],
    mean_thr: float,
    var_thr: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    z_planes, angles, phases, height, width = volume.shape
    patch_h = min(patch_hw[0], height)
    patch_w = min(patch_hw[1], width)
    stride_y = max(1, stride_hw[0])
    stride_x = max(1, stride_hw[1])
    y_positions = _make_indices(height, patch_h, stride_y)
    x_positions = _make_indices(width, patch_w, stride_x)

    raw_patches: List[np.ndarray] = []
    gt_patches: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for z in range(z_planes):
        for y0 in y_positions:
            for x0 in x_positions:
                raw_patch = volume[z, :, :, y0 : y0 + patch_h, x0 : x0 + patch_w]
                if _reject_patch(raw_patch, mean_thr, var_thr):
                    continue
                raw_patches.append(raw_patch)
                metas.append({"z_index": z, "y0": y0, "x0": x0, "patch_mode": "2d"})
                if gt is not None:
                    gt_slice = gt[z]
                    gt_patch = gt_slice[y0 : y0 + patch_h, x0 : x0 + patch_w]
                    gt_patches.append(gt_patch)

    if not raw_patches:
        raise RuntimeError("No valid 2D patches extracted; adjust thresholds or stride")

    x_arr = np.stack(raw_patches, axis=0).astype(np.float32)
    y_arr = np.stack(gt_patches, axis=0).astype(np.float32) if gt is not None else None
    return x_arr, y_arr, metas


def _extract_3d_patches(
    volume: np.ndarray,
    gt: Optional[np.ndarray],
    patch_dhw: Tuple[int, int, int],
    stride_dhw: Tuple[int, int, int],
    mean_thr: float,
    var_thr: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    z_planes, angles, phases, height, width = volume.shape
    patch_d = min(patch_dhw[0], z_planes)
    patch_h = min(patch_dhw[1], height)
    patch_w = min(patch_dhw[2], width)
    stride_d = max(1, stride_dhw[0])
    stride_y = max(1, stride_dhw[1])
    stride_x = max(1, stride_dhw[2])

    z_positions = _make_indices(z_planes, patch_d, stride_d)
    y_positions = _make_indices(height, patch_h, stride_y)
    x_positions = _make_indices(width, patch_w, stride_x)

    raw_patches: List[np.ndarray] = []
    gt_patches: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for z0 in z_positions:
        for y0 in y_positions:
            for x0 in x_positions:
                raw_patch = volume[
                    z0 : z0 + patch_d,
                    :,
                    :,
                    y0 : y0 + patch_h,
                    x0 : x0 + patch_w,
                ]
                if _reject_patch(raw_patch, mean_thr, var_thr):
                    continue
                raw_patches.append(raw_patch)
                metas.append(
                    {
                        "z_index": (z0, z0 + patch_d),
                        "y0": y0,
                        "x0": x0,
                        "patch_mode": "3d",
                    }
                )
                if gt is not None:
                    gt_slice = gt[z0 : z0 + patch_d]
                    gt_patch = gt_slice[:, y0 : y0 + patch_h, x0 : x0 + patch_w]
                    gt_patches.append(gt_patch)

    if not raw_patches:
        raise RuntimeError("No valid 3D patches extracted; adjust thresholds or stride")

    x_arr = np.stack(raw_patches, axis=0).astype(np.float32)
    y_arr = np.stack(gt_patches, axis=0).astype(np.float32) if gt is not None else None
    return x_arr, y_arr, metas


def _split_indices(z_planes: int, fractions: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
    if not np.isclose(sum(fractions), 1.0, atol=1e-3):
        raise ValueError("Fractions for dataset split must sum to 1")

    pattern = ("train",) * 14 + ("val",) * 3 + ("test",) * 3
    assignments: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for idx in range(z_planes):
        bucket = pattern[idx % len(pattern)]
        assignments[bucket].append(idx)

    return {
        key: np.array(sorted(vals), dtype=int)
        for key, vals in assignments.items()
        if vals
    }


def batch_iterator(
    dataset: SimPatchDataset,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
    if len(dataset) == 0:
        raise ValueError("Cannot iterate over an empty dataset")
    rng = np.random.default_rng()
    while True:
        indices = np.arange(len(dataset))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(dataset), batch_size):
            sel = indices[start : start + batch_size]
            raw = dataset.x[sel]
            psf = dataset.k[sel]
            targets = dataset.y[sel] if dataset.y is not None else None
            yield raw, targets, psf


def load_sim_stacks(
    path_widefield_tif: str,
    path_gt_tif: Optional[str] = None,
    *,
    angles_expected: int = 3,
    phases_expected: int = 5,
    patch_mode: str = "2d",
    patch_shape: Optional[Tuple[int, ...]] = None,
    stride: Optional[Tuple[int, ...]] = None,
    z_fraction_splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    mean_threshold: float = 0.05,
    variance_threshold: float = 1e-4,
    drift_threshold_px: float = 0.2,
    intensity_percentiles: Tuple[float, float] = (0.1, 99.9),
    gaussian_psf_size: Optional[Tuple[int, ...]] = None,
    gaussian_psf_sigma: float = 6.0,
) -> Dict[str, SimPatchDataset]:
    if not os.path.exists(path_widefield_tif):
        raise FileNotFoundError(f"Widefield SIM stack not found: {path_widefield_tif}")

    raw_meta = _extract_metadata(path_widefield_tif)

    with tif.TiffFile(path_widefield_tif) as tif_file:
        raw_stack = tif_file.asarray().astype(np.float32)
        saturation_ratio = np.mean(raw_stack >= np.max(raw_stack))
        if saturation_ratio > 0.01:
            raise RuntimeError(
                f"Input stack saturation exceeds 1% (ratio {saturation_ratio:.3f}); check acquisition"
            )
        reshaped = raw_stack.reshape(-1, raw_stack.shape[-2], raw_stack.shape[-1])

    volume, ordering_label, ordering_map, score_margin = _reshape_frames(
        reshaped, angles_expected, phases_expected
    )

    volume = _apply_registration(volume, threshold=drift_threshold_px)
    volume = _normalize_volume(volume, intensity_percentiles)

    gt_volume = None
    if path_gt_tif is not None and os.path.exists(path_gt_tif):
        with tif.TiffFile(path_gt_tif) as gt_file:
            gt_raw = gt_file.asarray().astype(np.float32)
        if gt_raw.ndim == 2:
            gt_raw = gt_raw[None, ...]
        if gt_raw.ndim == 3:
            pass
        else:
            gt_raw = gt_raw.reshape(-1, *gt_raw.shape[-2:])
        gt_volume = _ensure_spatial_alignment(volume.shape[-2:], gt_raw)
        gt_volume = _normalize_gt(gt_volume, intensity_percentiles)

    metadata = SimMetadata(
        pixel_size_nm_xy=raw_meta.get("pixel_size_nm_xy"),
        pixel_size_nm_z=raw_meta.get("pixel_size_nm_z"),
        exposure_ms=raw_meta.get("exposure_ms"),
        objective_na=raw_meta.get("objective_na"),
        refractive_index=raw_meta.get("refractive_index"),
        ordering_map=ordering_map,
        otf_id=None,
        file_id=os.path.basename(path_widefield_tif),
        angles=angles_expected,
        phases=phases_expected,
    )

    splits = _split_indices(volume.shape[0], z_fraction_splits)

    if patch_mode not in {"2d", "3d"}:
        raise ValueError("patch_mode must be either \"2d\" or \"3d\"")

    if patch_mode == "2d":
        default_patch = (256, 256)
        patch_hw = patch_shape if patch_shape is not None else default_patch
        if len(patch_hw) != 2:
            raise ValueError("2D patch_shape must be a tuple of (H, W)")
        stride_hw = stride if stride is not None else tuple(max(1, s // 2) for s in patch_hw)
    else:
        default_patch = (16, 256, 256)
        patch_dhw = patch_shape if patch_shape is not None else default_patch
        if len(patch_dhw) != 3:
            raise ValueError("3D patch_shape must be a tuple of (D, H, W)")
        stride_dhw = stride if stride is not None else tuple(max(1, s // 2) for s in patch_dhw)

    psf_size = gaussian_psf_size
    if psf_size is None:
        if patch_mode == "2d":
            psf_size = (min(64, volume.shape[-2]), min(64, volume.shape[-1]))
        else:
            psf_size = (min(16, volume.shape[0]), min(64, volume.shape[-2]), min(64, volume.shape[-1]))

    datasets: Dict[str, SimPatchDataset] = {}
    for split_name, z_idx in splits.items():
        sub_volume = volume[z_idx]
        sub_gt = gt_volume[z_idx] if gt_volume is not None else None
        if patch_mode == "2d":
            x_arr, y_arr, metas = _extract_2d_patches(
                sub_volume,
                sub_gt,
                patch_hw=patch_hw,
                stride_hw=stride_hw,
                mean_thr=mean_threshold,
                var_thr=variance_threshold,
            )
            kernel = _make_gaussian_kernel(psf_size, gaussian_psf_sigma)
            psf_arr = np.repeat(kernel[None, ...], x_arr.shape[0], axis=0)
        else:
            x_arr, y_arr, metas = _extract_3d_patches(
                sub_volume,
                sub_gt,
                patch_dhw=patch_dhw,
                stride_dhw=stride_dhw,
                mean_thr=mean_threshold,
                var_thr=variance_threshold,
            )
            kernel = _make_gaussian_kernel(psf_size, gaussian_psf_sigma)
            psf_arr = np.repeat(kernel[None, ...], x_arr.shape[0], axis=0)

        datasets[split_name] = SimPatchDataset(
            raw=x_arr,
            targets=y_arr,
            psf=psf_arr[..., None],
            patch_meta=metas,
            stack_meta=metadata,
            patch_mode=patch_mode,
        )

    datasets["ordering_label"] = ordering_label
    datasets["ordering_margin"] = score_margin
    return datasets

