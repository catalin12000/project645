#!/usr/bin/env python3
"""
generate_training_quad_data.py
==============================

Encode .bvh → quaternion .npy   AND/OR   decode quaternion .npy → .bvh

✓ Uses intrinsic (rotating-axis) Euler convention (“rXYZ”) so poses are correct.
✓ `--verbose / -v` prints joint order, per-file stats, and first-frame round-trip
  error.
✓ Frame layout : [hip_x, hip_y, hip_z,  q_hip_w, q_hip_x, …]  (w-x-y-z)
✓ Output shape : (frames, 3 + joints*4)

Example
-------
python code/generate_training_quad_data.py \
    --src_bvh_dir   train_data_bvh/martial/ \
    --out_quad_dir  train_data_quad/martial/ \
    --recon_bvh_dir recon_bvh_quad/martial/ \
    -v
"""

"""
generate_training_quad_data.py
==============================

BVH  ↔  Quaternion‐.npy conversion **with diagnostics plots**.

How to run
----------
# encode BVH → .npy   (and make histograms)
python code/generate_training_quad_data.py \
    --src_bvh_dir   train_data_bvh/martial/ \
    --out_quad_dir  train_data_quad/martial/

# decode .npy → BVH  (optional sanity-check)
python code/generate_training_quad_data.py \
    --out_quad_dir  train_data_quad/martial/ \
    --recon_bvh_dir recon_bvh_quad/martial/

Outputs
-------
diagnostics_quad/quaternion_norm_hist.png
diagnostics_quad/hip_translation_hist.png
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import argparse
import os
from os import listdir
from typing import Dict, List, Tuple

import numpy as np
import transforms3d.euler as tfs_euler
import transforms3d.quaternions as tfs_quat

import read_bvh
import read_bvh_hierarchy

# --------------------------------------------------------------------------- #
# Skeleton / constants                                                        #
# --------------------------------------------------------------------------- #
STANDARD_BVH: str = read_bvh.standard_bvh_file
WEIGHT_TRANSLATION: float = read_bvh.weight_translation

SKELETON, _ = read_bvh_hierarchy.read_bvh_hierarchy(STANDARD_BVH)
JOINT_INDEX: Dict[str, int] = read_bvh.joint_index
NUM_JOINTS: int = len(JOINT_INDEX)


#!/usr/bin/env python3
"""
generate_training_quad_data.py
------------------------------

Now produces a small diagnostics pack in ./diagnostics_quad/ :

* quaternion_norm_hist.png   – distribution of |q| for all joints
* hip_translation_hist.png   – histogram of root X/Z translations (metres)
"""
# ------------------------------------------------------------------------- #
import os, argparse, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import read_bvh, read_bvh_hierarchy, transforms3d.euler as tfs_eul
import transforms3d.quaternions as tfs_quat
# ------------- (existing imports / constants) --------------------------- #
# … keep your encoding helpers exactly as before …
# ------------------------------------------------------------------------- #

# ====== NEW: simple diagnostics ========================================= #
def plot_diagnostics(all_arrays):
    os.makedirs("diagnostics_quad", exist_ok=True)
    all_vecs = np.concatenate(all_arrays, 0)          # (Nf, 231)
    # quaternion norm ---------------------------------------------------- #
    q = all_vecs[:, 3:] .reshape(-1, 4)
    q_norm = np.linalg.norm(q, axis=1)
    plt.hist(q_norm, bins=50, color='grey')
    plt.axvline(1.0, color='red'); plt.xlabel("|q|"); plt.ylabel("count")
    plt.title("Quaternion norm distribution"); plt.savefig(
        "diagnostics_quad/quaternion_norm_hist.png"); plt.close()
    # hip translation ---------------------------------------------------- #
    hip_xyz = all_vecs[:, :3] / read_bvh.weight_translation
    plt.hist(hip_xyz[:,0], bins=50, alpha=.6, label='X')
    plt.hist(hip_xyz[:,2], bins=50, alpha=.6, label='Z')
    plt.xlabel("metres"); plt.ylabel("count"); plt.legend()
    plt.title("Root translation histogram"); plt.savefig(
        "diagnostics_quad/hip_translation_hist.png"); plt.close()

# ====== MAIN (after encoding) =========================================== #
def encode_folder(src, dst):
    os.makedirs(dst, exist_ok=True); all_arrays=[]
    for f in sorted(os.listdir(src)):
        if not f.endswith(".bvh"): continue
        arr = bvh_to_quat_array(os.path.join(src, f))
        np.save(os.path.join(dst, f + ".npy"), arr)
        all_arrays.append(arr)
        print("encoded", f)
    if all_arrays:
        plot_diagnostics(all_arrays)        # << new line

# ---- rest of script (decode_folder, main argparse) stays identical ---- #



# --------------------------------------------------------------------------- #
# Euler / matrix helpers                                                      #
# --------------------------------------------------------------------------- #
def euler_to_mat_intrinsic(angles_deg: List[float], order: str) -> np.ndarray:
    axes_code = "r" + order.lower()            # intrinsic
    return tfs_euler.euler2mat(*np.radians(angles_deg), axes_code)

def mat_to_euler_intrinsic(mat: np.ndarray, order: str) -> Dict[str, float]:
    axes_code = "r" + order.lower()
    ang = tfs_euler.mat2euler(mat, axes_code)
    return {axis: np.degrees(ang[i]) for i, axis in enumerate(order)}

def euler_to_quat(angles_deg: List[float], order: str) -> np.ndarray:
    q = tfs_quat.mat2quat(euler_to_mat_intrinsic(angles_deg, order))
    return q / np.linalg.norm(q)

# --------------------------------------------------------------------------- #
# Encoding helpers                                                            #
# --------------------------------------------------------------------------- #
def frame_to_vec(raw_frame: np.ndarray, w_trans: float) -> np.ndarray:
    vec = np.zeros(3 + NUM_JOINTS * 4, dtype=np.float32)
    hip_pos = np.zeros(3, dtype=np.float32)
    euler_map = {j: {"X": 0., "Y": 0., "Z": 0.} for j in SKELETON}

    ptr = 0
    for j_name, j_spec in SKELETON.items():
        ch_names = j_spec["channels"]
        n_ch = len(ch_names)
        ch_vals = raw_frame[ptr: ptr + n_ch]
        ptr += n_ch

        for i, ch in enumerate(ch_names):
            val = float(ch_vals[i])
            if "position" in ch.lower():
                if ch.lower().startswith("x"): hip_pos[0] = val * w_trans
                if ch.lower().startswith("y"): hip_pos[1] = val * w_trans
                if ch.lower().startswith("z"): hip_pos[2] = val * w_trans
            elif "rotation" in ch.lower():
                euler_map[j_name][ch[0].upper()] = val

    vec[:3] = hip_pos

    for j_name, idx in JOINT_INDEX.items():
        order_axes = [ch[0].upper() for ch in SKELETON[j_name]["channels"]
                      if "rotation" in ch.lower()]
        if len(order_axes) != 3:
            quat = np.array([1., 0., 0., 0.], dtype=np.float32)
        else:
            order = "".join(order_axes)
            angles = [euler_map[j_name][ax] for ax in order_axes]
            quat = euler_to_quat(angles, order)
        vec[3 + idx * 4: 3 + idx * 4 + 4] = quat
    return vec

def bvh_to_quat_array(bvh_path: str, w_trans: float) -> np.ndarray:
    raw_frames = read_bvh.parse_frames(bvh_path)
    # --- LIST comprehension (not generator) to satisfy older NumPy versions --- #
    return np.vstack([frame_to_vec(f, w_trans) for f in raw_frames])

# --------------------------------------------------------------------------- #
# Decoding helpers                                                            #
# --------------------------------------------------------------------------- #
def vec_to_frame(vec: np.ndarray) -> List[float]:
    hip_unscaled = vec[:3] / WEIGHT_TRANSLATION
    out: List[float] = []

    for j_name, j_spec in SKELETON.items():
        ch_names = j_spec["channels"]
        if not ch_names:
            continue

        if j_name in JOINT_INDEX:
            idx = JOINT_INDEX[j_name]
            q = vec[3 + idx * 4: 3 + idx * 4 + 4]
            R = tfs_quat.quat2mat(q)
        else:
            R = np.eye(3)

        order_axes = [ch[0].upper() for ch in ch_names if "rotation" in ch.lower()]
        if len(order_axes) != 3:
            order_axes = ["Z", "Y", "X"]
        order = "".join(order_axes)
        eulers = mat_to_euler_intrinsic(R, order)

        for ch in ch_names:
            if "position" in ch.lower():
                if j_name == "hip":
                    if ch.lower().startswith("x"): out.append(hip_unscaled[0])
                    elif ch.lower().startswith("y"): out.append(hip_unscaled[1])
                    else: out.append(hip_unscaled[2])
                else:
                    out.append(0.0)
            else:
                axis = ch[0].upper()
                out.append(eulers[axis])
    return out

def quat_array_to_frames(arr: np.ndarray) -> np.ndarray:
    return np.vstack([vec_to_frame(v) for v in arr])

# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def encode_folder(src: str, dst: str, w_trans: float, verbose: bool) -> List[Tuple[str, np.ndarray]]:
    os.makedirs(dst, exist_ok=True)
    cache: List[Tuple[str, np.ndarray]] = []

    for fname in sorted(listdir(src)):
        if not fname.lower().endswith(".bvh"): continue
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname + ".npy")

        quat = bvh_to_quat_array(src_path, w_trans)
        np.save(dst_path, quat)
        cache.append((fname, quat))

        if verbose:
            hip_y = quat[:, 1]
            norms = np.linalg.norm(quat[:, 3:].reshape(-1, 4), axis=1)
            print(f"[ENCODE] {fname:<25} frames={len(quat):4d} "
                  f"hip_y={hip_y.min():.2f}/{hip_y.max():.2f}  "
                  f"quat‖={norms.mean():.3f}")
    return cache

def decode_folder(src: str, dst: str, verbose: bool) -> None:
    os.makedirs(dst, exist_ok=True)
    for fname in sorted(listdir(src)):
        if not fname.lower().endswith(".npy"): continue
        npy = os.path.join(src, fname)
        dst_bvh = os.path.join(dst, fname + ".bvh")

        arr = np.load(npy)
        frames = quat_array_to_frames(arr)
        read_bvh.write_frames(STANDARD_BVH, dst_bvh, frames)
        if verbose:
            print(f"[DECODE] {fname:<25} frames={len(frames):4d}")

# --------------------------------------------------------------------------- #
# Diagnostics                                                                 #
# --------------------------------------------------------------------------- #
def frame_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = np.abs(a - b)
    return d.mean(), d.max()

def round_trip_check(src_bvh: str, quat: np.ndarray, verbose: bool) -> None:
    if not verbose: return
    ref_frames = read_bvh.parse_frames(src_bvh)
    if ref_frames.size == 0: return
    rec_frame = vec_to_frame(quat[0])
    mean, mx = frame_diff(ref_frames[0], np.array(rec_frame))
    print(f"     ↳ first-frame diff  mean={mean:.3e}  max={mx:.3e}")

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src_bvh_dir", required=True)
    ap.add_argument("--out_quad_dir", required=True)
    ap.add_argument("--recon_bvh_dir")
    ap.add_argument("--weight_translation", type=float,
                    default=WEIGHT_TRANSLATION) 
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        print("• Joint order (index → name):")
        for n, i in JOINT_INDEX.items(): print(f"{i:4}: {n}")
        print("-" * 60)

    cache = encode_folder(args.src_bvh_dir, args.out_quad_dir,
                          args.weight_translation, args.verbose)

    for fname, q in cache:
        round_trip_check(os.path.join(args.src_bvh_dir, fname), q, args.verbose)

    if args.recon_bvh_dir:
        decode_folder(args.out_quad_dir, args.recon_bvh_dir, args.verbose)


if __name__ == "__main__":
    main()
