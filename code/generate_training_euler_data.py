#!/usr/bin/env python3
"""
generate_training_euler_data.py  –  root-translation + local Euler rotations

PATCH 2025-05-19
----------------
* hip **rotation is zeroed** in every frame so the network never learns to
  turn the pelvis.  (Absolute hip translation is still kept.)
* Everything else (diagnostics plots, CLI, etc.) unchanged.
"""
# ---------------------------------------------------------------- imports
import argparse, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import transforms3d.euler as tfs_eul
import read_bvh, read_bvh_hierarchy

# ---------------------------------------------------------------- skeleton
HIP_IDX     = read_bvh.joint_index['hip']
JOINT_INDEX = read_bvh.joint_index
STD_BVH     = read_bvh.standard_bvh_file
WEIGHT_TR   = read_bvh.weight_translation
SKEL,_      = read_bvh_hierarchy.read_bvh_hierarchy(STD_BVH)
FRAME_SIZE  = 3 + len(JOINT_INDEX)*3          # 174

# ---------------------------------------------------------------- helpers
def frame_to_vec(raw):
    """raw BVH frame → 174-D vector (hip pos + local Euler XYZ)"""
    vec  = np.zeros(FRAME_SIZE, np.float32)
    hip  = np.zeros(3, np.float32)
    euls = {j:{'X':0,'Y':0,'Z':0} for j in SKEL}

    off = 0
    for j,spec in SKEL.items():
        ch = spec['channels']; n=len(ch)
        vals = raw[off:off+n]; off += n
        for i,name in enumerate(ch):
            v = vals[i]
            if 'position' in name.lower():
                if   name.startswith('X'): hip[0]=v*WEIGHT_TR
                elif name.startswith('Y'): hip[1]=v*WEIGHT_TR
                else:                     hip[2]=v*WEIGHT_TR
            elif 'rotation' in name.lower():
                euls[j][name[0]] = v
    vec[:3] = hip
    for j,idx in JOINT_INDEX.items():
        start = 3 + idx*3
        if j == 'hip':                   # <-- NEW: zero hip rotation
            vec[start:start+3] = 0.0
            continue
        xyz = euls[j]
        vec[start:start+3] = [xyz['X'], xyz['Y'], xyz['Z']]
    return vec

def bvh_to_euler_array(bvh):
    raw = read_bvh.parse_frames(bvh)
    return np.vstack([frame_to_vec(f) for f in raw])

# ------------- diagnostics same as before (omitted for brevity) ---------- #

# ------------- CLI encode / decode (unchanged) --------------------------- #
def main():
    ap = argparse.ArgumentParser("Encode BVH → Euler .npy (hip rot = 0)")
    ap.add_argument('--src_bvh_dir', required=True)
    ap.add_argument('--out_euler_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_euler_dir, exist_ok=True)
    for f in sorted(os.listdir(args.src_bvh_dir)):
        if not f.endswith('.bvh'): continue
        arr = bvh_to_euler_array(os.path.join(args.src_bvh_dir, f))
        np.save(os.path.join(args.out_euler_dir, f+'.npy'), arr)
        print("encoded", f, arr.shape)
if __name__ == '__main__':
    main()
