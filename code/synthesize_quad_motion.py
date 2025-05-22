#!/usr/bin/env python3
"""
synthesize_quad_motion.py  –  drift-controlled quaternion motion synthesis
-------------------------------------------------------------------------

• Adds *per-frame* MSE evaluation for the first 20 generated steps,
  identical in style to the Euler synthesizer.

Typical run
-----------
python code/synthesize_quad_motion.py \
    --dances_folder train_data_quad/martial/ \
    --read_weight_path checkpoints_quad/0019000.weight \
    --write_bvh_folder synth_out_quad/ \
    --frame_rate 60 --batch 11 --seed_frames 20 --generate_frames 400 \
    --root_step_clamp 2 --zero_drift --verbose --debug_drift
"""
# --------------------------------------------------------------------- #
import os, random, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import transforms3d.quaternions as tfs_quat, transforms3d.euler as tfs_eul
import read_bvh, read_bvh_hierarchy

WEIGHT_TR = read_bvh.weight_translation
STD_BVH   = read_bvh.standard_bvh_file
SKEL, _   = read_bvh_hierarchy.read_bvh_hierarchy(STD_BVH)
J_IDX     = read_bvh.joint_index
FRAME_SZ  = 3 + len(J_IDX) * 4

# -------------------- quat-vector → BVH frame ------------------------- #
def quat2eul(q, order):
    return np.degrees(
        tfs_eul.mat2euler(tfs_quat.quat2mat(q), "r" + order.lower())
    )

def vec_to_frame(v):
    hip = v[:3] / WEIGHT_TR
    out = []
    for j, spec in SKEL.items():
        ch = spec["channels"]
        if not ch:                       # end-site
            continue
        if j in J_IDX:
            q     = v[3 + J_IDX[j] * 4 : 3 + J_IDX[j] * 4 + 4]
            order = "".join(ax[0].upper() for ax in ch if "rotation" in ax)
            e_map = dict(zip(order, quat2eul(q, order)))
        for name in ch:
            if "position" in name.lower():
                out.append(hip["xyz".index(name[0].lower())] if j == "hip" else 0.0)
            else:
                out.append(e_map[name[0].upper()] if j in J_IDX else 0.0)
    return out

def arr_to_frames(arr):                  # (T,231) → (T,raw)
    return np.stack([vec_to_frame(v) for v in arr])

# -------------------- quaternion re-norm ------------------------------ #
def renorm(t):
    tr, rest = t[:, :3], t[:, 3:].reshape(t.size(0), -1, 4)
    rest     = rest / (rest.norm(2, 2, keepdim=True) + 1e-8)
    return torch.cat([tr, rest.reshape(t.size(0), -1)], 1)

# -------------------- AC-LSTM model ----------------------------------- #
class acLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.h = hidden
        self.l1 = nn.LSTMCell(FRAME_SZ, hidden)
        self.l2 = nn.LSTMCell(hidden,  hidden)
        self.l3 = nn.LSTMCell(hidden,  hidden)
        self.dec = nn.Linear(hidden, FRAME_SZ)

    def _zeros(self, b, d): return torch.zeros(b, self.h, device=d)
    def _init_hc(self, b, d): return ([self._zeros(b,d)]*3, [self._zeros(b,d)]*3)

    def _step(self, x, h, c):
        h0,c0 = self.l1(x,(h[0],c[0]))
        h1,c1 = self.l2(h0,(h[1],c[1]))
        h2,c2 = self.l3(h1,(h[2],c[2]))
        y     = renorm(self.dec(h2))
        return y, [h0,h1,h2], [c0,c1,c2]

    @torch.no_grad()
    def generate(self, seed_abs, gen_frames):
        B,S,_ = seed_abs.shape; dev = seed_abs.device
        h,c   = self._init_hc(B, dev)
        out   = []
        y     = None
        for t in range(S):            # teacher-forcing
            y,h,c = self._step(seed_abs[:,t], h, c)
            out.append(y)
        for _ in range(gen_frames):   # autoregressive
            y,h,c = self._step(y, h, c)
            out.append(y)
        return torch.stack(out, 1).cpu().numpy()   # (B,S+gen,231)

# -------------------- drift utilities ---------------------------------- #
def clamp_steps(seq, max_step):
    if max_step <= 0: return seq
    hip = seq[:, :3] / WEIGHT_TR
    for i in range(1, len(seq)):
        dx = hip[i,0]-hip[i-1,0]; dz = hip[i,2]-hip[i-1,2]
        if abs(dx) > max_step: hip[i,0] = hip[i-1,0] + np.sign(dx)*max_step
        if abs(dz) > max_step: hip[i,2] = hip[i-1,2] + np.sign(dz)*max_step
    seq[:, :3] = hip * WEIGHT_TR
    return seq

def remove_linear_drift(seq):
    hip = seq[:, :3] / WEIGHT_TR
    n   = len(seq); t = np.arange(n)
    for k in (0,2):                       # X and Z
        m = (hip[-1,k]-hip[0,k])/(n-1)
        hip[:,k] -= m * t
    seq[:, :3] = hip * WEIGHT_TR
    return seq

# -------------------- basic plotting ----------------------------------- #
def plot_hip(seed, pred, out_png):
    s, p = seed/WEIGHT_TR, pred/WEIGHT_TR
    fig,(a,b) = plt.subplots(1,2,figsize=(9,4))
    a.plot(s[:,1]); a.plot(p[:,1],'--'); a.set_title("Hip Y")
    b.plot(s[:,0], s[:,2]); b.plot(p[:,0], p[:,2],'--')
    b.set_aspect('equal'); b.set_title("Hip X-Z")
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

# -------------------- I/O helpers -------------------------------------- #
def load_npy(folder):
    return [np.load(os.path.join(folder,f))
            for f in os.listdir(folder) if f.endswith(".npy")]

def save_bvhs(batch, out_dir):
    for i, motion in enumerate(batch):
        frames = arr_to_frames(motion)
        read_bvh.write_frames(
            STD_BVH, os.path.join(out_dir, f"out{i:02d}.bvh"), frames
        )

# -------------------- main synthesis ----------------------------------- #
def synthesize(motions, model, args):
    sp           = args.frame_rate / 30.0
    seeds        = []
    sample_info  = []                      # (clip, seed_start) for eval

    for _ in range(args.batch):
        d  = random.choice(motions)
        st = random.randint(0, int(len(d) - sp*args.seed_frames - 1))
        seeds.append(np.asarray([d[int(st+i*sp)] for i in range(args.seed_frames)],
                                np.float32))
        sample_info.append((d, st))

    seed = np.stack(seeds)                 # (B,S,231)

    if args.verbose:
        hip_y = seed[:,:,1] / WEIGHT_TR
        print(f"> Seed hip-Y {hip_y.min():.2f} … {hip_y.max():.2f} m")

    pred = model.generate(torch.from_numpy(seed).cuda(), args.generate_frames)

    # ------------- drift stats BEFORE correction ----------------------- #
    if args.verbose:
        hip = pred[:,:,:3] / WEIGHT_TR
        print(f"> Raw drift  X μ {hip[:,:,0].mean():.2f} max {abs(hip[:,:,0]).max():.2f} | "
              f"Z μ {hip[:,:,2].mean():.2f} max {abs(hip[:,:,2]).max():.2f}")

    # ------------- per-frame clamp + linear zero drift ----------------- #
    for b in range(pred.shape[0]):
        pred[b] = clamp_steps(pred[b], args.root_step_clamp)
        if args.zero_drift:
            pred[b] = remove_linear_drift(pred[b])

    # ------------- drift stats AFTER correction ------------------------ #
    if args.verbose:
        hip = pred[:,:,:3] / WEIGHT_TR
        print(f"> Corrected drift  X μ {hip[:,:,0].mean():.2f} max {abs(hip[:,:,0]).max():.2f} | "
              f"Z μ {hip[:,:,2].mean():.2f} max {abs(hip[:,:,2]).max():.2f}")

    # ------------- DEBUG table (sample-0) ------------------------------ #
    if args.debug_drift:
        hip = pred[0,:,:3] / WEIGHT_TR
        dx  = np.diff(hip[:,0]); dz = np.diff(hip[:,2])
        print("\nframe |   hipX     hipZ | ΔX    ΔZ")
        for i in range(min(11,len(hip))):
            if i == 0:
                print(f"{i:5d} | {hip[i,0]:7.2f} {hip[i,2]:7.2f} |  ---   ---")
            else:
                print(f"{i:5d} | {hip[i,0]:7.2f} {hip[i,2]:7.2f} | {dx[i-1]:6.2f} {dz[i-1]:6.2f}")
        print()

    # ------------- NEW: Quantitative MSE on first 20 steps ------------- #
    eval_len = 20
    for b, (dance, st) in enumerate(sample_info):
        # collect ground-truth continuation
        gt_frames = []
        for i in range(eval_len):
            idx = int(st + (args.seed_frames + i) * sp)
            if idx >= len(dance): break
            gt_frames.append(dance[idx])
        if not gt_frames:
            continue
        gt_eval   = np.asarray(gt_frames, np.float32)
        pred_eval = pred[b, :len(gt_eval)]
        mse_clip  = np.mean((pred_eval - gt_eval) ** 2)

        print(f"[EVAL] sample-{b:02d}  MSE first {len(gt_eval)} steps: "
              f"{mse_clip:.6f}")

        # per-frame error table
        frame_mse = np.mean((pred_eval - gt_eval) ** 2, axis=1)
        print("        step |  frame-MSE")
        print("        -----+-----------")
        for t, err in enumerate(frame_mse):
            print(f"        {t:5d} | {err:.6f}")
        print()

    # ------------- Save BVHs & plot ------------------------------------ #
    os.makedirs(args.write_bvh_folder, exist_ok=True)
    save_bvhs(pred, args.write_bvh_folder)
    if not args.no_plots:
        plot_hip(seed[0], pred[0],
                 os.path.join(args.write_bvh_folder, "compare.png"))

# -------------------- CLI ---------------------------------------------- #
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dances_folder", required=True)
    pa.add_argument("--read_weight_path", required=True)
    pa.add_argument("--write_bvh_folder", required=True)
    pa.add_argument("--frame_rate", type=int, default=60)
    pa.add_argument("--batch", type=int, default=5)
    pa.add_argument("--seed_frames", type=int, default=15)
    pa.add_argument("--generate_frames", type=int, default=400)
    pa.add_argument("--hidden_size", type=int, default=1024)
    pa.add_argument("--root_step_clamp", type=float, default=0.5)
    pa.add_argument("--zero_drift", action="store_true")
    pa.add_argument("--no_plots",  action="store_true")
    pa.add_argument("--debug_drift", action="store_true")
    pa.add_argument("-v", "--verbose", action="store_true")
    args = pa.parse_args()

    motions = load_npy(args.dances_folder)

    torch.cuda.set_device(0)
    model = acLSTM(args.hidden_size).cuda()
    model.load_state_dict(torch.load(args.read_weight_path, map_location="cuda"))
    model.eval()

    synthesize(motions, model, args)

# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
