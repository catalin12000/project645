#!/usr/bin/env python3
"""
pytorch_train_quad_aclstm.py  –  quaternion Auto-Conditioned-LSTM training
(with live loss curves)

Run
---
python code/pytorch_train_quad_aclstm.py \
    --dances_folder          train_data_quad/martial/ \
    --write_weight_folder    checkpoints_quad/ \
    --write_bvh_motion_folder sampled_bvh_quad/ \
    --batch_size 32 \
    --seq_len 100 \
    --dance_frame_rate 60 \
    --total_iterations 20000 \
    --plot_every 100        # iterations between PNG updates (optional)
"""

# ────────────────────────────────────────────────────────────────────────────
import os, random, argparse, importlib.util, numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import read_bvh

# ---------- constants -------------------------------------------------------
JOINTS_NUM   = len(read_bvh.joint_index)          # 57
FRAME_SIZE   = 3 + JOINTS_NUM * 4                 # 231
HIP_IDX      = read_bvh.joint_index['hip']        # 0
COND_N, GT_N = 5, 5                               # teacher-forcing pattern

# ---------- utilities -------------------------------------------------------
def renorm_quat(t: torch.Tensor) -> torch.Tensor:
    """normalise each quaternion in a [B,231] tensor"""
    trs, rest = t[:, :3], t[:, 3:].reshape(t.size(0), -1, 4)
    rest = rest / (rest.norm(dim=2, keepdim=True) + 1e-8)
    return torch.cat([trs, rest.reshape(t.size(0), -1)], 1)

def load_npy(folder: str):
    return [np.load(os.path.join(folder, f))
            for f in os.listdir(folder) if f.endswith(".npy")]

# ---------- AC-LSTM network --------------------------------------------------
class acLSTM(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.hid = hidden
        self.l1 = nn.LSTMCell(FRAME_SIZE, hidden)
        self.l2 = nn.LSTMCell(hidden, hidden)
        self.l3 = nn.LSTMCell(hidden, hidden)
        self.dec = nn.Linear(hidden, FRAME_SIZE)

    # helper: teacher-forcing mask
    @staticmethod
    def _mask(T, gt, cond, dev):
        pat = np.tile(np.concatenate([np.ones(gt), np.zeros(cond)]), 100)[:T]
        return torch.tensor(pat, device=dev)

    def _init_hc(self, B, dev):
        z = lambda: torch.zeros(B, self.hid, device=dev)
        return [z(), z(), z()], [z(), z(), z()]

    def forward(self, seq):                         # seq (B,T,231)
        B, T, _ = seq.shape; dev = seq.device
        mask = self._mask(T, GT_N, COND_N, dev)
        h, c  = self._init_hc(B, dev)
        y     = torch.zeros(B, FRAME_SIZE, device=dev)
        outs  = []
        for t in range(T):
            x = seq[:, t] if mask[t] else y         # teacher forcing
            h[0], c[0] = self.l1(x, (h[0], c[0]))
            h[1], c[1] = self.l2(h[0], (h[1], c[1]))
            h[2], c[2] = self.l3(h[1], (h[2], c[2]))
            y          = renorm_quat(self.dec(h[2]))
            outs.append(y)
        return torch.stack(outs, 1)                 # (B,T,231)

    # -------- three-way loss ------------------------------------------------
    @staticmethod
    def split_loss(pred, tgt):
        """returns (total, root, rot) MSE losses"""
        root = nn.functional.mse_loss(pred[:, :, :3],  tgt[:, :, :3])
        rot  = nn.functional.mse_loss(pred[:, :, 3:],  tgt[:, :, 3:])
        return root + rot, root, rot

# ---------- plotting --------------------------------------------------------
PLOT_DIR = "training_plots_quad"
def update_plots(hist: np.ndarray):
    os.makedirs(PLOT_DIR, exist_ok=True)
    names = ["total", "root", "rot"]
    for i, n in enumerate(names):
        plt.clf(); plt.plot(hist[:, i]); plt.grid(True)
        plt.xlabel("iteration"); plt.ylabel(f"{n} MSE"); plt.title(n + " loss")
        plt.savefig(os.path.join(PLOT_DIR, f"loss_{n}.png")); plt.close()

# ---------- training iteration ---------------------------------------------
def train_step(batch_np, net, opt):
    dev = torch.device("cuda")
    batch = torch.tensor(batch_np, dtype=torch.float32, device=dev)  # (B,S+1,231)
    inp, tgt = batch[:, :-1], batch[:, 1:]                           # (B,S,231)
    pred = net(inp)
    total, root, rot = net.split_loss(pred, tgt)
    opt.zero_grad(); total.backward(); opt.step()
    return total.item(), root.item(), rot.item(), pred, tgt

# ---------- main training loop ---------------------------------------------
def train(dances, fps, args):
    torch.cuda.set_device(0)
    dev  = torch.device("cuda")
    net  = acLSTM(args.hidden_size).to(dev)
    if args.read_weight_path and os.path.exists(args.read_weight_path):
        net.load_state_dict(torch.load(args.read_weight_path, map_location=dev))
    opt  = torch.optim.Adam(net.parameters(), lr=1e-4)

    hist = []                                    # loss history list
    speed = fps / 30.0                           # frame-skip factor

    for it in range(args.total_iterations):
        # ---- build one minibatch ----------------------------------------
        batch = []
        while len(batch) < args.batch_size:
            clip = random.choice(dances)
            if len(clip) < args.seq_len + 2: continue
            st = random.randint(10, int(len(clip) - speed * (args.seq_len + 2)) - 10)
            seq = [clip[int(st + i * speed)] for i in range(args.seq_len + 2)]
            batch.append(seq)
        batch = np.asarray(batch, np.float32)    # (B,S+2,231)

        # ---- optimisation step -----------------------------------------
        tot, root, rot, *_ = train_step(batch, net, opt)
        hist.append([tot, root, rot])

        # ---- console & plots -------------------------------------------
        if it % 20 == 0:
            print(f"[{it:07d}] total {tot:.6f} | root {root:.6f} | rot {rot:.6f}")
        if it % args.plot_every == 0:
            update_plots(np.asarray(hist))
            np.save(os.path.join(PLOT_DIR, "loss_history.npy"), np.asarray(hist))
        if it % 1000 == 0:
            os.makedirs(args.write_weight_folder, exist_ok=True)
            torch.save(net.state_dict(),
                       os.path.join(args.write_weight_folder, f"{it:07d}.weight"))

# ---------- CLI -------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description="Quaternion ac-LSTM trainer with loss plots")
    p.add_argument('--dances_folder',          required=True)
    p.add_argument('--write_weight_folder',    required=True)
    p.add_argument('--write_bvh_motion_folder', required=True)   # (kept for compatibility)
    p.add_argument('--read_weight_path',       default="")
    p.add_argument('--hidden_size',            type=int, default=1024)
    p.add_argument('--batch_size',             type=int, default=32)
    p.add_argument('--seq_len',                type=int, default=100)
    p.add_argument('--dance_frame_rate',       type=int, default=60)
    p.add_argument('--total_iterations',       type=int, default=100000)
    p.add_argument('--plot_every',             type=int, default=100,
                   help="iterations between PNG refreshes")
    return p.parse_args()

# ---------- entry-point -----------------------------------------------------
def main():
    args = parse()
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)       # still created
    dances = load_npy(args.dances_folder)
    train(dances, args.dance_frame_rate, args)

if __name__ == '__main__':
    main()
