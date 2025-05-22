#!/usr/bin/env python3
"""
pytorch_train_euler_aclstm.py  (compact version with live loss plot)

Run
---
python code/pytorch_train_euler_aclstm.py \
    --dances_folder train_data_euler/martial/ \
    --write_weight_folder checkpoints_euler/ \
    --write_bvh_motion_folder sampled_bvh_euler/ \
    --seq_len 100 --batch_size 32 --total_iterations 100000
"""

#python code/pytorch_train_euler_aclstm.py --dances_folder train_data_euler/martial/ --write_weight_folder checkpoints_euler/ --write_bvh_motion_folder sampled_bvh_euler/ --seq_len 100 --batch_size 32 --total_iterations 20000

# ────────────────────────────────────────────────────────────────────────────
import os, random, argparse, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt                # NEW
from read_bvh import joint_index, weight_translation, write_frames, standard_bvh_file
import read_bvh_hierarchy

# --- constants -------------------------------------------------------------
HIP_IDX      = joint_index['hip']
JOINTS_NUM   = len(joint_index)
IN_FRAME_SZ  = JOINTS_NUM * 3          # 174 floats per frame

# --- model -----------------------------------------------------------------
class acLSTM(nn.Module):
    def __init__(self, in_frame_size=IN_FRAME_SZ, hidden_size=1024):
        super().__init__()
        self.lstm1 = nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self.dec   = nn.Linear(hidden_size, in_frame_size)

    def _init_hc(self, B, device):
        z = lambda: torch.zeros(B, self.dec.in_features, device=device)
        return [z(), z(), z()], [z(), z(), z()]

    @staticmethod
    def _cond_mask(T, gt_n, cond_n):
        mask = np.concatenate([np.ones(gt_n), np.zeros(cond_n)])
        return torch.tensor(np.tile(mask, int(np.ceil(T/len(mask))))[:T])

    def forward(self, inp, gt_n, cond_n):
        B, T, _ = inp.shape; dev = inp.device
        h,c = self._init_hc(B, dev)
        m   = self._cond_mask(T, gt_n, cond_n).to(dev)
        y   = torch.zeros(B, IN_FRAME_SZ, device=dev)
        outs=[]
        for t in range(T):
            x = inp[:,t] if m[t] else y
            h[0],c[0] = self.lstm1(x,(h[0],c[0]))
            h[1],c[1] = self.lstm2(h[0],(h[1],c[1]))
            h[2],c[2] = self.lstm3(h[1],(h[2],c[2]))
            y         = self.dec(h[2])
            outs.append(y.unsqueeze(1))
        return torch.cat(outs,1)

    def calculate_loss(self, pred, gt):
        hip_s = HIP_IDX*3; hip_e = hip_s+3
        hip_loss = nn.functional.mse_loss(pred[:,:,hip_s:hip_e], gt[:,:,hip_s:hip_e])

        mask = torch.ones_like(pred); mask[:,:,hip_s:hip_e]=0
        angle_diff = (pred-gt) * (np.pi/180)
        rot_loss = torch.sum((1-torch.cos(angle_diff))*mask)/(pred.numel()+1e-8)
        return hip_loss + rot_loss

# --- utilities -------------------------------------------------------------
def load_euler(folder):
    return [np.load(os.path.join(folder,f))
            for f in os.listdir(folder) if f.endswith('.npy')]

def apply_hip_delta(seq):
    seq = np.asarray(seq, np.float32).copy()             # NEW ensure np.array
    delta = seq[1:]-seq[:-1]
    seq[1:, HIP_IDX*3]     = delta[:,HIP_IDX*3]          # hip X
    seq[1:, HIP_IDX*3+2]   = delta[:,HIP_IDX*3+2]        # hip Z
    return seq

def write_bvh(fname, arr):
    skel,_=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
    recon=[]
    for f in arr:
        out=[]; hip=f[HIP_IDX*3:HIP_IDX*3+3]/weight_translation
        for j in skel:
            rot=f[joint_index[j]*3:joint_index[j]*3+3] if j!='hip' else [0,0,0]
            for ch in skel[j]['channels']:
                if 'position' in ch:
                    out.append(hip["XYZ".index(ch[0])] if j=='hip' else 0.0)
                else: out.append(rot["XYZ".index(ch[0])])
        recon.append(out)
    write_frames(standard_bvh_file,fname,np.asarray(recon))

# --- plotting --------------------------------------------------------------
def update_loss_plot(loss_hist, angle_diff_hist, out_dir="training_plots_euler"):
    os.makedirs(out_dir, exist_ok=True)

    # Plot loss curve
    plt.clf()
    plt.plot(loss_hist)
    plt.xlabel("iteration")
    plt.ylabel("loss (MSE)")
    plt.title("Euler-LSTM training loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # Plot angle diff curve
    if angle_diff_hist:
        angle_diff_array = np.array(angle_diff_hist)  # (iters, seq_len)
        final_curve = angle_diff_array[-1]            # last snapshot
        plt.clf()
        plt.plot(final_curve)
        plt.xlabel("frame")
        plt.ylabel("mean |angle diff| (degrees)")
        plt.title("Mean Angle Difference per Frame (Final Iteration)")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "angle_diff_curve.png"))
        plt.close()

# --- training --------------------------------------------------------------
def train(args):
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = load_euler(args.dances_folder); print("Loaded",len(clips),"clips")
    os.makedirs(args.write_weight_folder,      exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder,  exist_ok=True)

    net = acLSTM(hidden_size=args.hidden_size).to(dev)
    if args.read_weight_path and os.path.exists(args.read_weight_path):
        net.load_state_dict(torch.load(args.read_weight_path,map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4); net.train()

    loss_hist = []
    angle_diff_hist = []

    for it in range(args.total_iterations):
        # ----- minibatch ---------------------------------------------------
        batch=[]
        while len(batch)<args.batch_size:
            clip = random.choice(clips)
            if len(clip) < args.seq_len + 2: continue
            st = np.random.randint(0, len(clip) - args.seq_len - 1)
            batch.append(apply_hip_delta(clip[st:st + args.seq_len + 1]))
        batch = np.asarray(batch, np.float32)

        inp = torch.tensor(batch[:, :-1], device=dev)
        tgt = torch.tensor(batch[:, 1:] , device=dev)

        pred = net(inp, args.groundtruth_num, args.condition_num)
        loss = net.calculate_loss(pred, tgt)

        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(loss.item())

        # Angle diff tracking
        with torch.no_grad():
            angle_diff = (pred - tgt).abs().detach().cpu().numpy()
            mean_per_frame = np.mean(angle_diff, axis=(0, 2))  # shape: (seq_len,)
            angle_diff_hist.append(mean_per_frame)

        if it % 20 == 0:
            print(f"[{it:06d}] loss {loss.item():.6f}")

        if it % args.plot_every == 0:
            update_loss_plot(loss_hist, angle_diff_hist)

        if it % 1000 == 0:
            torch.save(net.state_dict(),
                       os.path.join(args.write_weight_folder, f"{it:07d}.weight"))
            gt_np = tgt[0].detach().cpu().numpy()
            pr_np = pred[0].detach().cpu().numpy()
            write_bvh(os.path.join(args.write_bvh_motion_folder, f"{it:07d}_gt.bvh"), gt_np)
            write_bvh(os.path.join(args.write_bvh_motion_folder, f"{it:07d}_out.bvh"), pr_np)

# --- CLI -------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dances_folder", required=True)
    p.add_argument("--write_weight_folder", required=True)
    p.add_argument("--write_bvh_motion_folder", required=True)
    p.add_argument("--read_weight_path", default="")
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_iterations", type=int, default=100000)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--groundtruth_num", type=int, default=5)
    p.add_argument("--condition_num", type=int, default=5)
    p.add_argument("--plot_every", type=int, default=100,
                   help="iterations between loss-curve updates")
    args=p.parse_args()

    train(args)
