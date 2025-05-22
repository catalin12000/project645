import os
import torch
import numpy as np
import argparse
from read_bvh import joint_index, weight_translation, write_frames, standard_bvh_file
import read_bvh_hierarchy
# ython code/synthesize_euler_motion.py --read_weight_path checkpoints_euler/0019000.weight --initial_seq_folder train_data_euler/martial/ --write_bvh_motion_folder synth_out_euler/ --initial_seq_len 30 --generate_frames_number 400 --batch_size 10
# Constants
Hip_index = joint_index['hip']
In_frame_size_default = len(joint_index) * 3
Hidden_size_default = 1024
POS_SCALING = weight_translation


class acLSTM(torch.nn.Module):
    def __init__(self, in_frame_size, hidden_size, out_frame_size):
        super().__init__()
        self.lstm1 = torch.nn.LSTMCell(in_frame_size, hidden_size)
        self.lstm2 = torch.nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = torch.nn.LSTMCell(hidden_size, hidden_size)
        self.dec = torch.nn.Linear(hidden_size, out_frame_size)

    def init_hidden(self, batch_size, device):
        return ([torch.zeros(batch_size, self.dec.in_features, device=device) for _ in range(3)],
                [torch.zeros(batch_size, self.dec.in_features, device=device) for _ in range(3)])

    def forward_lstm(self, x, h, c):
        h[0], c[0] = self.lstm1(x, (h[0], c[0]))
        h[1], c[1] = self.lstm2(h[0], (h[1], c[1]))
        h[2], c[2] = self.lstm3(h[1], (h[2], c[2]))
        return self.dec(h[2]), h, c

    def forward(self, seed_seq, gen_len):
        batch_size, seed_len, _ = seed_seq.shape
        device = seed_seq.device
        h, c = self.init_hidden(batch_size, device)
        outputs = []

        current = torch.zeros(batch_size, self.dec.out_features, device=device)

        for t in range(seed_len):
            current, h, c = self.forward_lstm(seed_seq[:, t], h, c)
            outputs.append(current.unsqueeze(1))

        for _ in range(gen_len):
            current, h, c = self.forward_lstm(current, h, c)
            outputs.append(current.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def write_bvh_from_euler(euler_seq, filepath):
    skeleton, _ = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)
    reconstructed = []

    for frame in euler_seq:
        frame_out = []
        hip = frame[Hip_index * 3: Hip_index * 3 + 3] / POS_SCALING
        joint_rot = {
            name: frame[i * 3: i * 3 + 3].tolist()
            for name, i in joint_index.items() if name != 'hip'
        }
        joint_rot['hip'] = [0.0, 0.0, 0.0]

        for joint in skeleton:
            angles = joint_rot.get(joint, [0.0, 0.0, 0.0])
            for ch in skeleton[joint]['channels']:
                if 'position' in ch:
                    frame_out.append(hip["XYZ".index(ch[0])] if joint == 'hip' else 0.0)
                elif 'rotation' in ch:
                    frame_out.append(angles["XYZ".index(ch[0])])
        reconstructed.append(frame_out)

    write_frames(standard_bvh_file, filepath, np.array(reconstructed))


def synthesize_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = acLSTM(args.in_frame_size, args.hidden_size, args.out_frame_size).to(device)
    model.load_state_dict(torch.load(args.read_weight_path, map_location=device))
    model.eval()

    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    files = [f for f in os.listdir(args.initial_seq_folder) if f.endswith(".npy")]
    if not files:
        raise RuntimeError(f"No .npy files found in {args.initial_seq_folder}")

    files = files[:args.batch_size]

    for i, fname in enumerate(files):
        seed_path = os.path.join(args.initial_seq_folder, fname)
        data = np.load(seed_path)
        if data.shape[0] < args.initial_seq_len + 20:
            print(f"[SKIP] {fname} is too short ({data.shape[0]} frames)")
            continue

        seed = data[:args.initial_seq_len]
        seed_tensor = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            full_seq = model(seed_tensor, args.generate_frames_number).squeeze(0).cpu().numpy()

        # === Quantitative Evaluation: Compare first 20 generated steps with real motion ===
        eval_len = 20
        gt_eval = data[args.initial_seq_len : args.initial_seq_len + eval_len]
        pred_eval = full_seq[:eval_len]

        mse = np.mean((pred_eval - gt_eval) ** 2)
        print(f"[EVAL] {fname} Quantitative MSE (first {eval_len} steps): {mse:.6f}")

        # === Save .bvh output for qualitative evaluation ===
        out_path = os.path.join(args.write_bvh_motion_folder, f"{os.path.splitext(fname)[0]}_gen.bvh")
        write_bvh_from_euler(full_seq, out_path)
        print(f"[✓] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_weight_path', type=str, required=True, help='Path to trained model .weight file')
    parser.add_argument('--initial_seq_folder', type=str, required=True, help='Folder with seed .npy files')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Where to save .bvh results')

    parser.add_argument('--dance_frame_rate', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--initial_seq_len', type=int, default=30)
    parser.add_argument('--generate_frames_number', type=int, default=400)

    parser.add_argument('--in_frame_size', type=int, default=In_frame_size_default)
    parser.add_argument('--out_frame_size', type=int, default=In_frame_size_default)
    parser.add_argument('--hidden_size', type=int, default=Hidden_size_default)

    args = parser.parse_args()
    synthesize_batch(args)
