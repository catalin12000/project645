# MAI645 Final Project – Character Motion Prediction

This project explores character motion prediction using auto-conditioned LSTM (acLSTM) models trained on 3D motion capture data. We evaluate three motion representations: **Positional**, **Euler**, and **Quaternion**, and study their impact on synthesis quality and prediction stability.

## Project Structure

```
project645/
├── code/
│   ├── pytorch_train_pos_aclstm.py         # Train positional model
│   ├── pytorch_train_euler_aclstm.py       # Train Euler model
│   ├── pytorch_train_quad_aclstm.py        # Train quaternion model
│   ├── synthesize_euler_motion.py          # Synthesize Euler motions with MSE
│   ├── synthesize_quad_motion.py           # Synthesize quaternion motions (drift-corrected)
│   ├── read_bvh.py / read_bvh_hierarchy.py # BVH I/O and hierarchy utils
│   ├── rotation2xyz.py                     # Rotation to XYZ utilities
│   └── generate_training_quad_data.py      # Quaternion preprocessing tools
│
├── train_data_pos/     # Positional data (.npy)
├── train_data_euler/   # Euler data
├── train_data_quad/    # Quaternion data
├── checkpoints_*/      # Saved model checkpoints
├── synth_out_*/        # Synthesized motions for each representation
├── sampled_bvh_*/      # Sampled outputs during training
├── training_plots/     # Loss visualization
└── MAI645_Final_Report.docx
```

## 🛠️ Environment Setup

Install dependencies:
```bash
pip install torch numpy matplotlib transforms3d
```

##  Training Scripts

###  Positional
```bash
python code/pytorch_train_pos_aclstm.py --dances_folder train_data_pos/martial/ \
  --write_weight_folder checkpoints_pos/ --write_bvh_motion_folder sampled_bvh_pos/ \
  --in_frame 171 --out_frame 171 --batch_size 32 --seq_len 100 --total_iterations 200000
```

###  Euler
```bash
python code/pytorch_train_euler_aclstm.py --dances_folder train_data_euler/martial/ \
  --write_weight_folder checkpoints_euler/ --write_bvh_motion_folder sampled_bvh_euler/ \
  --in_frame 171 --out_frame 171 --batch_size 32 --seq_len 100 --total_iterations 200000
```

###  Quaternion
```bash
python code/pytorch_train_quad_aclstm.py --dances_folder train_data_quad/martial/ \
  --write_weight_folder checkpoints_quad/ --write_bvh_motion_folder sampled_bvh_quad/ \
  --in_frame 231 --out_frame 231 --batch_size 32 --seq_len 100 --total_iterations 200000
```

Each script saves model weights and visualizations.

##  Evaluation Scripts

### Euler Synthesis
```bash
python code/synthesize_euler_motion.py \
  --read_weight_path checkpoints_euler/0019000.weight \
  --initial_seq_folder train_data_euler/martial/ \
  --write_bvh_motion_folder synth_out_euler/ \
  --initial_seq_len 30 --generate_frames_number 400 --batch_size 10
```

### Quaternion Synthesis (Drift Correction)
```bash
python code/synthesize_quad_motion.py \
  --dances_folder train_data_quad/martial/ \
  --read_weight_path checkpoints_quad/0019000.weight \
  --write_bvh_folder synth_out_quad/ \
  --frame_rate 60 --batch 11 --seed_frames 20 --generate_frames 400 \
  --root_step_clamp 2 --zero_drift --verbose
```



##  Loss Functions

| Representation | Loss Used | Notes |
|----------------|-----------|-------|
| Positional     | MSE       | On XYZ positions |
| Euler          | MSE       | On rotation angles |
| Quaternion     | MSE       | On [root + quats] vector |






