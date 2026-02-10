"""
DFormerv2-Base Configuration for UAVScenes Dataset

Model: DFormerv2-Base (53.9M parameters)
Dataset: UAVScenes RGB + HAG
Training Resolution: 768x768 (for fair comparison with CMNeXt)

IMPORTANT NOTES:
1. Uses DFormerv2 paper's augmentation settings (NOT CMNeXt's!)
2. Scale range: 0.5-1.75 (paper default, NOT 0.5-2.0)
3. NO photometric distortion, NO gaussian blur (faithful to paper)
4. HAG normalization: 50m max (same as CMNeXt for fair comparison)

Benchmark Target:
- CMNeXt UAVScenes 768x768: ~76.94% mIoU
- DFormerv2-Base should achieve comparable or better performance
"""

from .._base_.datasets.UAVScenes import *

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
C.backbone = "DFormerv2_B"  # DFormerv2-Base backbone
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Base_pretrained.pth"
C.decoder = "ham"  # Hybrid Attention Module decoder
C.decoder_embed_dim = 512
C.aux_in_chans = 1  # Input channels for aux modality (1=HAG, 2=DELIVER LiDAR)
C.aux_channels = 1  # Dataset aux channels (1=HAG native)

# =============================================================================
# OPTIMIZER CONFIGURATION (Standardized for fair comparison)
# =============================================================================
C.optimizer = "AdamW"
C.lr = 6e-5           # Standard LR for fair comparison
C.lr_power = 0.9      # PolyLR power (CMNeXt paper setting)
C.momentum = 0.9
C.weight_decay = 0.01
C.warmup_ratio = 0.1  # Initial LR = LR * warmup_ratio (CMNeXt paper setting)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
C.batch_size = 8  # Same as CMNeXt for fair comparison (A100 40GB)
C.nepochs = 60   # Training epochs
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8  # Data loading workers

# Augmentation (Standardized with CMNeXt for fair comparison)
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # Scale range 0.5-2.0
C.use_photometric = True   # Photometric distortion (brightness, contrast, saturation, hue)
C.use_gaussian_blur = True  # Gaussian blur (p=0.5, kernel=5)

# Warmup
C.warm_up_epoch = 3   # Warmup epochs

# Model settings
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.2
C.aux_rate = 0.0  # No auxiliary loss

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
C.eval_iter = 5  # Evaluate every 5 epochs
C.eval_stride_rate = 2 / 3  # ~33% overlap for sliding window
C.eval_scale_array = [1.0]  # Single-scale evaluation for speed
C.eval_flip = True  # Flip augmentation during eval
C.eval_crop_size = [768, 768]  # Sliding window size (height, width)

# =============================================================================
# CHECKPOINT CONFIGURATION
# =============================================================================
C.checkpoint_start_epoch = 50  # Start saving checkpoints after epoch 50
C.checkpoint_step = 10  # Save every 10 epochs

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
C.save_dir = "output/UAVScenes_DFormerv2_B"
