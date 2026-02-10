"""
DFormerv2-Base Configuration for DELIVER Dataset

Model: DFormerv2-Base (53.9M parameters)
Dataset: DELIVER RGB + LiDAR
Training Resolution: 1024x1024
"""

from .._base_.datasets.DELIVER import *

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
C.backbone = "DFormerv2_B"
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Base_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.aux_in_chans = 3  # LiDAR: 1-ch stacked to 3-ch (same as CMNeXt original)
C.aux_channels = 3

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================
C.optimizer = "AdamW"
C.lr = 6e-5
C.lr_power = 1.0
C.momentum = 0.9
C.weight_decay = 0.01
C.warmup_ratio = 1e-6

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
C.batch_size = 4  # Reduced for 1024x1024
C.nepochs = 60
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8

# Augmentation
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
C.use_photometric = True
C.use_gaussian_blur = True
C.warm_up_epoch = 3

# Model settings
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.2
C.aux_rate = 0.0

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
C.eval_iter = 5
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1.0]
C.eval_flip = True
C.eval_crop_size = [1024, 1024]

# =============================================================================
# CHECKPOINT CONFIGURATION
# =============================================================================
C.checkpoint_start_epoch = 50
C.checkpoint_step = 10

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
C.save_dir = "output/DELIVER_DFormerv2_B"
