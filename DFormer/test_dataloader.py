"""
Test script for UAVScenes dataset loading.
Verifies the data loading pipeline works correctly before training.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.dataloader.UAVScenesDataset import UAVScenesDataset, CLASS_NAMES

def test_dataset():
    """Test UAVScenes dataset loading."""

    print("=" * 60)
    print("Testing UAVScenes Dataset Loading")
    print("=" * 60)

    # Create dataset setting
    data_setting = {
        "dataset_path": "/home/bariskaya/Projelerim/UAV/UAVScenesData",
        "transform_gt": False,
        "class_names": CLASS_NAMES,
        "dataset_name": "UAVScenes",
        "backbone": "DFormerv2_B",
        "hag_max_meters": 150.0,
    }

    # Test train dataset without preprocessing
    print("\n[1] Testing train dataset (no preprocessing)...")
    try:
        train_dataset = UAVScenesDataset(data_setting, "train", preprocess=None)
        print(f"    Train samples: {len(train_dataset)}")

        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"    Sample keys: {list(sample.keys())}")
            print(f"    RGB shape: {sample['data'].shape}")
            print(f"    Label shape: {sample['label'].shape}")
            print(f"    HAG (modal_x) shape: {sample['modal_x'].shape}")
            print(f"    Filename: {sample['fn'][:80]}...")

            # Check data ranges
            rgb = sample['data'].numpy()
            label = sample['label'].numpy()
            hag = sample['modal_x'].numpy()

            print(f"\n    RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")
            print(f"    Label unique values: {np.unique(label)[:10]}...")
            print(f"    HAG range: [{hag.min():.2f}, {hag.max():.2f}]")
        else:
            print("    ERROR: No training samples found!")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test val dataset
    print("\n[2] Testing val dataset (no preprocessing)...")
    try:
        val_dataset = UAVScenesDataset(data_setting, "val", preprocess=None)
        print(f"    Val samples: {len(val_dataset)}")

        if len(val_dataset) > 0:
            sample = val_dataset[0]
            print(f"    Sample keys: {list(sample.keys())}")
            print(f"    RGB shape: {sample['data'].shape}")
        else:
            print("    ERROR: No validation samples found!")
            return False
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with preprocessing
    print("\n[3] Testing with preprocessing (TrainPre)...")
    try:
        from utils.dataloader.dataloader import TrainPre
        from easydict import EasyDict as edict

        # Create minimal config for preprocessing
        config = edict()
        config.image_height = 768
        config.image_width = 768
        config.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

        norm_mean = np.array([0.485, 0.456, 0.406])
        norm_std = np.array([0.229, 0.224, 0.225])

        train_pre = TrainPre(norm_mean, norm_std, sign=True, config=config)

        train_dataset_aug = UAVScenesDataset(data_setting, "train", preprocess=train_pre)
        sample = train_dataset_aug[0]

        print(f"    RGB shape after aug: {sample['data'].shape}")
        print(f"    Label shape after aug: {sample['label'].shape}")
        print(f"    HAG shape after aug: {sample['modal_x'].shape}")

        # Check normalized data ranges
        rgb = sample['data'].numpy()
        hag = sample['modal_x'].numpy()

        print(f"\n    RGB range (normalized): [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"    HAG range (normalized): [{hag.min():.2f}, {hag.max():.2f}]")

        # Verify shapes are correct for training
        assert sample['data'].shape == (3, 768, 768), f"RGB shape mismatch: {sample['data'].shape}"
        assert sample['label'].shape == (768, 768), f"Label shape mismatch: {sample['label'].shape}"
        assert sample['modal_x'].shape == (3, 768, 768), f"HAG shape mismatch: {sample['modal_x'].shape}"

        print("\n    ✓ All shapes correct!")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test DataLoader
    print("\n[4] Testing DataLoader...")
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(train_dataset_aug, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))

        print(f"    Batch RGB shape: {batch['data'].shape}")
        print(f"    Batch Label shape: {batch['label'].shape}")
        print(f"    Batch HAG shape: {batch['modal_x'].shape}")

        assert batch['data'].shape == (2, 3, 768, 768), "Batch RGB shape mismatch"
        assert batch['label'].shape == (2, 768, 768), "Batch Label shape mismatch"
        assert batch['modal_x'].shape == (2, 3, 768, 768), "Batch HAG shape mismatch"

        print("\n    ✓ DataLoader working correctly!")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed! Data loading pipeline is working.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
