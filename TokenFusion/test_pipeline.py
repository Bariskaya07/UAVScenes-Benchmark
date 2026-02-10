"""
Pipeline Test Script for TokenFusion UAVScenes

Tests:
1. Module imports
2. Model instantiation
3. Forward pass with dummy data
4. Loss computation
5. Dataset loading (if data exists)
6. Transform pipeline

Run: python test_pipeline.py
"""

import sys
import traceback

def test_imports():
    """Test all module imports."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    try:
        print("  Importing torch...", end=" ")
        import torch
        print(f"✓ (version: {torch.__version__})")

        print("  Importing timm...", end=" ")
        import timm
        print(f"✓ (version: {timm.__version__})")

        print("  Importing numpy...", end=" ")
        import numpy as np
        print(f"✓ (version: {np.__version__})")

        print("  Importing cv2...", end=" ")
        import cv2
        print(f"✓ (version: {cv2.__version__})")

        print("  Importing yaml...", end=" ")
        import yaml
        print("✓")

        print("  Importing models.modules...", end=" ")
        from models.modules import ModuleParallel, LayerNormParallel, TokenExchange, num_parallel
        print(f"✓ (num_parallel={num_parallel})")

        print("  Importing models.mix_transformer...", end=" ")
        from models.mix_transformer import mit_b0, mit_b1, mit_b2, MixVisionTransformer
        print("✓")

        print("  Importing models.segformer...", end=" ")
        from models.segformer import WeTr, SegFormerHead
        print("✓")

        print("  Importing datasets.uavscenes...", end=" ")
        from datasets.uavscenes import UAVScenesDataset, load_hag, remap_labels, TRAIN_SCENES, TEST_SCENES
        print(f"✓ (train_scenes={len(TRAIN_SCENES)}, test_scenes={len(TEST_SCENES)})")

        print("  Importing utils.transforms...", end=" ")
        from utils.transforms import TrainTransform, ValTransform
        print("✓")

        print("  Importing utils.optimizer...", end=" ")
        from utils.optimizer import PolyWarmupAdamW
        print("✓")

        print("  Importing utils.metrics...", end=" ")
        from utils.metrics import ConfusionMatrix, compute_metrics
        print("✓")

        print("  Importing utils.helpers...", end=" ")
        from utils.helpers import AverageMeter, sliding_window_inference
        print("✓")

        print("\n✅ All imports successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_model_instantiation():
    """Test model creation."""
    print("=" * 60)
    print("TEST 2: Model Instantiation")
    print("=" * 60)

    try:
        import torch
        from models.segformer import WeTr

        # Test mit_b2 (our target backbone)
        print("  Creating WeTr with mit_b2 backbone...", end=" ")
        model = WeTr(
            backbone='mit_b2',
            num_classes=19,
            embedding_dim=256,
            pretrained=None  # No pretrained weights for test
        )
        print("✓")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

        # Check model structure
        print(f"  Encoder embed_dims: {model.encoder.embed_dims}")
        print(f"  Decoder num_classes: {model.decoder.num_classes}")
        print(f"  Num parallel: {model.num_parallel}")

        print("\n✅ Model instantiation successful!\n")
        return True, model

    except Exception as e:
        print(f"\n❌ Model instantiation error: {e}")
        traceback.print_exc()
        return False, None


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("TEST 3: Forward Pass (Dummy Data)")
    print("=" * 60)

    try:
        import torch

        # Create dummy inputs (batch=2, 3 channels, 768x768)
        print("  Creating dummy inputs (B=2, C=3, H=768, W=768)...", end=" ")
        rgb = torch.randn(2, 3, 768, 768)
        hag = torch.randn(2, 3, 768, 768)
        print("✓")

        # Forward pass
        print("  Running forward pass...", end=" ")
        model.eval()
        with torch.no_grad():
            outputs, masks = model([rgb, hag])
        print("✓")

        # Check outputs
        print(f"  Output 0 (RGB) shape: {outputs[0].shape}")
        print(f"  Output 1 (HAG) shape: {outputs[1].shape}")
        print(f"  Output 2 (Ensemble) shape: {outputs[2].shape}")
        print(f"  Number of mask layers: {len(masks)}")
        print(f"  First mask shapes: [{masks[0][0].shape}, {masks[0][1].shape}]")

        # Verify output shapes
        expected_h = 768 // 4  # 1/4 resolution from decoder
        expected_w = 768 // 4
        assert outputs[0].shape == (2, 19, expected_h, expected_w), f"Unexpected output shape"
        print(f"  Output resolution: 1/4 of input (192x192) ✓")

        print("\n✅ Forward pass successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Forward pass error: {e}")
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print("=" * 60)
    print("TEST 4: Loss Computation")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F

        # Create dummy outputs and targets
        print("  Creating dummy outputs and targets...", end=" ")
        outputs = [
            torch.randn(2, 19, 192, 192),  # RGB pred
            torch.randn(2, 19, 192, 192),  # HAG pred
            torch.randn(2, 19, 192, 192),  # Ensemble
        ]
        target = torch.randint(0, 19, (2, 768, 768))
        masks = [[torch.rand(2, 192*192), torch.rand(2, 192*192)] for _ in range(16)]
        print("✓")

        # Compute loss
        print("  Computing loss...", end=" ")
        total_loss = 0
        lamda = 1e-3

        for output in outputs[:2]:
            output_up = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)
            soft_output = F.log_softmax(output_up, dim=1)
            loss = F.nll_loss(soft_output, target, ignore_index=255)
            total_loss += loss

        # L1 sparsity
        l1_loss = sum([torch.abs(m).sum() for mask in masks for m in mask])
        total_loss += lamda * l1_loss
        print("✓")

        print(f"  Segmentation loss: {(total_loss - lamda * l1_loss).item():.4f}")
        print(f"  L1 sparsity loss: {l1_loss.item():.4f}")
        print(f"  Total loss: {total_loss.item():.4f}")

        print("\n✅ Loss computation successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Loss computation error: {e}")
        traceback.print_exc()
        return False


def test_transforms():
    """Test data transforms."""
    print("=" * 60)
    print("TEST 5: Data Transforms")
    print("=" * 60)

    try:
        import numpy as np
        from utils.transforms import TrainTransform, ValTransform

        # Create dummy sample
        print("  Creating dummy sample...", end=" ")
        sample = {
            'rgb': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
            'hag': np.random.rand(1080, 1920, 3).astype(np.float32),
            'label': np.random.randint(0, 19, (1080, 1920), dtype=np.uint8),
            'inputs': ['rgb', 'hag']
        }
        print(f"✓ (RGB: {sample['rgb'].shape}, HAG: {sample['hag'].shape})")

        # Test train transform
        print("  Testing TrainTransform (768x768 crop)...", end=" ")
        train_tf = TrainTransform(crop_size=768)
        train_sample = train_tf(sample.copy())
        print("✓")
        print(f"    RGB: {train_sample['rgb'].shape} (expected: torch.Size([3, 768, 768]))")
        print(f"    HAG: {train_sample['hag'].shape} (expected: torch.Size([3, 768, 768]))")
        print(f"    Label: {train_sample['label'].shape} (expected: torch.Size([768, 768]))")

        # Verify shapes
        assert train_sample['rgb'].shape == (3, 768, 768), "RGB shape mismatch"
        assert train_sample['hag'].shape == (3, 768, 768), "HAG shape mismatch"
        assert train_sample['label'].shape == (768, 768), "Label shape mismatch"

        # Test val transform
        print("  Testing ValTransform...", end=" ")
        val_tf = ValTransform()
        val_sample = val_tf(sample.copy())
        print("✓")
        print(f"    RGB: {val_sample['rgb'].shape}")
        print(f"    HAG: {val_sample['hag'].shape}")

        print("\n✅ Transform tests successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Transform error: {e}")
        traceback.print_exc()
        return False


def test_optimizer():
    """Test optimizer creation."""
    print("=" * 60)
    print("TEST 6: Optimizer")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from utils.optimizer import PolyWarmupAdamW

        # Create dummy model
        print("  Creating dummy model...", end=" ")
        model = nn.Linear(256, 19)
        print("✓")

        # Create optimizer
        print("  Creating PolyWarmupAdamW...", end=" ")
        optimizer = PolyWarmupAdamW(
            model.parameters(),
            lr=6e-5,
            weight_decay=0.01,
            warmup_iter=1500,
            max_iter=40000
        )
        print("✓")

        # Test step
        print("  Testing optimizer step...", end=" ")
        x = torch.randn(2, 256)
        y = torch.randint(0, 19, (2,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✓")

        print(f"  Initial LR: {optimizer.get_lr():.6f}")

        print("\n✅ Optimizer test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Optimizer error: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("=" * 60)
    print("TEST 7: Metrics")
    print("=" * 60)

    try:
        import numpy as np
        from utils.metrics import ConfusionMatrix, UAVSCENES_CLASSES

        # Create confusion matrix
        print("  Creating ConfusionMatrix (19 classes)...", end=" ")
        cm = ConfusionMatrix(num_classes=19, ignore_label=255)
        print("✓")

        # Add dummy predictions
        print("  Adding dummy predictions...", end=" ")
        pred = np.random.randint(0, 19, (100, 100))
        target = np.random.randint(0, 19, (100, 100))
        target[0:10, 0:10] = 255  # Add some ignore labels
        cm.update(pred, target)
        print("✓")

        # Compute metrics
        print("  Computing metrics...", end=" ")
        metrics = cm.get_metrics()
        print("✓")

        print(f"  mIoU: {metrics['miou'] * 100:.2f}%")
        print(f"  Static mIoU: {metrics['static_miou'] * 100:.2f}%")
        print(f"  Dynamic mIoU: {metrics['dynamic_miou'] * 100:.2f}%")
        print(f"  Pixel Acc: {metrics['pixel_acc'] * 100:.2f}%")

        print("\n✅ Metrics test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Metrics error: {e}")
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading (if data exists)."""
    print("=" * 60)
    print("TEST 8: Dataset Loading (Optional)")
    print("=" * 60)

    import os
    data_root = "/home/bariskaya/Projelerim/UAV/UAVScenesData"

    if not os.path.exists(data_root):
        print(f"  ⚠️  Data root not found: {data_root}")
        print("  Skipping dataset test (will work on VM)")
        print("\n⏭️  Dataset test skipped\n")
        return True

    try:
        from datasets.uavscenes import UAVScenesDataset
        from utils.transforms import TrainTransform

        print(f"  Data root: {data_root}")

        # Create dataset
        print("  Creating UAVScenesDataset (train)...", end=" ")
        transform = TrainTransform(crop_size=768)
        dataset = UAVScenesDataset(
            data_root=data_root,
            split='train',
            transform=transform,
            hag_max_height=50.0
        )
        print(f"✓ ({len(dataset)} samples)")

        # Load one sample
        print("  Loading sample 0...", end=" ")
        sample = dataset[0]
        print("✓")
        print(f"    RGB: {sample['rgb'].shape}")
        print(f"    HAG: {sample['hag'].shape}")
        print(f"    Label: {sample['label'].shape}")

        print("\n✅ Dataset test successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Dataset error: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test config file loading."""
    print("=" * 60)
    print("TEST 9: Config Loading")
    print("=" * 60)

    try:
        import yaml

        config_path = "configs/uavscenes_rgb_hag.yaml"
        print(f"  Loading config: {config_path}...", end=" ")

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print("✓")

        # Check critical values
        print("  Checking critical values:")
        print(f"    backbone: {cfg['model']['backbone']}")
        print(f"    num_classes: {cfg['dataset']['num_classes']}")
        print(f"    image_size: {cfg['training']['image_size']}")
        print(f"    batch_size: {cfg['training']['batch_size']}")
        print(f"    lr: {cfg['optimizer']['lr']}")
        print(f"    lamda: {cfg['loss']['lamda']}")
        print(f"    mask_threshold: {cfg['tokenfusion']['mask_threshold']}")
        print(f"    hag_max_meters: {cfg['hag']['max_meters']}")

        # Verify values
        assert cfg['model']['backbone'] == 'mit_b2', "Wrong backbone"
        assert cfg['dataset']['num_classes'] == 19, "Wrong num_classes"
        assert cfg['training']['image_size'] == 768, "Wrong image_size"
        assert cfg['loss']['lamda'] == 1e-3, "Wrong lamda"

        print("\n✅ Config loading successful!\n")
        return True

    except Exception as e:
        print(f"\n❌ Config error: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("TokenFusion UAVScenes Pipeline Test")
    print("=" * 60 + "\n")

    results = {}

    # Run all tests
    results['imports'] = test_imports()

    if results['imports']:
        results['config'] = test_config_loading()
        success, model = test_model_instantiation()
        results['model'] = success

        if results['model'] and model is not None:
            results['forward'] = test_forward_pass(model)

        results['loss'] = test_loss_computation()
        results['transforms'] = test_transforms()
        results['optimizer'] = test_optimizer()
        results['metrics'] = test_metrics()
        results['dataset'] = test_dataset()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix before training.")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = main()
    sys.exit(0 if success else 1)
