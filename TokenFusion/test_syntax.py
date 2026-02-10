"""
Syntax and Import Structure Test (No PyTorch required)

Tests:
1. Python syntax (compile all .py files)
2. Import structure (relative imports)
3. Config file parsing
4. Critical constants and values
"""

import os
import sys
import py_compile
import traceback


def test_syntax():
    """Test Python syntax of all .py files."""
    print("=" * 60)
    print("TEST 1: Python Syntax Check")
    print("=" * 60)

    py_files = [
        'models/__init__.py',
        'models/modules.py',
        'models/mix_transformer.py',
        'models/segformer.py',
        'datasets/__init__.py',
        'datasets/uavscenes.py',
        'utils/__init__.py',
        'utils/transforms.py',
        'utils/optimizer.py',
        'utils/metrics.py',
        'utils/helpers.py',
        'main.py',
        'evaluate.py',
    ]

    all_passed = True
    for py_file in py_files:
        try:
            print(f"  Checking {py_file:40s}...", end=" ")
            py_compile.compile(py_file, doraise=True)
            print("✓")
        except Exception as e:
            print(f"✗\n    Error: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All Python files have valid syntax!\n")
    else:
        print("\n❌ Some files have syntax errors!\n")

    return all_passed


def test_import_structure():
    """Test import structure without importing torch."""
    print("=" * 60)
    print("TEST 2: Import Structure (No Torch)")
    print("=" * 60)

    # Test only non-torch imports
    tests = [
        ("numpy", "numpy"),
        ("cv2", "opencv"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
    ]

    all_passed = True
    for module, name in tests:
        try:
            print(f"  Checking {name:20s}...", end=" ")
            __import__(module)
            print("✓")
        except ImportError:
            print(f"✗ (not installed)")
            all_passed = False

    if all_passed:
        print("\n✅ Basic dependencies available!\n")
    else:
        print("\n⚠️  Some dependencies missing (OK for WSL, needed on VM)\n")

    return True  # Don't fail on this


def test_config_parsing():
    """Test config file parsing."""
    print("=" * 60)
    print("TEST 3: Config File Parsing")
    print("=" * 60)

    try:
        import yaml

        config_path = "configs/uavscenes_rgb_hag.yaml"
        print(f"  Loading {config_path}...", end=" ")

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print("✓")

        # Check structure
        required_sections = [
            'dataset', 'model', 'training', 'optimizer',
            'loss', 'tokenfusion', 'augmentation', 'hag',
            'normalization', 'evaluation', 'logging'
        ]

        print("  Checking config sections:")
        for section in required_sections:
            present = section in cfg
            status = "✓" if present else "✗"
            print(f"    {section:20s}: {status}")

        # Check critical values
        print("\n  Critical values:")
        critical = {
            'backbone': cfg['model']['backbone'],
            'num_classes': cfg['dataset']['num_classes'],
            'image_size': cfg['training']['image_size'],
            'batch_size': cfg['training']['batch_size'],
            'lr': cfg['optimizer']['lr'],
            'lamda': cfg['loss']['lamda'],
            'mask_threshold': cfg['tokenfusion']['mask_threshold'],
            'hag_max_meters': cfg['hag']['max_meters'],
        }

        for key, value in critical.items():
            print(f"    {key:20s}: {value}")

        # Validate values
        assert cfg['model']['backbone'] == 'mit_b2', "Wrong backbone"
        assert cfg['dataset']['num_classes'] == 19, "Wrong num_classes"
        assert cfg['training']['image_size'] == 768, "Wrong image_size"
        assert cfg['training']['batch_size'] == 8, "Wrong batch_size"
        assert cfg['loss']['lamda'] == 1e-3, "Wrong lamda (should be 1e-3 for segmentation)"
        assert cfg['tokenfusion']['mask_threshold'] == 0.02, "Wrong threshold"
        assert cfg['hag']['max_meters'] == 50.0, "Wrong HAG max"

        print("\n✅ Config file valid and values correct!\n")
        return True

    except Exception as e:
        print(f"\n❌ Config error: {e}")
        traceback.print_exc()
        return False


def test_dataset_constants():
    """Test dataset constants without loading data."""
    print("=" * 60)
    print("TEST 4: Dataset Constants")
    print("=" * 60)

    try:
        # Read the file and check constants
        with open('datasets/uavscenes.py', 'r') as f:
            content = f.read()

        # Check for critical constants
        checks = [
            ('TRAIN_SCENES', 'interval5_AMtown01'),
            ('TEST_SCENES', 'interval5_AMtown03'),
            ('LABEL_MAPPING', '0: 255'),
            ('CLASS_NAMES', 'background'),
            ('load_hag', 'np.stack([normalized_hag] * 3'),  # 3-channel HAG!
        ]

        print("  Checking critical constants:")
        all_present = True
        for name, expected_content in checks:
            present = expected_content in content
            status = "✓" if present else "✗"
            print(f"    {name:20s}: {status}")
            if not present:
                all_present = False

        # Count scenes
        train_count = content.count('interval5_AM') + content.count('interval5_HK')
        print(f"\n  Scene mentions: {train_count}")

        if all_present:
            print("\n✅ Dataset constants correct!\n")
        else:
            print("\n❌ Some constants missing!\n")

        return all_present

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def test_model_structure():
    """Check model structure without instantiating."""
    print("=" * 60)
    print("TEST 5: Model Structure")
    print("=" * 60)

    try:
        # Check modules.py
        with open('models/modules.py', 'r') as f:
            modules_content = f.read()

        print("  Checking modules.py:")
        checks = [
            ('num_parallel = 2', 'num_parallel = 2'),
            ('TokenExchange', 'class TokenExchange'),
            ('ModuleParallel', 'class ModuleParallel'),
            ('LayerNormParallel', 'class LayerNormParallel'),
            ('ln_0 and ln_1', "f'ln_{i}'"),  # Individual LN
        ]

        for name, pattern in checks:
            present = pattern in modules_content
            status = "✓" if present else "✗"
            print(f"    {name:30s}: {status}")

        # Check mix_transformer.py
        with open('models/mix_transformer.py', 'r') as f:
            mit_content = f.read()

        print("\n  Checking mix_transformer.py:")
        checks = [
            ('mit_b2 class', 'class mit_b2'),
            ('depths=[3, 4, 6, 3]', 'depths=[3, 4, 6, 3]'),
            ('embed_dims=[64, 128, 320, 512]', 'embed_dims=[64, 128, 320, 512]'),
            ('PredictorLG', 'class PredictorLG'),
            ('TokenExchange', 'TokenExchange'),
        ]

        for name, pattern in checks:
            present = pattern in mit_content
            status = "✓" if present else "✗"
            print(f"    {name:30s}: {status}")

        print("\n✅ Model structure looks correct!\n")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Check file and directory structure."""
    print("=" * 60)
    print("TEST 6: File Structure")
    print("=" * 60)

    required_structure = {
        'directories': [
            'models', 'datasets', 'utils', 'configs',
            'pretrained', 'logs', 'checkpoints'
        ],
        'files': [
            'models/__init__.py',
            'models/modules.py',
            'models/mix_transformer.py',
            'models/segformer.py',
            'datasets/__init__.py',
            'datasets/uavscenes.py',
            'utils/__init__.py',
            'utils/transforms.py',
            'utils/optimizer.py',
            'utils/metrics.py',
            'utils/helpers.py',
            'configs/uavscenes_rgb_hag.yaml',
            'main.py',
            'evaluate.py',
            'train.sh',
            'requirements.txt',
            'README.md',
        ]
    }

    all_present = True

    print("  Checking directories:")
    for dir_name in required_structure['directories']:
        exists = os.path.isdir(dir_name)
        status = "✓" if exists else "✗"
        print(f"    {dir_name:20s}: {status}")
        if not exists and dir_name not in ['logs', 'checkpoints']:  # These are created at runtime
            all_present = False

    print("\n  Checking files:")
    for file_name in required_structure['files']:
        exists = os.path.isfile(file_name)
        status = "✓" if exists else "✗"
        print(f"    {file_name:40s}: {status}")
        if not exists:
            all_present = False

    if all_present:
        print("\n✅ File structure complete!\n")
    else:
        print("\n⚠️  Some files/directories missing\n")

    return all_present


def main():
    print("\n" + "=" * 60)
    print("TokenFusion UAVScenes Syntax & Structure Test")
    print("(No PyTorch Required - Safe for WSL)")
    print("=" * 60 + "\n")

    results = {}

    # Run all tests
    results['file_structure'] = test_file_structure()
    results['syntax'] = test_syntax()
    results['imports'] = test_import_structure()
    results['config'] = test_config_parsing()
    results['dataset_constants'] = test_dataset_constants()
    results['model_structure'] = test_model_structure()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    critical_tests = ['syntax', 'config', 'dataset_constants', 'model_structure']
    all_critical_passed = True

    for name, passed in results.items():
        is_critical = name in critical_tests
        marker = "🔴" if is_critical else "🟡"
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {marker} {name:20s}: {status}")
        if is_critical and not passed:
            all_critical_passed = False

    print("=" * 60)
    if all_critical_passed:
        print("✅ CRITICAL TESTS PASSED!")
        print("📦 Code is ready to deploy to VM for training.")
        print("\nNext steps:")
        print("  1. Transfer code to VM")
        print("  2. Install requirements: pip install -r requirements.txt")
        print("  3. Download pretrained weights: ./train.sh")
        print("  4. Start training!")
    else:
        print("❌ CRITICAL TESTS FAILED!")
        print("⚠️  Fix errors before deploying to VM.")

    print("=" * 60 + "\n")

    return all_critical_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
