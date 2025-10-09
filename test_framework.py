"""
Simple test script to verify the deception circuits framework works correctly.
"""

import torch
import pandas as pd
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.append('.')

try:
    from deception_circuits import (
        DeceptionDataLoader,
        DeceptionTrainingPipeline,
        CircuitAnalyzer,
        VisualizationTools
    )
    print("[PASS] All modules imported successfully")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)


def test_data_loader():
    """Test the data loader functionality."""
    print("\nTesting Data Loader...")
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'statement': ['Test statement 1', 'Test statement 2'],
            'response': ['Response 1', 'Response 2'],
            'label': [0, 1],
            'scenario': ['test', 'test']
        })
        
        # Save to CSV
        test_csv = Path('test_data.csv')
        sample_data.to_csv(test_csv, index=False)
        
        # Test loading
        loader = DeceptionDataLoader(device='cpu')
        results = loader.load_csv(test_csv)
        
        assert len(results['dataframe']) == 2
        assert results['num_truthful'] == 1
        assert results['num_deceptive'] == 1
        
        print("[PASS] Data loader test passed")
        
        # Clean up
        test_csv.unlink()
        
    except Exception as e:
        print(f"[FAIL] Data loader test failed: {e}")
        return False
    
    return True


def test_linear_probe():
    """Test linear probe functionality."""
    print("\nTesting Linear Probe...")
    
    try:
        from deception_circuits.linear_probe import DeceptionLinearProbe, LinearProbeTrainer
        
        # Create sample data
        activations = torch.randn(100, 768)
        labels = torch.randint(0, 2, (100,)).float()
        
        # Test probe creation
        probe = DeceptionLinearProbe(input_dim=768)
        outputs = probe(activations)
        
        assert outputs.shape == (100,)
        assert torch.all((outputs >= 0) & (outputs <= 1))
        
        # Test training
        trainer = LinearProbeTrainer(device='cpu')
        results = trainer.train_probe(activations, labels, epochs=5)
        
        assert 'probe' in results
        assert 'final_metrics' in results
        
        print("[PASS] Linear probe test passed")
        
    except Exception as e:
        print(f"[FAIL] Linear probe test failed: {e}")
        return False
    
    return True


def test_sparse_autoencoder():
    """Test sparse autoencoder functionality."""
    print("\nTesting Sparse Autoencoder...")
    
    try:
        from deception_circuits.sparse_autoencoder import DeceptionSparseAutoencoder, AutoencoderTrainer
        
        # Create sample data
        activations = torch.randn(100, 768)
        
        # Test autoencoder creation
        autoencoder = DeceptionSparseAutoencoder(input_dim=768, bottleneck_dim=512)
        reconstructed, encoded, _ = autoencoder(activations)
        
        assert reconstructed.shape == activations.shape
        assert encoded.shape == (100, 512)
        
        # Test training
        trainer = AutoencoderTrainer(device='cpu')
        results = trainer.train_unsupervised_autoencoder(activations, epochs=5)
        
        assert 'autoencoder' in results
        assert 'final_metrics' in results
        
        print("[PASS] Sparse autoencoder test passed")
        
    except Exception as e:
        print(f"[FAIL] Sparse autoencoder test failed: {e}")
        return False
    
    return True


def test_activation_patching():
    """Test activation patching functionality."""
    print("\nTesting Activation Patching...")
    
    try:
        from deception_circuits.activation_patching import ActivationPatcher, CausalTester
        
        # Create sample data
        activations = torch.randn(10, 5, 768)  # 10 samples, 5 layers, 768 dims
        
        # Test patcher
        patcher = ActivationPatcher(device='cpu')
        
        # Test basic patching
        patched = patcher.patch_activations(activations, activations, layer_indices=2)
        assert torch.allclose(patched, activations)
        
        # Test steering
        steering_vector = torch.randn(768)
        steered = patcher.steering_intervention(activations, steering_vector, layer_indices=2)
        assert not torch.allclose(steered, activations)
        
        # Test causal tester
        tester = CausalTester(device='cpu')
        
        truthful_acts = activations[:5]
        deceptive_acts = activations[5:]
        truthful_labels = torch.zeros(5)
        deceptive_labels = torch.ones(5)
        
        results = tester.test_deception_suppression(
            truthful_acts, deceptive_acts, truthful_labels, deceptive_labels, layer_idx=2
        )
        
        assert isinstance(results, dict)
        
        print("[PASS] Activation patching test passed")
        
    except Exception as e:
        print(f"[FAIL] Activation patching test failed: {e}")
        return False
    
    return True


def test_analysis_tools():
    """Test analysis and visualization tools."""
    print("\nTesting Analysis Tools...")
    
    try:
        # Create mock results
        mock_results = {
            'probe_results': {
                'layer_results': {
                    'layer_0': {
                        'layer_idx': 0,
                        'final_auc': 0.8,
                        'final_accuracy': 0.75
                    },
                    'layer_1': {
                        'layer_idx': 1,
                        'final_auc': 0.9,
                        'final_accuracy': 0.85
                    }
                },
                'test_results': {
                    'layer_0': {
                        'auc': 0.8,
                        'accuracy': 0.75,
                        'precision': 0.7,
                        'recall': 0.8,
                        'f1': 0.75
                    },
                    'layer_1': {
                        'auc': 0.9,
                        'accuracy': 0.85,
                        'precision': 0.8,
                        'recall': 0.9,
                        'f1': 0.85
                    }
                }
            }
        }
        
        # Test analyzer
        analyzer = CircuitAnalyzer(mock_results)
        probe_analysis = analyzer.analyze_probe_performance()
        
        assert 'best_layer' in probe_analysis
        assert 'statistics' in probe_analysis
        
        # Test visualization tools
        viz_tools = VisualizationTools()
        
        # Test that we can create a figure (without actually plotting)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig = viz_tools.plot_layer_performance(mock_results)
        assert fig is not None
        
        print("[PASS] Analysis tools test passed")
        
    except Exception as e:
        print(f"[FAIL] Analysis tools test failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("DECEPTION CIRCUITS FRAMEWORK TESTS")
    print("=" * 50)
    
    tests = [
        test_data_loader,
        test_linear_probe,
        test_sparse_autoencoder,
        test_activation_patching,
        test_analysis_tools
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("[SUCCESS] All tests passed! The framework is working correctly.")
        return True
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)



