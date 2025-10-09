"""
Deception Circuits Research Infrastructure

A comprehensive framework for discovering and manipulating deception circuits 
in LLM reasoning traces using linear probes and sparse autoencoders.

Based on the research project: "Can deception circuits be discovered and 
causally manipulated inside LLM reasoning traces?"

This framework implements the methodology from the BLKY project, which aims to:
1. Discover internal circuits that drive deceptive behavior in LLMs
2. Test whether these circuits can be causally manipulated
3. Determine if deception circuits generalize across different contexts

Key Components:
- CSV data loading for truthful vs deceptive pairs
- Linear probe training and evaluation
- Sparse autoencoder feature discovery
- Activation patching for causal testing
- Cross-context generalization analysis

Usage Example:
    from deception_circuits import DeceptionTrainingPipeline
    
    pipeline = DeceptionTrainingPipeline(device="cpu")
    results = pipeline.run_full_experiment("deception_data.csv")

=== FOR FIRST-TIME READERS ===

Welcome to the Deception Circuits Research Framework! This is a complete toolkit
for understanding how AI models lie and finding ways to detect/control deception.

=== WHAT IS THIS FRAMEWORK? ===

This framework helps you answer a crucial question: "How do AI models lie, and can we detect it?"

When AI models are deceptive, they must encode that deception somewhere in their internal
neural computations. This framework helps you find those "deception circuits" - the specific
patterns of neural activity that correspond to lying vs telling the truth.

=== THE RESEARCH PROCESS ===

1. **COLLECT DATA**: Get examples of truthful vs deceptive responses
2. **TRAIN DETECTORS**: Create AI models that can spot deception signals
3. **FIND CIRCUITS**: Identify which parts of the neural network are responsible
4. **TEST CAUSALITY**: Prove that manipulating those circuits changes behavior
5. **CHECK GENERALIZATION**: See if circuits work across different types of lies

=== QUICK START ===

```python
# Import the main pipeline
from deception_circuits import DeceptionTrainingPipeline

# Create and run a complete experiment
pipeline = DeceptionTrainingPipeline(device="cpu")
results = pipeline.run_full_experiment("your_data.csv")

# See which layers are best at detecting deception
print(f"Best layer: {results['analysis_results']['best_probe_layer']}")
```

=== WHAT YOU CAN DO ===

- **Discover Deception Circuits**: Find where lies are encoded in neural networks
- **Test Causal Interventions**: Prove that circuits actually cause deception
- **Cross-Scenario Analysis**: See if circuits generalize across different types of lies
- **Create Visualizations**: Generate plots and dashboards of your results
- **Save Everything**: Complete experiment logging and model checkpoints

=== WHO IS THIS FOR? ===

- AI Safety Researchers studying model behavior
- Mechanistic Interpretability researchers
- Anyone interested in understanding AI deception
- Researchers working on AI alignment and transparency

This framework puts the power of cutting-edge AI interpretability research in your hands!
"""

__version__ = "0.1.0"
__author__ = "Krish Jain"

# Import all main classes for easy access
from .data_loader import DeceptionDataLoader
from .linear_probe import DeceptionLinearProbe, LinearProbeTrainer
from .sparse_autoencoder import DeceptionSparseAutoencoder, AutoencoderTrainer
from .activation_patching import ActivationPatcher, CausalTester
from .analysis import CircuitAnalyzer, VisualizationTools
from .training_pipeline import DeceptionTrainingPipeline

# Define what gets imported when someone does "from deception_circuits import *"
__all__ = [
    "DeceptionDataLoader",      # Loads CSV data with activation support
    "DeceptionLinearProbe",     # Linear classifier for deception detection
    "LinearProbeTrainer",       # Trains probes across multiple layers
    "DeceptionSparseAutoencoder", # Discovers interpretable features
    "AutoencoderTrainer",       # Trains autoencoders with sparsity
    "ActivationPatcher",        # Core patching operations
    "CausalTester",             # Comprehensive causal testing framework
    "CircuitAnalyzer",          # Analyzes results and identifies circuits
    "VisualizationTools",       # Creates plots and dashboards
    "DeceptionTrainingPipeline" # Main orchestration class
]
