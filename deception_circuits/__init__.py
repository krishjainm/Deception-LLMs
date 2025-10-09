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
