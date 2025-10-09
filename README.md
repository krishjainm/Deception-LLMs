# Deception Circuits Research Infrastructure

A comprehensive framework for discovering and manipulating deception circuits in LLM reasoning traces using linear probes and sparse autoencoders.

## Overview

This project implements the research methodology described in "Can deception circuits be discovered and causally manipulated inside LLM reasoning traces?" The framework provides tools for:

- **Data Loading**: CSV-based loading of truthful vs deceptive reasoning pairs
- **Linear Probes**: Training probes to detect deception signals in model activations
- **Sparse Autoencoders**: Discovering interpretable features related to deception
- **Causal Testing**: Activation patching and steering for causal intervention
- **Analysis & Visualization**: Comprehensive analysis and visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/krishjainm/Deception-LLMs.git
cd Deception-LLMs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from deception_circuits import DeceptionTrainingPipeline

# Initialize pipeline
pipeline = DeceptionTrainingPipeline(device="cpu", output_dir="results")

# Run experiment
results = pipeline.run_full_experiment(
    csv_path="your_data.csv",
    max_samples=1000,
    probe_config={'epochs': 100, 'lr': 0.01},
    autoencoder_config={'epochs': 100, 'lr': 0.001}
)
```

### CSV Data Format

Your CSV should contain the following columns:
- `statement`: The input prompt/statement
- `response`: The model's response
- `label`: 0 for truthful, 1 for deceptive
- `scenario`: Type of deception (poker, roleplay, sandbagging, etc.)

Example:
```csv
statement,response,label,scenario
"What is 2+2?","4",0,general
"What is 2+2?","I'm not sure",1,sandbagging
"Do you have a strong hand?","Yes, very strong",0,poker
"Do you have a strong hand?","Yes, very strong",1,poker
```

## Core Components

### 1. Data Loading (`data_loader.py`)

```python
from deception_circuits import DeceptionDataLoader

loader = DeceptionDataLoader(device="cpu")
data = loader.load_csv("data.csv", activation_dir="activations/")
train_data, test_data = loader.split_data(test_size=0.2)
```

### 2. Linear Probes (`linear_probe.py`)

```python
from deception_circuits import LinearProbeTrainer

trainer = LinearProbeTrainer(device="cpu")
results = trainer.train_probe(activations, labels, epochs=100)
```

### 3. Sparse Autoencoders (`sparse_autoencoder.py`)

```python
from deception_circuits import AutoencoderTrainer

trainer = AutoencoderTrainer(device="cpu")
results = trainer.train_unsupervised_autoencoder(
    activations, epochs=100, l1_coeff=0.01
)
```

### 4. Causal Testing (`activation_patching.py`)

```python
from deception_circuits import CausalTester

tester = CausalTester(device="cpu")
results = tester.test_deception_suppression(
    truthful_activations, deceptive_activations, 
    truthful_labels, deceptive_labels, layer_idx=20
)
```

### 5. Analysis & Visualization (`analysis.py`)

```python
from deception_circuits import CircuitAnalyzer, VisualizationTools

analyzer = CircuitAnalyzer(results)
probe_analysis = analyzer.analyze_probe_performance()

viz_tools = VisualizationTools()
fig = viz_tools.plot_layer_performance(results)
```

## Example Experiment

Run the complete example:

```bash
python example_usage.py
```

This will:
1. Create sample deception data
2. Train linear probes and sparse autoencoders
3. Analyze results across layers
4. Run causal intervention tests
5. Generate visualizations

## Research Applications

### Deception Scenarios

The framework supports multiple deception types:

- **Poker Bluffing**: Strategic deception in game contexts
- **Sandbagging**: Deliberate underperformance
- **Roleplay**: Character-based deception
- **Pressure/Blackmail**: Deception under threat
- **Password Gating**: Conditional knowledge hiding

### Cross-Context Generalization

Test whether deception circuits generalize across scenarios:

```python
# Train on poker scenarios, test on sandbagging
cross_results = pipeline._cross_scenario_analysis(
    df, train_activations, train_labels, test_activations, test_labels
)
```

### Causal Interventions

Test causal relationships using activation patching:

```python
# Test 1: Replace deceptive activations with truthful ones
suppression_results = tester.test_deception_suppression(...)

# Test 2: Inject deception into truthful runs
injection_results = tester.test_deception_injection(...)

# Test 3: Cross-context patching
cross_context_results = tester.test_cross_context_patching(...)
```

## Architecture

```
deception_circuits/
├── __init__.py              # Main package interface
├── data_loader.py           # CSV data loading and processing
├── linear_probe.py          # Linear probe implementation
├── sparse_autoencoder.py    # Sparse autoencoder implementation
├── training_pipeline.py     # Main training orchestration
├── analysis.py              # Analysis and visualization tools
└── activation_patching.py   # Causal testing framework
```

## Key Features

- **Multi-Layer Analysis**: Train probes and autoencoders across all model layers
- **Cross-Scenario Testing**: Test generalization across different deception contexts
- **Causal Validation**: Prove causal relationships through activation patching
- **Comprehensive Visualization**: Interactive dashboards and static plots
- **Flexible Data Format**: Easy CSV-based data loading
- **Extensible Design**: Easy to add new deception scenarios or analysis methods

## Based on LLMProbe

This infrastructure builds upon the [LLMProbe](https://github.com/jammastergirish/LLMProbe) framework, extending it specifically for deception circuit research with:

- Enhanced CSV data loading
- Multi-scenario support
- Causal testing capabilities
- Deception-specific analysis tools

## Citation

If you use this infrastructure in your research, please cite:

```bibtex
@misc{deception_circuits_2024,
  title={Deception Circuits Research Infrastructure},
  author={Krish Jain},
  year={2024},
  url={https://github.com/krishjainm/Deception-LLMs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
