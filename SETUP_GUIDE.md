# Deception Circuits Research Infrastructure - Setup Guide

## 🎯 Project Overview

This repository contains a comprehensive framework for discovering and manipulating deception circuits in LLM reasoning traces, implementing the methodology from the BLKY project: **"Can deception circuits be discovered and causally manipulated inside LLM reasoning traces?"**

## 📁 Repository Structure

```
Deception-LLMs/
├── deception_circuits/           # Main framework package
│   ├── __init__.py              # Package interface with comprehensive imports
│   ├── data_loader.py           # CSV data loading with activation support
│   ├── linear_probe.py          # Linear probe implementation for deception detection
│   ├── sparse_autoencoder.py    # Sparse autoencoder for feature discovery
│   ├── training_pipeline.py     # Main orchestration class
│   ├── analysis.py              # Analysis and visualization tools
│   └── activation_patching.py   # Causal testing framework
├── example_usage.py             # Complete demonstration script
├── test_framework.py            # Comprehensive test suite
├── requirements.txt             # All necessary dependencies
├── README.md                    # Detailed documentation
└── SETUP_GUIDE.md              # This setup guide
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/krishjainm/Deception-LLMs.git
cd Deception-LLMs

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Example

```bash
# Run the complete example experiment
python example_usage.py
```

### 3. Test the Framework

```bash
# Run comprehensive tests
python test_framework.py
```

## 🔬 Research Methodology

This framework implements the core BLKY project methodology:

### **Deception Scenarios**
- **Poker Bluffing**: Strategic deception in game contexts
- **Sandbagging**: Deliberate underperformance  
- **Roleplay**: Character-based deception
- **Pressure/Blackmail**: Deception under threat
- **Password Gating**: Conditional knowledge hiding

### **Analysis Pipeline**
1. **Data Collection**: Paired truthful vs deceptive examples
2. **Circuit Discovery**: Linear probes + sparse autoencoders across layers
3. **Causal Testing**: Activation patching, steering, cross-context generalization
4. **Analysis**: Feature importance, layer-wise performance, generalization gaps

## 📊 CSV Data Format

Your deception data should follow this format:

```csv
statement,response,label,scenario
"What is 2+2?","4",0,general
"What is 2+2?","I'm not sure",1,sandbagging
"Do you have a strong hand?","Yes, very strong",0,poker
"Do you have a strong hand?","Yes, very strong",1,poker
```

Where:
- `statement`: Input prompt/question
- `response`: Model's response  
- `label`: 0 = truthful, 1 = deceptive
- `scenario`: Type of deception (poker, roleplay, sandbagging, etc.)

## 🧪 Key Experiments

### **Linear Probe Training**
```python
from deception_circuits import DeceptionTrainingPipeline

pipeline = DeceptionTrainingPipeline(device="cpu")
results = pipeline.run_full_experiment("your_data.csv")
```

### **Causal Testing**
```python
from deception_circuits import CausalTester

tester = CausalTester(device="cpu")
# Test if replacing deceptive activations with truthful ones reduces deception
results = tester.test_deception_suppression(truthful_acts, deceptive_acts, ...)
```

### **Cross-Scenario Analysis**
```python
# Test if poker deception circuits work for sandbagging scenarios
cross_results = pipeline._cross_scenario_analysis(...)
```

## 📈 Expected Results

The framework will help you discover:

1. **Most Informative Layers**: Which transformer layers contain the strongest deception signals
2. **Deception Features**: Specific neurons/features that correlate with deception
3. **Circuit Generalization**: Whether deception circuits work across different scenarios
4. **Causal Relationships**: Whether discovered circuits actually drive deceptive behavior

## 🔧 Advanced Usage

### **Custom Deception Scenarios**
Add new scenarios by extending the CSV format and updating the data loader.

### **Model Integration**
The framework works with any model that can provide layer-wise activations. Simply save activations as PyTorch tensors and point the data loader to them.

### **Visualization**
Generate interactive dashboards and static plots:
```python
from deception_circuits import VisualizationTools

viz = VisualizationTools()
fig = viz.plot_layer_performance(results)
fig.savefig("layer_performance.png")
```

## 📚 Key Files Explained

- **`deception_circuits/__init__.py`**: Main package interface with clear documentation
- **`deception_circuits/data_loader.py`**: Handles CSV loading with comprehensive validation
- **`deception_circuits/linear_probe.py`**: Implements linear classifiers for deception detection
- **`deception_circuits/sparse_autoencoder.py`**: Discovers interpretable features in activations
- **`deception_circuits/training_pipeline.py`**: Orchestrates complete experiments
- **`deception_circuits/analysis.py`**: Analyzes results and identifies circuits
- **`deception_circuits/activation_patching.py`**: Implements causal testing experiments

## 🎯 Research Questions This Framework Answers

1. **Where does deception emerge?** Layer-wise analysis shows where deception signals first appear
2. **What features matter?** Feature importance analysis identifies key neurons/dimensions
3. **Do circuits generalize?** Cross-scenario testing reveals shared vs context-specific mechanisms
4. **Are circuits causal?** Activation patching proves whether circuits drive behavior
5. **Can we control deception?** Steering experiments test intervention possibilities

## 🚀 Next Steps

1. **Collect Data**: Gather paired truthful/deceptive examples from your target scenarios
2. **Run Experiments**: Use the framework to discover deception circuits
3. **Analyze Results**: Identify the most promising circuits for intervention
4. **Test Causality**: Use activation patching to prove causal relationships
5. **Publish Findings**: Document discoveries about deception mechanisms

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{deception_circuits_2025,
  title={Deception Circuits Research Infrastructure},
  author={Krish Jain},
  year={2025},
  url={https://github.com/krishjainm/Deception-LLMs}
}
```



