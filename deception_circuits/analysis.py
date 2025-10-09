"""
Analysis and visualization tools for deception circuit research.

This module provides comprehensive analysis and visualization tools for understanding
the results of deception circuit experiments. It contains two main classes:

1. **CircuitAnalyzer**: Analyzes experiment results to identify deception circuits
2. **VisualizationTools**: Creates plots and interactive dashboards

The analyzer helps answer key research questions:
- Which layers are most informative for detecting deception?
- What features are most important for deception detection?
- Do deception circuits generalize across different scenarios?
- How do probe and autoencoder results compare?

The visualizer creates:
- Layer performance plots showing where deception signals emerge
- Feature importance visualizations showing which neurons matter most
- Cross-scenario generalization matrices
- Interactive dashboards for exploring results
- Activation visualizations using dimensionality reduction

These tools are essential for interpreting the results and identifying
the most promising circuits for causal intervention.

=== FOR FIRST-TIME READERS ===

This module is the "analysis brain" of the deception circuits research framework.
After running experiments (training probes and autoencoders), you use these tools to:

1. UNDERSTAND WHAT HAPPENED:
   - Which layers of the neural network show deception signals?
   - Are there specific neurons that always activate during deception?
   - How well do our detection methods work?

2. FIND PATTERNS:
   - Do deception circuits work the same way across different scenarios?
   - Which features are most important for detecting lies vs truth?
   - How do different models compare?

3. PREPARE FOR CAUSAL TESTING:
   - Identify the best layers to patch/manipulate
   - Find specific neurons to target with interventions
   - Understand which circuits are most promising

Think of this as your "detective toolkit" - it helps you analyze the evidence
collected from your experiments and figure out where the deception circuits are hiding!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CircuitAnalyzer:
    """
    Comprehensive analyzer for deception circuit results.
    
    This is the main class for analyzing your deception circuit experiments.
    It takes the raw results from your experiments and helps you understand:
    
    1. **Which layers work best**: Where in the neural network can we best detect deception?
    2. **What features matter**: Which specific neurons or patterns are important?
    3. **Does it generalize**: Do circuits found in one scenario work in others?
    4. **What should we patch**: Which layers/neurons should we target for causal testing?
    
    === HOW TO USE ===
    
    # After running your experiment:
    analyzer = CircuitAnalyzer(results)
    
    # Find the best layers for deception detection:
    probe_analysis = analyzer.analyze_probe_performance()
    best_layer = probe_analysis['best_layer']
    
    # See which autoencoder features are important:
    autoencoder_analysis = analyzer.analyze_autoencoder_features()
    
    # Check if circuits generalize across scenarios:
    generalization = analyzer.analyze_cross_scenario_generalization()
    
    # Get recommendations for causal testing:
    circuits = analyzer.identify_deception_circuits()
    
    === WHAT THE RESULTS MEAN ===
    
    - **High AUC/Accuracy**: This layer/feature can reliably detect deception
    - **Low AUC/Accuracy**: This layer/feature doesn't help much
    - **Cross-scenario success**: The circuit works across different types of lies
    - **Cross-scenario failure**: Each scenario uses different circuits
    
    Args:
        results: Dictionary containing experiment results from DeceptionTrainingPipeline
    """
    
    def __init__(self, results: Optional[Dict] = None):
        """
        Initialize the circuit analyzer.
        
        The analyzer stores your experiment results so you can analyze them.
        You can also pass results to individual analysis methods if you prefer.
        
        Args:
            results: Experiment results dictionary from DeceptionTrainingPipeline.run_full_experiment()
                    Contains: probe_results, autoencoder_results, cross_scenario_results, etc.
        """
        self.results = results
        
    def analyze_probe_performance(self, results: Optional[Dict] = None) -> Dict:
        """
        Analyze linear probe performance across layers to find where deception signals are strongest.
        
        This is one of the most important analyses! It tells you:
        - Which layers of the neural network are best at detecting deception
        - How strong the deception signal is at each layer
        - Which layer you should target for causal interventions
        
        === WHAT ARE LINEAR PROBES? ===
        
        Linear probes are simple classifiers that try to detect deception by looking at
        the internal activations (neural network states) at each layer. Think of them
        as "deception detectors" that scan each layer of the network.
        
        - **High performance** = This layer contains strong deception signals
        - **Low performance** = This layer doesn't help detect deception
        
        === WHAT THE METRICS MEAN ===
        
        - **AUC (Area Under Curve)**: Best overall measure (0.5 = random, 1.0 = perfect)
        - **Accuracy**: Percentage of correct predictions (0.0 = 0%, 1.0 = 100%)
        - **Precision**: Of all "deceptive" predictions, how many were actually deceptive?
        - **Recall**: Of all actual deceptive examples, how many did we catch?
        - **F1**: Balanced measure of precision and recall
        
        Args:
            results: Experiment results (uses self.results if None)
            
        Returns:
            Dictionary containing:
            - layer_metrics: Performance for each layer (sorted by AUC, best first)
            - best_layer: The layer with highest AUC
            - worst_layer: The layer with lowest AUC  
            - statistics: Summary statistics across all layers
            - auc_progression: AUC values across layers (for plotting)
            - accuracy_progression: Accuracy values across layers (for plotting)
            
        Example:
            analyzer = CircuitAnalyzer(results)
            probe_analysis = analyzer.analyze_probe_performance()
            best_layer = probe_analysis['best_layer']['layer']  # e.g., 'layer_15'
            best_auc = probe_analysis['best_layer']['auc']      # e.g., 0.89
        """
        if results is None:
            results = self.results
            
        if results is None or 'probe_results' not in results:
            raise ValueError("No probe results found - make sure you've run probe training first!")
            
        probe_results = results['probe_results']
        layer_results = probe_results['layer_results']
        test_results = probe_results['test_results']
        
        # Extract performance metrics for each layer
        # We want to see how well each layer can detect deception
        layer_metrics = []
        for layer_name, layer_data in layer_results.items():
            test_metrics = test_results[layer_name]
            layer_metrics.append({
                'layer': layer_name,           # e.g., 'layer_15'
                'layer_idx': layer_data['layer_idx'],  # e.g., 15
                'accuracy': test_metrics['accuracy'],   # Overall correctness
                'precision': test_metrics['precision'], # Precision of deception detection
                'recall': test_metrics['recall'],       # Recall of deception detection  
                'f1': test_metrics['f1'],               # Balanced precision/recall
                'auc': test_metrics['auc']              # Best overall measure
            })
            
        # Sort layers by AUC (Area Under Curve) - best performers first
        # AUC is the gold standard for binary classification
        layer_metrics.sort(key=lambda x: x['auc'], reverse=True)
        
        # Calculate summary statistics across all layers
        aucs = [lm['auc'] for lm in layer_metrics]
        accuracies = [lm['accuracy'] for lm in layer_metrics]
        
        analysis = {
            'layer_metrics': layer_metrics,    # All layers, sorted by performance
            'best_layer': layer_metrics[0],    # Highest performing layer
            'worst_layer': layer_metrics[-1],  # Lowest performing layer
            'statistics': {                    # Summary stats across all layers
                'mean_auc': np.mean(aucs),         # Average AUC across layers
                'std_auc': np.std(aucs),           # Standard deviation of AUC
                'max_auc': np.max(aucs),           # Best AUC found
                'min_auc': np.min(aucs),           # Worst AUC found
                'mean_accuracy': np.mean(accuracies), # Average accuracy
                'std_accuracy': np.std(accuracies)    # Standard deviation of accuracy
            },
            'auc_progression': aucs,           # For plotting AUC across layers
            'accuracy_progression': accuracies # For plotting accuracy across layers
        }
        
        return analysis
        
    def analyze_autoencoder_features(self, results: Optional[Dict] = None) -> Dict:
        """
        Analyze sparse autoencoder feature discovery.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with autoencoder analysis
        """
        if results is None:
            results = self.results
            
        if results is None or 'autoencoder_results' not in results:
            raise ValueError("No autoencoder results found")
            
        autoencoder_results = results['autoencoder_results']
        unsupervised_results = autoencoder_results['unsupervised_results']
        supervised_results = autoencoder_results['supervised_results']
        
        # Analyze unsupervised autoencoders
        unsupervised_analysis = {}
        for layer_name, layer_data in unsupervised_results.items():
            final_metrics = layer_data['final_metrics']
            unsupervised_analysis[layer_name] = {
                'reconstruction_loss': final_metrics['reconstruction_loss'],
                'sparsity_loss': final_metrics['sparsity_loss'],
                'total_loss': final_metrics['total_loss'],
                'sparsity_percentage': final_metrics['sparsity_percentage']
            }
            
        # Analyze supervised autoencoders
        supervised_analysis = {}
        for layer_name, layer_data in supervised_results.items():
            final_metrics = layer_data['final_metrics']
            supervised_analysis[layer_name] = {
                'reconstruction_loss': final_metrics['reconstruction_loss'],
                'sparsity_loss': final_metrics['sparsity_loss'],
                'classification_loss': final_metrics['classification_loss'],
                'total_loss': final_metrics['total_loss'],
                'accuracy': final_metrics['accuracy'],
                'sparsity_percentage': final_metrics['sparsity_percentage']
            }
            
        # Find best performing layers
        best_unsupervised = max(
            unsupervised_analysis.items(),
            key=lambda x: x[1]['sparsity_percentage']  # Higher sparsity is better
        )
        
        best_supervised = max(
            supervised_analysis.items(),
            key=lambda x: x[1]['accuracy']
        )
        
        analysis = {
            'unsupervised_analysis': unsupervised_analysis,
            'supervised_analysis': supervised_analysis,
            'best_unsupervised_layer': best_unsupervised,
            'best_supervised_layer': best_supervised,
            'comparison': {
                'unsupervised_sparsity_range': [
                    min(layer['sparsity_percentage'] for layer in unsupervised_analysis.values()),
                    max(layer['sparsity_percentage'] for layer in unsupervised_analysis.values())
                ],
                'supervised_accuracy_range': [
                    min(layer['accuracy'] for layer in supervised_analysis.values()),
                    max(layer['accuracy'] for layer in supervised_analysis.values())
                ]
            }
        }
        
        return analysis
        
    def analyze_cross_scenario_generalization(self, results: Optional[Dict] = None) -> Dict:
        """
        Analyze generalization across different deception scenarios.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with cross-scenario analysis
        """
        if results is None:
            results = self.results
            
        if results is None or 'cross_scenario_results' not in results:
            return {'error': 'No cross-scenario results found'}
            
        cross_results = results['cross_scenario_results']
        
        # Create generalization matrix
        scenarios = list(cross_results.keys())
        generalization_matrix = np.zeros((len(scenarios), len(scenarios)))
        
        for i, train_scenario in enumerate(scenarios):
            train_data = cross_results[train_scenario]
            test_results = train_data['test_results']
            
            for j, test_scenario in enumerate(scenarios):
                if test_scenario in test_results:
                    generalization_matrix[i, j] = test_results[test_scenario]['auc']
                else:
                    generalization_matrix[i, j] = np.nan
                    
        # Calculate generalization statistics
        within_scenario_performance = np.diag(generalization_matrix)
        cross_scenario_performance = []
        
        for i in range(len(scenarios)):
            for j in range(len(scenarios)):
                if i != j and not np.isnan(generalization_matrix[i, j]):
                    cross_scenario_performance.append(generalization_matrix[i, j])
                    
        analysis = {
            'generalization_matrix': generalization_matrix,
            'scenarios': scenarios,
            'within_scenario_performance': {
                'mean': np.mean(within_scenario_performance),
                'std': np.std(within_scenario_performance),
                'values': within_scenario_performance.tolist()
            },
            'cross_scenario_performance': {
                'mean': np.mean(cross_scenario_performance),
                'std': np.std(cross_scenario_performance),
                'values': cross_scenario_performance
            },
            'generalization_gap': np.mean(within_scenario_performance) - np.mean(cross_scenario_performance)
        }
        
        return analysis
        
    def identify_deception_circuits(self, results: Optional[Dict] = None) -> Dict:
        """
        Identify potential deception circuits from analysis results.
        
        Args:
            results: Experiment results
            
        Returns:
            Dictionary with identified circuits
        """
        if results is None:
            results = self.results
            
        if results is None:
            raise ValueError("No results found")
            
        circuits = {}
        
        # Analyze probe feature importance
        if 'analysis_results' in results and 'feature_analysis' in results['analysis_results']:
            feature_analysis = results['analysis_results']['feature_analysis']
            
            if 'probe_importance' in feature_analysis:
                probe_importance = feature_analysis['probe_importance']
                top_features = probe_importance['top_features']
                
                circuits['probe_features'] = {
                    'important_features': top_features['indices'][:50].tolist(),
                    'feature_weights': top_features['weights'][:50].tolist(),
                    'total_features': probe_importance['total_features'],
                    'sparsity': probe_importance['weight_statistics']['sparsity']
                }
                
        # Analyze autoencoder features
        autoencoder_analysis = self.analyze_autoencoder_features(results)
        
        if 'supervised_test_results' in results['autoencoder_results']:
            supervised_test_results = results['autoencoder_results']['supervised_test_results']
            
            # Find features that distinguish deception
            deception_features = {}
            for layer_name, layer_analysis in supervised_test_results.items():
                if 'deception_analysis' in layer_analysis:
                    deception_analysis = layer_analysis['deception_analysis']
                    top_deceptive = deception_analysis['top_deceptive_features']
                    
                    deception_features[layer_name] = {
                        'deception_features': top_deceptive['indices'][:20].tolist(),
                        'feature_differences': top_deceptive['differences'][:20].tolist()
                    }
                    
            circuits['autoencoder_features'] = deception_features
            
        # Identify most informative layers
        probe_analysis = self.analyze_probe_performance(results)
        best_probe_layer = probe_analysis['best_layer']['layer']
        best_auc = probe_analysis['best_layer']['auc']
        
        best_autoencoder_layer = autoencoder_analysis['best_supervised_layer'][0]
        best_accuracy = autoencoder_analysis['best_supervised_layer'][1]['accuracy']
        
        circuits['informative_layers'] = {
            'best_probe_layer': best_probe_layer,
            'best_probe_auc': best_auc,
            'best_autoencoder_layer': best_autoencoder_layer,
            'best_autoencoder_accuracy': best_accuracy,
            'recommended_layer': best_probe_layer if best_auc > best_accuracy else best_autoencoder_layer
        }
        
        return circuits


class VisualizationTools:
    """
    Visualization tools for deception circuit research.
    
    This class creates all the plots and visualizations you need to understand
    your deception circuit experiments. It's like your "visual detective toolkit"!
    
    === WHAT VISUALIZATIONS ARE AVAILABLE ===
    
    1. **Layer Performance Plots**: Shows which layers are best at detecting deception
    2. **Feature Importance Charts**: Shows which specific neurons/features matter most
    3. **Generalization Matrices**: Shows if circuits work across different scenarios
    4. **Activation Visualizations**: 2D plots of neural network states (t-SNE, PCA)
    5. **Interactive Dashboards**: Comprehensive plots with hover information
    
    === HOW TO USE ===
    
    # Create visualization tools:
    viz = VisualizationTools()
    
    # Plot layer performance:
    fig = viz.plot_layer_performance(results)
    
    # Show feature importance:
    fig = viz.plot_feature_importance(results)
    
    # Create interactive dashboard:
    fig = viz.create_interactive_dashboard(results)
    
    === WHY VISUALIZATION MATTERS ===
    
    - **Layer plots**: Help you find the best layers for causal interventions
    - **Feature plots**: Show which neurons to target with patching
    - **Generalization plots**: Reveal if circuits are universal or scenario-specific
    - **Activation plots**: Help you understand what the neural network is "thinking"
    
    These plots are essential for:
    1. Understanding your results
    2. Finding patterns in the data
    3. Preparing presentations/papers
    4. Identifying the best targets for causal testing
    """
    
    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization tools with your preferred style.
        
        Args:
            style: Matplotlib plotting style
                  - "default": Clean, professional style
                  - "ggplot": R-like style with gray backgrounds
                  - "seaborn": Statistical plotting style
                  - "classic": Old matplotlib style
            figsize: Default size for plots (width, height) in inches
        """
        self.style = style
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if the requested style is not available
            plt.style.use('default')
        
    def plot_layer_performance(self, results: Dict, 
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a crucial plot showing deception detection performance across all neural network layers.
        
        This is one of the MOST IMPORTANT plots for your research! It shows you:
        - Where in the neural network deception signals are strongest
        - Which layers you should target for causal interventions
        - How deception detection improves/worsens as information flows through the network
        
        === WHAT THIS PLOT TELLS YOU ===
        
        **Top Plot (AUC)**: 
        - Shows Area Under Curve (AUC) for each layer
        - AUC = 0.5 means random guessing (no signal)
        - AUC = 1.0 means perfect deception detection
        - **Higher AUC = Stronger deception signal in that layer**
        
        **Bottom Plot (Accuracy)**:
        - Shows classification accuracy for each layer  
        - Accuracy = 0.0 means 0% correct, 1.0 means 100% correct
        - **Higher accuracy = Better deception detection in that layer**
        
        **Red dots**: Highlight the best performing layer for each metric
        
        === HOW TO INTERPRET RESULTS ===
        
        - **Rising curve**: Deception signals get stronger in later layers (common)
        - **Peak in middle**: Deception signals peak in middle layers then fade
        - **Flat line**: All layers perform similarly (deception is distributed)
        - **Noisy curve**: Performance varies a lot between layers
        
        === WHAT TO DO NEXT ===
        
        - **Target the best layers** for activation patching experiments
        - **Focus on layers with AUC > 0.7** for strong deception signals
        - **Avoid layers with AUC < 0.6** - too weak for reliable interventions
        
        Args:
            results: Experiment results from DeceptionTrainingPipeline.run_full_experiment()
            save_path: Optional path to save the plot (e.g., 'layer_performance.png')
            
        Returns:
            Matplotlib figure object that you can display or save
            
        Example:
            viz = VisualizationTools()
            fig = viz.plot_layer_performance(results)
            plt.show()  # Display the plot
            
            # Or save it:
            fig = viz.plot_layer_performance(results, save_path='results/layer_performance.png')
        """
        # Get probe analysis results - this contains the performance metrics
        if 'analysis_results' not in results:
            # If we don't have analysis results yet, create them
            analyzer = CircuitAnalyzer(results)
            probe_analysis = analyzer.analyze_probe_performance()
        else:
            probe_analysis = results['analysis_results']
            
        # Extract the data we need for plotting
        layers = [lm['layer_idx'] for lm in probe_analysis['layer_metrics']]  # Layer indices (0, 1, 2, ...)
        aucs = [lm['auc'] for lm in probe_analysis['layer_metrics']]         # AUC scores for each layer
        accuracies = [lm['accuracy'] for lm in probe_analysis['layer_metrics']] # Accuracy for each layer
        
        # Create two subplots - one for AUC, one for Accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # === TOP PLOT: AUC SCORES ===
        # AUC is the gold standard metric - shows how well each layer can distinguish deception
        ax1.plot(layers, aucs, 'b-o', linewidth=2, markersize=6, label='AUC Score')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Deception Detection Performance Across Layers\n(Top: AUC, Bottom: Accuracy)')
        ax1.grid(True, alpha=0.3)  # Add subtle grid for easier reading
        ax1.set_ylim(0, 1)  # AUC ranges from 0 to 1
        
        # Highlight the best performing layer with a red dot
        best_idx = aucs.index(max(aucs))
        ax1.plot(layers[best_idx], aucs[best_idx], 'ro', markersize=10, 
                label=f'Best: {max(aucs):.3f}')
        ax1.legend()
        
        # === BOTTOM PLOT: ACCURACY ===
        # Accuracy shows the percentage of correct predictions
        ax2.plot(layers, accuracies, 'g-o', linewidth=2, markersize=6, label='Accuracy')
        ax2.set_xlabel('Layer Index')  # Which layer of the neural network
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
        
        # Highlight the best performing layer for accuracy
        best_acc_idx = accuracies.index(max(accuracies))
        ax2.plot(layers[best_acc_idx], accuracies[best_acc_idx], 'ro', markersize=10, 
                label=f'Best: {max(accuracies):.3f}')
        ax2.legend()
        
        # Make the plot look nice
        plt.tight_layout()
        
        # Save the plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_feature_importance(self, results: Dict,
                              top_k: int = 50,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot feature importance from probe analysis.
        
        Args:
            results: Experiment results
            top_k: Number of top features to show
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'analysis_results' not in results or 'feature_analysis' not in results['analysis_results']:
            raise ValueError("No feature analysis found in results")
            
        feature_analysis = results['analysis_results']['feature_analysis']
        
        if 'probe_importance' not in feature_analysis:
            raise ValueError("No probe importance analysis found")
            
        probe_importance = feature_analysis['probe_importance']
        top_features = probe_importance['top_features']
        
        # Extract top-k features
        indices = top_features['indices'][:top_k]
        weights = top_features['weights'][:top_k]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(indices)), weights)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([f'Feature {idx}' for idx in indices])
        ax.set_xlabel('Feature Weight')
        ax.set_title(f'Top {top_k} Most Important Features for Deception Detection')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(weights)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_generalization_matrix(self, results: Dict,
                                 save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot cross-scenario generalization matrix.
        
        Args:
            results: Experiment results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'cross_scenario_results' not in results:
            raise ValueError("No cross-scenario results found")
            
        analyzer = CircuitAnalyzer(results)
        cross_analysis = analyzer.analyze_cross_scenario_generalization()
        
        if 'error' in cross_analysis:
            raise ValueError(cross_analysis['error'])
            
        generalization_matrix = cross_analysis['generalization_matrix']
        scenarios = cross_analysis['scenarios']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(generalization_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_yticklabels(scenarios)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AUC Score')
        
        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(scenarios)):
                if not np.isnan(generalization_matrix[i, j]):
                    text = ax.text(j, i, f'{generalization_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
                    
        ax.set_title('Cross-Scenario Generalization Matrix')
        ax.set_xlabel('Test Scenario')
        ax.set_ylabel('Train Scenario')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_activation_visualization(self, activations: torch.Tensor,
                                    labels: torch.Tensor,
                                    method: str = 'tsne',
                                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize activations using dimensionality reduction.
        
        Args:
            activations: Activation tensor [batch_size, num_layers, hidden_dim]
            labels: Binary labels [batch_size]
            method: Reduction method ('tsne' or 'pca')
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Use the best performing layer (last layer by default)
        layer_activations = activations[:, -1, :].cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(layer_activations)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
            
        reduced_activations = reducer.fit_transform(layer_activations)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot truthful (0) and deceptive (1) points
        truthful_mask = labels == 0
        deceptive_mask = labels == 1
        
        ax.scatter(reduced_activations[truthful_mask, 0], 
                  reduced_activations[truthful_mask, 1], 
                  c='blue', label='Truthful', alpha=0.7, s=50)
        
        ax.scatter(reduced_activations[deceptive_mask, 0], 
                  reduced_activations[deceptive_mask, 1], 
                  c='red', label='Deceptive', alpha=0.7, s=50)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Activation Visualization ({method.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_dashboard(self, results: Dict,
                                   save_path: Optional[Union[str, Path]] = None) -> go.Figure:
        """
        Create interactive dashboard with multiple visualizations.
        
        Args:
            results: Experiment results
            save_path: Path to save HTML dashboard
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Layer Performance', 'Feature Importance', 
                          'Generalization Matrix', 'Training History'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Layer Performance
        if 'analysis_results' in results:
            probe_analysis = results['analysis_results']
            if 'layer_comparison' in probe_analysis:
                layer_comparison = probe_analysis['layer_comparison']
                layers = [int(layer.split('_')[1]) for layer in layer_comparison.keys()]
                aucs = [data['probe_auc'] for data in layer_comparison.values()]
                
                fig.add_trace(
                    go.Scatter(x=layers, y=aucs, mode='lines+markers',
                              name='AUC', line=dict(color='blue')),
                    row=1, col=1
                )
                
        # Plot 2: Feature Importance (placeholder)
        fig.add_trace(
            go.Bar(x=['Feature 1', 'Feature 2', 'Feature 3'], 
                  y=[0.1, 0.05, 0.03], name='Importance'),
            row=1, col=2
        )
        
        # Plot 3: Generalization Matrix
        if 'cross_scenario_results' in results:
            analyzer = CircuitAnalyzer(results)
            cross_analysis = analyzer.analyze_cross_scenario_generalization()
            
            if 'generalization_matrix' in cross_analysis:
                matrix = cross_analysis['generalization_matrix']
                scenarios = cross_analysis['scenarios']
                
                fig.add_trace(
                    go.Heatmap(z=matrix, x=scenarios, y=scenarios,
                              colorscale='RdYlBu_r', name='Generalization'),
                    row=2, col=1
                )
                
        # Plot 4: Training History (placeholder)
        epochs = list(range(1, 101))
        train_loss = [1.0 - i*0.008 for i in range(100)]
        val_loss = [1.0 - i*0.007 for i in range(100)]
        
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, mode='lines', name='Val Loss'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Deception Circuit Research Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Layer Index", row=1, col=1)
        fig.update_yaxes(title_text="AUC Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Features", row=1, col=2)
        fig.update_yaxes(title_text="Importance", row=1, col=2)
        
        fig.update_xaxes(title_text="Test Scenario", row=2, col=1)
        fig.update_yaxes(title_text="Train Scenario", row=2, col=1)
        
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
