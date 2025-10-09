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
    
    Provides methods for analyzing probe performance, autoencoder features,
    and cross-scenario generalization.
    """
    
    def __init__(self, results: Optional[Dict] = None):
        """
        Initialize analyzer.
        
        Args:
            results: Experiment results dictionary
        """
        self.results = results
        
    def analyze_probe_performance(self, results: Optional[Dict] = None) -> Dict:
        """
        Analyze linear probe performance across layers.
        
        Args:
            results: Experiment results (uses self.results if None)
            
        Returns:
            Dictionary with probe analysis
        """
        if results is None:
            results = self.results
            
        if results is None or 'probe_results' not in results:
            raise ValueError("No probe results found")
            
        probe_results = results['probe_results']
        layer_results = probe_results['layer_results']
        test_results = probe_results['test_results']
        
        # Extract metrics for each layer
        layer_metrics = []
        for layer_name, layer_data in layer_results.items():
            test_metrics = test_results[layer_name]
            layer_metrics.append({
                'layer': layer_name,
                'layer_idx': layer_data['layer_idx'],
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'auc': test_metrics['auc']
            })
            
        # Sort by AUC
        layer_metrics.sort(key=lambda x: x['auc'], reverse=True)
        
        # Calculate statistics
        aucs = [lm['auc'] for lm in layer_metrics]
        accuracies = [lm['accuracy'] for lm in layer_metrics]
        
        analysis = {
            'layer_metrics': layer_metrics,
            'best_layer': layer_metrics[0],
            'worst_layer': layer_metrics[-1],
            'statistics': {
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'max_auc': np.max(aucs),
                'min_auc': np.min(aucs),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            },
            'auc_progression': aucs,
            'accuracy_progression': accuracies
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
    
    Creates plots and interactive visualizations for results analysis.
    """
    
    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization tools.
        
        Args:
            style: Matplotlib style (default, ggplot, classic, etc.)
            figsize: Default figure size
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
        Plot performance across layers.
        
        Args:
            results: Experiment results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if 'analysis_results' not in results:
            analyzer = CircuitAnalyzer(results)
            probe_analysis = analyzer.analyze_probe_performance()
        else:
            probe_analysis = results['analysis_results']
            
        # Extract data
        layers = [lm['layer_idx'] for lm in probe_analysis['layer_metrics']]
        aucs = [lm['auc'] for lm in probe_analysis['layer_metrics']]
        accuracies = [lm['accuracy'] for lm in probe_analysis['layer_metrics']]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot AUC
        ax1.plot(layers, aucs, 'b-o', linewidth=2, markersize=6)
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Deception Detection Performance Across Layers')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Highlight best layer
        best_idx = aucs.index(max(aucs))
        ax1.plot(layers[best_idx], aucs[best_idx], 'ro', markersize=10, label=f'Best: {max(aucs):.3f}')
        ax1.legend()
        
        # Plot Accuracy
        ax2.plot(layers, accuracies, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Highlight best layer
        best_acc_idx = accuracies.index(max(accuracies))
        ax2.plot(layers[best_acc_idx], accuracies[best_acc_idx], 'ro', markersize=10, label=f'Best: {max(accuracies):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
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
