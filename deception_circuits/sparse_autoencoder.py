"""
Sparse autoencoder implementation for deception circuit feature discovery.

This module implements sparse autoencoders for discovering interpretable features
related to deception in model activations. Sparse autoencoders are a key tool in
mechanistic interpretability because they:

1. Learn to reconstruct activations using a sparse set of features
2. Discover interpretable directions in activation space
3. Can identify specific features that correlate with deception
4. Provide a compressed representation of the activation space

The module contains three main classes:
1. DeceptionSparseAutoencoder: Unsupervised sparse autoencoder
2. SupervisedDeceptionAutoencoder: Joint reconstruction + classification
3. AutoencoderTrainer: Handles training of both types

Sparse autoencoders work by:
- Encoding activations into a bottleneck layer with sparsity constraints
- Decoding back to the original activation space
- Learning features that can reconstruct the activations while being sparse
- Optionally learning to classify deception simultaneously

Based on the LLMProbe sparse autoencoder architecture, adapted specifically
for deception research to discover interpretable features in model activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path


class DeceptionSparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for discovering deception-related features.
    
    This autoencoder learns to reconstruct model activations while discovering
    sparse, interpretable features that may correspond to deception circuits.
    The key insight is that if deception follows specific patterns in the
    activation space, the autoencoder should be able to learn these patterns
    as sparse features.
    
    The autoencoder architecture:
    1. Encoder: Maps activations to a bottleneck representation
    2. Sparsity constraint: Only a subset of bottleneck neurons are active
    3. Decoder: Reconstructs original activations from sparse representation
    4. Tied weights: Optional weight sharing between encoder and decoder
    
    Attributes:
        input_dim (int): Dimension of input activations
        bottleneck_dim (int): Dimension of bottleneck layer
        tied_weights (bool): Whether to tie encoder/decoder weights
        activation_type (str): Type of sparsity constraint ('ReLU' or 'BatchTopK')
        topk_percent (int): Percentage of neurons to keep active (for BatchTopK)
        dropout (float): Dropout rate for regularization
        encoder (nn.Linear): Encoder network
        decoder (nn.Linear): Decoder network
        dropout_layer (nn.Dropout): Optional dropout layer
    """
    
    def __init__(self, input_dim: int, 
                 bottleneck_dim: int = 0,
                 tied_weights: bool = True,
                 activation_type: str = "ReLU",
                 topk_percent: int = 10,
                 dropout: float = 0.0):
        """
        Initialize sparse autoencoder.
        
        Args:
            input_dim: Dimension of input activations (e.g., 768, 1024)
            bottleneck_dim: Dimension of bottleneck layer (0 = same as input)
                          - If > input_dim: overcomplete (more features than inputs)
                          - If < input_dim: undercomplete (fewer features than inputs)
                          - If 0: same as input_dim
            tied_weights: Whether to tie encoder/decoder weights (saves parameters)
            activation_type: Type of sparsity constraint:
                           - 'ReLU': Standard ReLU activation with L1 penalty
                           - 'BatchTopK': Keep only top-k% of neurons active per batch
            topk_percent: Percentage of neurons to keep active (only for BatchTopK)
            dropout: Dropout rate for regularization (0.0 = no dropout)
        """
        super().__init__()
        
        # Store configuration parameters
        self.input_dim = input_dim
        self.bottleneck_dim = input_dim if bottleneck_dim == 0 else bottleneck_dim
        self.tied_weights = tied_weights
        self.activation_type = activation_type
        self.topk_percent = topk_percent
        self.dropout = dropout
        
        # Encoder: Maps from input activations to bottleneck representation
        # This learns which features are important for reconstructing the input
        self.encoder = nn.Linear(input_dim, self.bottleneck_dim)
        
        # Decoder: Maps from bottleneck representation back to input space
        # Always create even if tied weights to avoid errors during forward pass
        self.decoder = nn.Linear(self.bottleneck_dim, input_dim)
        
        # Dropout layer for regularization during training
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
            
    def batch_topk_activation(self, x: torch.Tensor, percent: int = 10) -> torch.Tensor:
        """
        Apply batch-wise top-k activation for sparsity.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            percent: Percentage of activations to keep
            
        Returns:
            Sparsified tensor
        """
        batch_size, hidden_dim = x.shape
        k = max(1, int(hidden_dim * percent / 100))
        
        # Get top-k values and indices for each example
        _, indices = torch.topk(x.abs(), k, dim=1)
        
        # Create mask
        mask = torch.zeros_like(x)
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, k)
        mask.scatter_(1, indices, 1.0)
        
        return x * mask
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to bottleneck representation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (raw_encoded, activated_encoded)
        """
        # Apply dropout if enabled
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
            
        # Linear encoding
        h = self.encoder(x)
        
        # Apply activation function
        if self.activation_type == "ReLU":
            h_activated = F.relu(h)
        elif self.activation_type == "BatchTopK":
            h_activated = self.batch_topk_activation(h, self.topk_percent)
        else:
            h_activated = F.relu(h)  # Default to ReLU
            
        return h, h_activated
        
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode bottleneck representation back to input space.
        
        Args:
            h: Encoded representation [batch_size, bottleneck_dim]
            
        Returns:
            Reconstructed tensor [batch_size, input_dim]
        """
        if self.tied_weights:
            # Manually tie weights before decoding
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
                
        return self.decoder(h)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed, activated_encoded, raw_encoded)
        """
        h, h_activated = self.encode(x)
        x_reconstructed = self.decode(h_activated)
        
        return x_reconstructed, h_activated, h
        
    def get_reconstruction_loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss (MSE)."""
        return F.mse_loss(x_reconstructed, x)
        
    def get_sparsity_loss(self, h_activated: torch.Tensor, l1_coeff: float = 0.01) -> torch.Tensor:
        """Calculate L1 sparsity penalty."""
        return l1_coeff * torch.mean(torch.abs(h_activated))
        
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations for analysis."""
        _, h_activated, _ = self.forward(x)
        return h_activated
        
    def get_feature_importance(self, x: torch.Tensor) -> Dict:
        """Analyze which features are most active."""
        h_activated = self.get_feature_activations(x)
        
        # Calculate feature statistics
        feature_means = h_activated.mean(dim=0)
        feature_stds = h_activated.std(dim=0)
        feature_sparsity = (h_activated == 0).float().mean(dim=0)
        
        # Find most active features
        top_features = torch.topk(feature_means, k=min(50, len(feature_means)))
        
        return {
            'feature_means': feature_means.detach().cpu().numpy(),
            'feature_stds': feature_stds.detach().cpu().numpy(),
            'feature_sparsity': feature_sparsity.detach().cpu().numpy(),
            'top_features': {
                'indices': top_features.indices.detach().cpu().numpy(),
                'values': top_features.values.detach().cpu().numpy()
            }
        }


class SupervisedDeceptionAutoencoder(nn.Module):
    """
    Supervised sparse autoencoder that learns deception classification
    alongside reconstruction.
    """
    
    def __init__(self, input_dim: int,
                 bottleneck_dim: int = 0,
                 tied_weights: bool = True,
                 activation_type: str = "ReLU",
                 topk_percent: int = 10,
                 dropout: float = 0.0):
        """
        Initialize supervised autoencoder.
        
        Args:
            input_dim: Dimension of input activations
            bottleneck_dim: Dimension of bottleneck layer
            tied_weights: Whether to tie encoder/decoder weights
            activation_type: Type of activation function
            topk_percent: Percentage for top-k activation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = input_dim if bottleneck_dim == 0 else bottleneck_dim
        self.tied_weights = tied_weights
        self.activation_type = activation_type
        self.topk_percent = topk_percent
        self.dropout = dropout
        
        # Encoder
        self.encoder = nn.Linear(input_dim, self.bottleneck_dim)
        
        # Decoder
        self.decoder = nn.Linear(self.bottleneck_dim, input_dim)
        
        # Classification head
        self.classifier = nn.Linear(self.bottleneck_dim, 1)
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
            
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to bottleneck representation."""
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
            
        h = self.encoder(x)
        
        if self.activation_type == "ReLU":
            h_activated = F.relu(h)
        elif self.activation_type == "BatchTopK":
            h_activated = self.batch_topk_activation(h, self.topk_percent)
        else:
            h_activated = F.relu(h)
            
        return h, h_activated
        
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation."""
        if self.tied_weights:
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
        return self.decoder(h)
        
    def classify(self, h_activated: torch.Tensor) -> torch.Tensor:
        """Classify deception from encoded features."""
        return torch.sigmoid(self.classifier(h_activated)).squeeze(-1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through supervised autoencoder.
        
        Returns:
            Tuple of (reconstructed, activated_encoded, raw_encoded, classification_probs)
        """
        h, h_activated = self.encode(x)
        x_reconstructed = self.decode(h_activated)
        classification_probs = self.classify(h_activated)
        
        return x_reconstructed, h_activated, h, classification_probs
        
    def get_reconstruction_loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss."""
        return F.mse_loss(x_reconstructed, x)
        
    def get_sparsity_loss(self, h_activated: torch.Tensor, l1_coeff: float = 0.01) -> torch.Tensor:
        """Calculate sparsity loss."""
        return l1_coeff * torch.mean(torch.abs(h_activated))
        
    def get_classification_loss(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss."""
        return F.binary_cross_entropy(probs, labels)
        
    def batch_topk_activation(self, x: torch.Tensor, percent: int = 10) -> torch.Tensor:
        """Apply batch-wise top-k activation."""
        batch_size, hidden_dim = x.shape
        k = max(1, int(hidden_dim * percent / 100))
        
        _, indices = torch.topk(x.abs(), k, dim=1)
        mask = torch.zeros_like(x)
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, k)
        mask.scatter_(1, indices, 1.0)
        
        return x * mask
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict:
        """
        Get feature importance for interpretability.
        
        Args:
            x: Input activations [batch_size, input_dim]
            
        Returns:
            Dictionary with feature importance metrics
        """
        with torch.no_grad():
            # Get bottleneck representation
            encoded, _ = self.encode(x)
            
            # Calculate feature importance as encoder weights
            feature_importance = torch.abs(self.encoder.weight).mean(dim=0)
            
            # Get top features
            top_features = torch.topk(feature_importance, k=min(10, len(feature_importance)))
            
            return {
                'feature_importance': feature_importance.detach().cpu().numpy(),
                'top_features': top_features.indices.detach().cpu().numpy(),
                'top_scores': top_features.values.detach().cpu().numpy(),
                'encoded_mean': encoded.mean(dim=0).detach().cpu().numpy(),
                'encoded_std': encoded.std(dim=0).detach().cpu().numpy()
            }


class AutoencoderTrainer:
    """
    Trainer for sparse autoencoders on deception detection.
    
    Handles both unsupervised and supervised training.
    """
    
    def __init__(self, device: str = "cpu", lr: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            device: Device for training
            lr: Learning rate
        """
        self.device = device
        self.lr = lr
        
    def train_unsupervised_autoencoder(self, activations: torch.Tensor,
                                     epochs: int = 100,
                                     l1_coeff: float = 0.01,
                                     bottleneck_dim: int = 0,
                                     tied_weights: bool = True,
                                     activation_type: str = "ReLU",
                                     topk_percent: int = 10,
                                     validation_split: float = 0.2,
                                     early_stopping_patience: int = 10) -> Dict:
        """
        Train unsupervised sparse autoencoder.
        
        Args:
            activations: Activation tensor [batch_size, input_dim]
            epochs: Number of training epochs
            l1_coeff: L1 sparsity coefficient
            bottleneck_dim: Bottleneck dimension
            tied_weights: Whether to tie weights
            activation_type: Activation function type
            topk_percent: Top-k percentage
            validation_split: Validation split
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        activations = activations.to(self.device)
        
        # Split data
        n_train = int(len(activations) * (1 - validation_split))
        train_activations = activations[:n_train]
        val_activations = activations[n_train:]
        
        # Initialize autoencoder
        autoencoder = DeceptionSparseAutoencoder(
            input_dim=activations.shape[1],
            bottleneck_dim=bottleneck_dim,
            tied_weights=tied_weights,
            activation_type=activation_type,
            topk_percent=topk_percent
        ).to(self.device)
        
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.lr)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_reconstruction': [],
            'val_reconstruction': [],
            'train_sparsity': [],
            'val_sparsity': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            autoencoder.train()
            optimizer.zero_grad()
            
            reconstructed, h_activated, _ = autoencoder(train_activations)
            recon_loss = autoencoder.get_reconstruction_loss(train_activations, reconstructed)
            sparsity_loss = autoencoder.get_sparsity_loss(h_activated, l1_coeff)
            total_loss = recon_loss + sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Validation
            autoencoder.eval()
            with torch.no_grad():
                val_reconstructed, val_h_activated, _ = autoencoder(val_activations)
                val_recon_loss = autoencoder.get_reconstruction_loss(val_activations, val_reconstructed)
                val_sparsity_loss = autoencoder.get_sparsity_loss(val_h_activated, l1_coeff)
                val_total_loss = val_recon_loss + val_sparsity_loss
                
            # Record metrics
            history['train_loss'].append(total_loss.item())
            history['val_loss'].append(val_total_loss.item())
            history['train_reconstruction'].append(recon_loss.item())
            history['val_reconstruction'].append(val_recon_loss.item())
            history['train_sparsity'].append(sparsity_loss.item())
            history['val_sparsity'].append(val_sparsity_loss.item())
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                best_state = autoencoder.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best model
        autoencoder.load_state_dict(best_state)
        
        # Calculate final metrics
        autoencoder.eval()
        with torch.no_grad():
            final_reconstructed, final_h_activated, _ = autoencoder(val_activations)
            final_recon_loss = autoencoder.get_reconstruction_loss(val_activations, final_reconstructed)
            final_sparsity_loss = autoencoder.get_sparsity_loss(final_h_activated, l1_coeff)
            
            # Calculate sparsity percentage
            sparsity_percentage = 100 * (final_h_activated == 0).float().mean().item()
            
        results = {
            'autoencoder': autoencoder,
            'history': history,
            'final_metrics': {
                'reconstruction_loss': final_recon_loss.item(),
                'sparsity_loss': final_sparsity_loss.item(),
                'total_loss': final_recon_loss.item() + final_sparsity_loss.item(),
                'sparsity_percentage': sparsity_percentage
            },
            'best_epoch': len(history['val_loss']) - patience_counter - 1
        }
        
        return results
        
    def train_supervised_autoencoder(self, activations: torch.Tensor,
                                   labels: torch.Tensor,
                                   epochs: int = 100,
                                   l1_coeff: float = 0.01,
                                   lambda_classify: float = 1.0,
                                   bottleneck_dim: int = 0,
                                   tied_weights: bool = True,
                                   activation_type: str = "ReLU",
                                   topk_percent: int = 10,
                                   validation_split: float = 0.2,
                                   early_stopping_patience: int = 10) -> Dict:
        """
        Train supervised sparse autoencoder.
        
        Args:
            activations: Activation tensor [batch_size, input_dim]
            labels: Binary labels [batch_size]
            epochs: Number of training epochs
            l1_coeff: L1 sparsity coefficient
            lambda_classify: Weight for classification loss
            bottleneck_dim: Bottleneck dimension
            tied_weights: Whether to tie weights
            activation_type: Activation function type
            topk_percent: Top-k percentage
            validation_split: Validation split
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        activations = activations.to(self.device)
        labels = labels.to(self.device)
        
        # Split data
        n_train = int(len(activations) * (1 - validation_split))
        train_activations = activations[:n_train]
        train_labels = labels[:n_train]
        val_activations = activations[n_train:]
        val_labels = labels[n_train:]
        
        # Initialize autoencoder
        autoencoder = SupervisedDeceptionAutoencoder(
            input_dim=activations.shape[1],
            bottleneck_dim=bottleneck_dim,
            tied_weights=tied_weights,
            activation_type=activation_type,
            topk_percent=topk_percent
        ).to(self.device)
        
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.lr)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_reconstruction': [],
            'val_reconstruction': [],
            'train_sparsity': [],
            'val_sparsity': [],
            'train_classification': [],
            'val_classification': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            autoencoder.train()
            optimizer.zero_grad()
            
            reconstructed, h_activated, _, classification_probs = autoencoder(train_activations)
            
            recon_loss = autoencoder.get_reconstruction_loss(train_activations, reconstructed)
            sparsity_loss = autoencoder.get_sparsity_loss(h_activated, l1_coeff)
            class_loss = autoencoder.get_classification_loss(classification_probs, train_labels)
            
            total_loss = recon_loss + sparsity_loss + lambda_classify * class_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            train_preds = (classification_probs > 0.5).long()
            train_acc = (train_preds == train_labels).float().mean().item()
            
            # Validation
            autoencoder.eval()
            with torch.no_grad():
                val_reconstructed, val_h_activated, _, val_classification_probs = autoencoder(val_activations)
                
                val_recon_loss = autoencoder.get_reconstruction_loss(val_activations, val_reconstructed)
                val_sparsity_loss = autoencoder.get_sparsity_loss(val_h_activated, l1_coeff)
                val_class_loss = autoencoder.get_classification_loss(val_classification_probs, val_labels)
                
                val_total_loss = val_recon_loss + val_sparsity_loss + lambda_classify * val_class_loss
                
                val_preds = (val_classification_probs > 0.5).long()
                val_acc = (val_preds == val_labels).float().mean().item()
                
            # Record metrics
            history['train_loss'].append(total_loss.item())
            history['val_loss'].append(val_total_loss.item())
            history['train_reconstruction'].append(recon_loss.item())
            history['val_reconstruction'].append(val_recon_loss.item())
            history['train_sparsity'].append(sparsity_loss.item())
            history['val_sparsity'].append(val_sparsity_loss.item())
            history['train_classification'].append(class_loss.item())
            history['val_classification'].append(val_class_loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                best_state = autoencoder.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best model
        autoencoder.load_state_dict(best_state)
        
        # Final evaluation
        autoencoder.eval()
        with torch.no_grad():
            final_reconstructed, final_h_activated, _, final_classification_probs = autoencoder(val_activations)
            final_recon_loss = autoencoder.get_reconstruction_loss(val_activations, final_reconstructed)
            final_sparsity_loss = autoencoder.get_sparsity_loss(final_h_activated, l1_coeff)
            final_class_loss = autoencoder.get_classification_loss(final_classification_probs, val_labels)
            
            final_preds = (final_classification_probs > 0.5).long()
            final_acc = (final_preds == val_labels).float().mean().item()
            sparsity_percentage = 100 * (final_h_activated == 0).float().mean().item()
            
        results = {
            'autoencoder': autoencoder,
            'history': history,
            'final_metrics': {
                'reconstruction_loss': final_recon_loss.item(),
                'sparsity_loss': final_sparsity_loss.item(),
                'classification_loss': final_class_loss.item(),
                'total_loss': final_recon_loss.item() + final_sparsity_loss.item() + lambda_classify * final_class_loss.item(),
                'accuracy': final_acc,
                'sparsity_percentage': sparsity_percentage
            },
            'best_epoch': len(history['val_loss']) - patience_counter - 1
        }
        
        return results
        
    def analyze_features(self, autoencoder: Union[DeceptionSparseAutoencoder, SupervisedDeceptionAutoencoder],
                        activations: torch.Tensor,
                        labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Analyze discovered features in the autoencoder.
        
        Args:
            autoencoder: Trained autoencoder
            activations: Input activations
            labels: Optional labels for supervised analysis
            
        Returns:
            Dictionary with feature analysis
        """
        autoencoder.eval()
        activations = activations.to(self.device)
        
        with torch.no_grad():
            # Get feature activations
            if isinstance(autoencoder, SupervisedDeceptionAutoencoder):
                _, h_activated, _, _ = autoencoder(activations)
            else:
                _, h_activated, _ = autoencoder(activations)
                
            # Analyze features
            feature_analysis = autoencoder.get_feature_importance(activations)
            
            # Add label-based analysis if available
            if labels is not None:
                labels = labels.to(self.device)
                truthful_mask = labels == 0
                deceptive_mask = labels == 1
                
                if truthful_mask.sum() > 0 and deceptive_mask.sum() > 0:
                    truthful_features = h_activated[truthful_mask]
                    deceptive_features = h_activated[deceptive_mask]
                    
                    # Compare feature activations between truthful and deceptive
                    truthful_means = truthful_features.mean(dim=0)
                    deceptive_means = deceptive_features.mean(dim=0)
                    
                    feature_differences = deceptive_means - truthful_means
                    top_deceptive_features = torch.topk(feature_differences.abs(), k=min(20, len(feature_differences)))
                    
                    feature_analysis['deception_analysis'] = {
                        'feature_differences': feature_differences.detach().cpu().numpy(),
                        'top_deceptive_features': {
                            'indices': top_deceptive_features.indices.detach().cpu().numpy(),
                            'differences': top_deceptive_features.values.detach().cpu().numpy()
                        }
                    }
                    
        return feature_analysis
