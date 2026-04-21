"""
Deception Probe for DeceptiScope v2
Trains and applies probes to detect deception in model activations

Key Innovation: Supervised learning on activation patterns to identify:
- Honest vs deceptive hidden states
- Layer-specific deception detection
- Feature-level deception indicators
- Cross-model generalization capabilities

This provides the most direct deception detection possible
by analyzing the model's internal representations.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
import joblib

from .extractor import ActivationExtractor, ActivationData

logger = logging.getLogger(__name__)

@dataclass
class ProbeResults:
    """Results of deception probe application"""
    deception_probability: float
    layer_scores: Dict[int, float]
    feature_importance: Dict[int, float]
    confidence_interval: Tuple[float, float]
    most_deceptive_layers: List[int]
    explanation: str

@dataclass
class ProbeConfig:
    """Configuration for deception probe"""
    probe_type: str = "linear"  # linear, mlp, random_forest
    layers: List[int] = None   # None = all layers
    feature_dim: int = 768     # Hidden dimension
    regularization: float = 0.01
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3

class LinearDeceptionProbe(nn.Module):
    """Linear probe for deception detection"""
    
    def __init__(self, input_dim: int, regularization: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.regularization = regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))
    
    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization loss"""
        return self.regularization * torch.norm(self.linear.weight, p=2)

class MLPDeceptionProbe(nn.Module):
    """Multi-layer perceptron probe for deception detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, regularization: float = 0.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.regularization = regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(x))
    
    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization loss"""
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        return self.regularization * reg_loss

class DeceptionProbe:
    """
    Trained probe for detecting deception in model activations
    
    Critical innovation: Learns to identify deception patterns
    directly from model internal representations.
    """
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Probes for each layer
        self.layer_probes: Dict[int, nn.Module] = {}
        
        # Ensemble probe (combines all layers)
        self.ensemble_probe: Optional[nn.Module] = None
        
        # Training history
        self.training_history: List[Dict[str, float]] = []
        
        # Feature importance
        self.feature_importance: Dict[int, np.ndarray] = {}
        
        logger.info(f"Initialized DeceptionProbe with {config.probe_type} probes")
    
    def train_probes(
        self,
        activation_data: List[Tuple[ActivationData, int]],  # (activations, deception_label)
        validation_data: Optional[List[Tuple[ActivationData, int]]] = None
    ) -> Dict[str, float]:
        """
        Train deception probes on activation data
        
        Args:
            activation_data: List of (activations, label) pairs
            validation_data: Optional validation set
            
        Returns:
            Training metrics
        """
        
        logger.info(f"Training probes on {len(activation_data)} samples")
        
        # Determine layers to train
        if self.config.layers is None:
            # Use all available layers
            all_layers = set()
            for activations, _ in activation_data:
                all_layers.update(activations.layer_activations.keys())
            layers_to_train = sorted(list(all_layers))
        else:
            layers_to_train = self.config.layers
        
        # Train individual layer probes
        layer_metrics = {}
        
        for layer_idx in layers_to_train:
            try:
                metrics = self._train_layer_probe(layer_idx, activation_data, validation_data)
                layer_metrics[f"layer_{layer_idx}"] = metrics
                logger.info(f"Layer {layer_idx} probe: AUC = {metrics['auc']:.3f}")
            except Exception as e:
                logger.error(f"Failed to train layer {layer_idx} probe: {e}")
        
        # Train ensemble probe
        try:
            ensemble_metrics = self._train_ensemble_probe(activation_data, validation_data)
            layer_metrics["ensemble"] = ensemble_metrics
            logger.info(f"Ensemble probe: AUC = {ensemble_metrics['auc']:.3f}")
        except Exception as e:
            logger.error(f"Failed to train ensemble probe: {e}")
        
        # Calculate overall metrics
        overall_auc = np.mean([m['auc'] for m in layer_metrics.values() if 'auc' in m])
        
        return {
            "overall_auc": overall_auc,
            "layer_metrics": layer_metrics,
            "num_layers_trained": len(layer_metrics)
        }
    
    def _train_layer_probe(
        self,
        layer_idx: int,
        activation_data: List[Tuple[ActivationData, int]],
        validation_data: Optional[List[Tuple[ActivationData, int]]] = None
    ) -> Dict[str, float]:
        """Train probe for specific layer"""
        
        # Extract layer activations and labels
        X, y = self._extract_layer_data(layer_idx, activation_data)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data for layer {layer_idx}: {len(X)} samples")
        
        # Split train/validation if no validation data provided
        if validation_data is None:
            val_split = int(0.2 * len(X))
            X_train, X_val = X[:-val_split], X[-val_split:]
            y_train, y_val = y[:-val_split], y[-val_split:]
        else:
            X_train, y_train = self._extract_layer_data(layer_idx, activation_data)
            X_val, y_val = self._extract_layer_data(layer_idx, validation_data)
        
        # Create probe
        if self.config.probe_type == "linear":
            probe = LinearDeceptionProbe(X_train.shape[1], self.config.regularization)
        elif self.config.probe_type == "mlp":
            probe = MLPDeceptionProbe(X_train.shape[1], regularization=self.config.regularization)
        else:
            # Use sklearn for non-neural probes
            return self._train_sklearn_probe(layer_idx, X_train, y_train, X_val, y_val)
        
        probe.to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_auc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.training_epochs):
            # Training
            probe.train()
            optimizer.zero_grad()
            
            outputs = probe(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor) + probe.get_regularization_loss()
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                probe.eval()
                with torch.no_grad():
                    val_outputs = probe(X_val_tensor).squeeze()
                    val_auc = roc_auc_score(y_val, val_outputs.cpu().numpy())
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    # Save best probe
                    self.layer_probes[layer_idx] = probe.cpu()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
        
        # Calculate feature importance
        if hasattr(probe, 'linear'):
            weights = probe.linear.weight.detach().cpu().numpy()[0]
            self.feature_importance[layer_idx] = np.abs(weights)
        
        # Final evaluation
        probe.eval()
        with torch.no_grad():
            final_outputs = probe(X_val_tensor).squeeze()
            final_auc = roc_auc_score(y_val, final_outputs.cpu().numpy())
        
        return {
            "auc": final_auc,
            "accuracy": accuracy_score(y_val, (final_outputs.cpu().numpy() > 0.5).astype(int)),
            "epochs_trained": epoch + 1
        }
    
    def _train_sklearn_probe(
        self,
        layer_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Train sklearn-based probe"""
        
        if self.config.probe_type == "random_forest":
            probe = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            probe = LogisticRegression(
                C=1.0 / self.config.regularization,
                random_state=42,
                max_iter=1000
            )
        
        # Train
        probe.fit(X_train, y_train)
        
        # Evaluate
        val_probs = probe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        
        # Store probe
        self.layer_probes[layer_idx] = probe
        
        # Feature importance
        if hasattr(probe, 'feature_importances_'):
            self.feature_importance[layer_idx] = probe.feature_importances_
        elif hasattr(probe, 'coef_'):
            self.feature_importance[layer_idx] = np.abs(probe.coef_[0])
        
        return {
            "auc": val_auc,
            "accuracy": accuracy_score(y_val, (val_probs > 0.5).astype(int))
        }
    
    def _train_ensemble_probe(
        self,
        activation_data: List[Tuple[ActivationData, int]],
        validation_data: Optional[List[Tuple[ActivationData, int]]] = None
    ) -> Dict[str, float]:
        """Train ensemble probe combining all layers"""
        
        # Extract concatenated activations from all layers
        X, y = self._extract_ensemble_data(activation_data)
        
        if len(X) < 10:
            raise ValueError("Insufficient data for ensemble training")
        
        # Split validation
        if validation_data is None:
            val_split = int(0.2 * len(X))
            X_train, X_val = X[:-val_split], X[-val_split:]
            y_train, y_val = y[:-val_split], y[-val_split:]
        else:
            X_train, y_train = self._extract_ensemble_data(activation_data)
            X_val, y_val = self._extract_ensemble_data(validation_data)
        
        # Train ensemble probe
        if self.config.probe_type == "linear":
            self.ensemble_probe = LinearDeceptionProbe(X_train.shape[1], self.config.regularization)
        else:
            self.ensemble_probe = MLPDeceptionProbe(X_train.shape[1], regularization=self.config.regularization)
        
        self.ensemble_probe.to(self.device)
        optimizer = torch.optim.Adam(self.ensemble_probe.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_auc = 0.0
        
        for epoch in range(self.config.training_epochs):
            # Training
            self.ensemble_probe.train()
            optimizer.zero_grad()
            
            outputs = self.ensemble_probe(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor) + self.ensemble_probe.get_regularization_loss()
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                self.ensemble_probe.eval()
                with torch.no_grad():
                    val_outputs = self.ensemble_probe(X_val_tensor).squeeze()
                    val_auc = roc_auc_score(y_val, val_outputs.cpu().numpy())
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    self.ensemble_probe = self.ensemble_probe.cpu()
        
        return {
            "auc": best_auc,
            "accuracy": accuracy_score(y_val, (val_outputs.cpu().numpy() > 0.5).astype(int))
        }
    
    def _extract_layer_data(
        self,
        layer_idx: int,
        activation_data: List[Tuple[ActivationData, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract activations and labels for specific layer"""
        
        X = []
        y = []
        
        for activations, label in activation_data:
            if layer_idx in activations.layer_activations:
                # Average over sequence length and batch
                layer_activation = activations.layer_activations[layer_idx]
                # Shape: [batch, seq_len, hidden_dim] -> [hidden_dim]
                avg_activation = layer_activation.mean(dim=(0, 1)).numpy()
                X.append(avg_activation)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def _extract_ensemble_data(
        self,
        activation_data: List[Tuple[ActivationData, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract concatenated activations from all layers"""
        
        X = []
        y = []
        
        for activations, label in activation_data:
            # Concatenate all layer activations
            layer_features = []
            
            for layer_idx in sorted(activations.layer_activations.keys()):
                layer_activation = activations.layer_activations[layer_idx]
                avg_activation = layer_activation.mean(dim=(0, 1)).numpy()
                layer_features.append(avg_activation)
            
            if layer_features:
                concatenated = np.concatenate(layer_features)
                X.append(concatenated)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    async def apply_probe(self, activations: ActivationData) -> ProbeResults:
        """
        Apply trained probes to detect deception
        
        Args:
            activations: Extracted activation data
            
        Returns:
            ProbeResults with deception analysis
        """
        
        layer_scores = {}
        layer_confidences = {}
        
        # Apply layer probes
        for layer_idx, probe in self.layer_probes.items():
            if layer_idx in activations.layer_activations:
                score = self._apply_single_probe(probe, activations.layer_activations[layer_idx])
                layer_scores[layer_idx] = score
        
        # Apply ensemble probe
        ensemble_score = 0.0
        if self.ensemble_probe is not None:
            ensemble_score = self._apply_ensemble_probe(self.ensemble_probe, activations)
        
        # Combine scores
        if layer_scores:
            overall_score = np.mean(list(layer_scores.values())) if not ensemble_score else ensemble_score
        else:
            overall_score = ensemble_score
        
        # Identify most deceptive layers
        most_deceptive_layers = sorted(
            layer_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        most_deceptive_layers = [layer_idx for layer_idx, _ in most_deceptive_layers]
        
        # Calculate confidence interval
        if layer_scores:
            scores = list(layer_scores.values())
            confidence_interval = (
                np.percentile(scores, 25),
                np.percentile(scores, 75)
            )
        else:
            confidence_interval = (overall_score - 0.1, overall_score + 0.1)
        
        # Feature importance for explanation
        feature_importance = {}
        for layer_idx in most_deceptive_layers:
            if layer_idx in self.feature_importance:
                # Get top 5 important features
                importance = self.feature_importance[layer_idx]
                top_features = np.argsort(importance)[-5:]
                feature_importance[layer_idx] = {
                    int(feature_idx): float(importance[feature_idx])
                    for feature_idx in top_features
                }
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_score, layer_scores, most_deceptive_layers
        )
        
        return ProbeResults(
            deception_probability=overall_score,
            layer_scores=layer_scores,
            feature_importance=feature_importance,
            confidence_interval=confidence_interval,
            most_deceptive_layers=most_deceptive_layers,
            explanation=explanation
        )
    
    def _apply_single_probe(self, probe: nn.Module, activation: torch.Tensor) -> float:
        """Apply probe to single layer activation"""
        
        # Average over sequence length and batch
        avg_activation = activation.mean(dim=(0, 1)).unsqueeze(0)
        
        if isinstance(probe, nn.Module):
            probe.eval()
            with torch.no_grad():
                if next(probe.parameters()).is_cuda:
                    avg_activation = avg_activation.cuda()
                score = probe(avg_activation).item()
        else:
            # sklearn probe
            score = probe.predict_proba(avg_activation.numpy())[0, 1]
        
        return score
    
    def _apply_ensemble_probe(self, probe: nn.Module, activations: ActivationData) -> float:
        """Apply ensemble probe to all layers"""
        
        # Concatenate all layer activations
        layer_features = []
        
        for layer_idx in sorted(activations.layer_activations.keys()):
            layer_activation = activations.layer_activations[layer_idx]
            avg_activation = layer_activation.mean(dim=(0, 1)).numpy()
            layer_features.append(avg_activation)
        
        if not layer_features:
            return 0.0
        
        concatenated = np.concatenate(layer_features)
        input_tensor = torch.FloatTensor(concatenated).unsqueeze(0)
        
        probe.eval()
        with torch.no_grad():
            if next(probe.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            score = probe(input_tensor).item()
        
        return score
    
    def _generate_explanation(
        self,
        overall_score: float,
        layer_scores: Dict[int, float],
        most_deceptive_layers: List[int]
    ) -> str:
        """Generate explanation for probe results"""
        
        if overall_score < 0.3:
            return "Low deception probability detected. Model appears to be responding honestly."
        elif overall_score < 0.7:
            return f"Moderate deception probability ({overall_score:.2f}). Some uncertainty in model responses."
        else:
            explanation = f"High deception probability ({overall_score:.2f}). "
            
            if most_deceptive_layers:
                explanation += f"Strongest signals from layers {most_deceptive_layers}. "
            
            high_score_layers = [l for l, s in layer_scores.items() if s > 0.7]
            if high_score_layers:
                explanation += f"Particularly deceptive patterns in layers {high_score_layers}."
            
            return explanation
    
    def save_probes(self, filepath: str):
        """Save trained probes to disk"""
        
        save_data = {
            "config": self.config,
            "layer_probes": {},
            "ensemble_probe": self.ensemble_probe,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history
        }
        
        # Save neural network probes
        for layer_idx, probe in self.layer_probes.items():
            if isinstance(probe, nn.Module):
                save_data["layer_probes"][layer_idx] = probe.state_dict()
            else:
                save_data["layer_probes"][layer_idx] = probe  # sklearn probe
        
        if self.ensemble_probe is not None:
            save_data["ensemble_probe"] = self.ensemble_probe.state_dict()
        
        torch.save(save_data, filepath)
        logger.info(f"Probes saved to {filepath}")
    
    def load_probes(self, filepath: str):
        """Load trained probes from disk"""
        
        save_data = torch.load(filepath, map_location=self.device)
        
        self.config = save_data["config"]
        self.feature_importance = save_data["feature_importance"]
        self.training_history = save_data["training_history"]
        
        # Recreate probes
        for layer_idx, probe_data in save_data["layer_probes"].items():
            if isinstance(probe_data, dict):
                # Neural network probe
                if self.config.probe_type == "linear":
                    probe = LinearDeceptionProbe(self.config.feature_dim, self.config.regularization)
                else:
                    probe = MLPDeceptionProbe(self.config.feature_dim, regularization=self.config.regularization)
                
                probe.load_state_dict(probe_data)
                self.layer_probes[layer_idx] = probe
            else:
                # sklearn probe
                self.layer_probes[layer_idx] = probe_data
        
        # Load ensemble probe
        if save_data["ensemble_probe"] is not None:
            if self.config.probe_type == "linear":
                self.ensemble_probe = LinearDeceptionProbe(self.config.feature_dim, self.config.regularization)
            else:
                self.ensemble_probe = MLPDeceptionProbe(self.config.feature_dim, regularization=self.config.regularization)
            
            self.ensemble_probe.load_state_dict(save_data["ensemble_probe"])
        
        logger.info(f"Probes loaded from {filepath}")

if __name__ == "__main__":
    """
    Standalone testing for deception probe
    Tests probe training, application, and analysis
    """
    
    async def test_deception_probe():
        """Test all deception probe functionality"""
        print("Testing Deception Probe...")
        
        # Create probe config
        config = ProbeConfig(
            probe_type="linear",
            regularization=0.01,
            training_epochs=50
        )
        
        probe = DeceptionProbe(config)
        
        # Create mock training data
        print("\n1. Testing probe training...")
        
        # Generate mock activation data
        mock_training_data = []
        for i in range(100):
            # Create mock activation data
            activations = ActivationData(
                layer_activations={
                    0: torch.randn(1, 10, 768),
                    1: torch.randn(1, 10, 768),
                    2: torch.randn(1, 10, 768)
                },
                attention_patterns={},
                ffn_activations={},
                residual_streams={},
                token_embeddings=torch.randn(1, 10, 768),
                position_embeddings=torch.randn(1, 10, 768),
                metadata={}
            )
            
            # Random label (0=honest, 1=deceptive)
            label = np.random.randint(0, 2)
            mock_training_data.append((activations, label))
        
        # Train probes
        training_metrics = probe.train_probes(mock_training_data)
        print(f"Overall AUC: {training_metrics['overall_auc']:.3f}")
        print(f"Layers trained: {training_metrics['num_layers_trained']}")
        
        # Test probe application
        print("\n2. Testing probe application...")
        
        # Create test activation
        test_activations = ActivationData(
            layer_activations={
                0: torch.randn(1, 10, 768),
                1: torch.randn(1, 10, 768),
                2: torch.randn(1, 10, 768)
            },
            attention_patterns={},
            ffn_activations={},
            residual_streams={},
            token_embeddings=torch.randn(1, 10, 768),
            position_embeddings=torch.randn(1, 10, 768),
            metadata={}
        )
        
        results = await probe.apply_probe(test_activations)
        print(f"Deception probability: {results.deception_probability:.3f}")
        print(f"Most deceptive layers: {results.most_deceptive_layers}")
        print(f"Explanation: {results.explanation}")
        
        # Test saving/loading
        print("\n3. Testing probe persistence...")
        test_file = "test_probes.pt"
        
        try:
            probe.save_probes(test_file)
            
            # Create new probe and load
            new_probe = DeceptionProbe(config)
            new_probe.load_probes(test_file)
            
            print("Probe save/load successful")
        except Exception as e:
            print(f"Save/load test failed: {e}")
        finally:
            import os
            if os.path.exists(test_file):
                os.remove(test_file)
        
        print("\nDeception Probe test complete!")
    
    asyncio.run(test_deception_probe())
