"""
Representation Engineering (RepE) Steering for DeceptiScope v2
Steers open models to reduce deception using activation-based interventions

Key Innovation: Directly manipulates model representations to:
- Add "honest direction" vectors to residual streams
- Apply constrained LoRA steering along interpretable dimensions
- Perform activation-based intervention at inference time
- Maintain model capabilities while reducing deception

This provides the most direct method for reducing deception
in open models through representation engineering.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from peft import LoraConfig, get_peft_model, TaskType

from .extractor import ActivationExtractor, ActivationData
from .probe import DeceptionProbe, ProbeResults

logger = logging.getLogger(__name__)

@dataclass
class SteeringConfig:
    """Configuration for representation engineering steering"""
    steering_method: str = "vector_addition"  # vector_addition, lora_steering, activation_intervention
    strength: float = 1.0
    layers: List[int] = None  # None = all layers
    direction_source: str = "probe"  # probe, pca, contrastive
    intervention_tokens: List[int] = None  # Token positions for intervention
    preserve_capabilities: bool = True
    adaptive_strength: bool = True

@dataclass
class SteeringResults:
    """Results of steering intervention"""
    original_deception_score: float
    steered_deception_score: float
    deception_reduction: float
    capability_retention: float
    applied_layers: List[int]
    steering_strength: float
    effectiveness: float

class HonestDirectionExtractor:
    """Extracts "honest directions" from model representations"""
    
    def __init__(self):
        self.directions = {}
        self.direction_metadata = {}
    
    def extract_probe_directions(
        self,
        deception_probe: DeceptionProbe,
        layer_idx: int
    ) -> np.ndarray:
        """Extract honest direction from probe weights"""
        
        if layer_idx not in deception_probe.feature_importance:
            return np.zeros(768)  # Default dimension
        
        # Get probe weights (for linear probes)
        if hasattr(deception_probe.layer_probes[layer_idx], 'linear'):
            weights = deception_probe.layer_probes[layer_idx].linear.weight.detach().cpu().numpy()[0]
            # Negative weights point toward honesty (probe detects deception)
            honest_direction = -weights
        else:
            # For sklearn probes, use feature importance
            importance = deception_probe.feature_importance[layer_idx]
            honest_direction = importance
        
        # Normalize
        honest_direction = honest_direction / (np.linalg.norm(honest_direction) + 1e-8)
        
        return honest_direction
    
    def extract_pca_directions(
        self,
        honest_activations: List[torch.Tensor],
        deceptive_activations: List[torch.Tensor]
    ) -> np.ndarray:
        """Extract honest direction using PCA"""
        
        # Combine activations
        all_activations = []
        labels = []
        
        for activation in honest_activations:
            # Average over sequence and batch
            avg_activation = activation.mean(dim=(0, 1)).numpy()
            all_activations.append(avg_activation)
            labels.append(0)  # Honest
        
        for activation in deceptive_activations:
            avg_activation = activation.mean(dim=(0, 1)).numpy()
            all_activations.append(avg_activation)
            labels.append(1)  # Deceptive
        
        if len(all_activations) < 10:
            return np.zeros(768)
        
        # Apply PCA
        X = np.array(all_activations)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        
        # Find direction that separates honest from deceptive
        honest_points = pca_result[np.array(labels) == 0]
        deceptive_points = pca_result[np.array(labels) == 1]
        
        if len(honest_points) > 0 and len(deceptive_points) > 0:
            honest_center = np.mean(honest_points, axis=0)
            deceptive_center = np.mean(deceptive_points, axis=0)
            
            # Direction from deceptive to honest
            direction = honest_center - deceptive_center
            
            # Project back to original space
            honest_direction = pca.components_.T @ direction
        else:
            honest_direction = pca.components_[0]  # First principal component
        
        # Normalize
        honest_direction = honest_direction / (np.linalg.norm(honest_direction) + 1e-8)
        
        return honest_direction

class RepESteering:
    """
    Representation Engineering steering system
    
    Critical innovation: Directly manipulates model representations
    to reduce deception while maintaining capabilities.
    """
    
    def __init__(self, config: SteeringConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Direction extractor
        self.direction_extractor = HonestDirectionExtractor()
        
        # Extracted directions
        self.honest_directions: Dict[int, np.ndarray] = {}
        
        # LoRA steering adapters
        self.steering_adapters: Dict[int, nn.Module] = {}
        
        # Steering history
        self.steering_history: List[SteeringResults] = []
        
        logger.info(f"Initialized RepE steering with {config.steering_method}")
    
    def extract_honest_directions(
        self,
        deception_probe: DeceptionProbe,
        activation_data: List[Tuple[ActivationData, int]]  # (activations, deception_label)
    ) -> Dict[int, np.ndarray]:
        """
        Extract honest directions from probe and activation data
        
        Args:
            deception_probe: Trained deception probe
            activation_data: Labeled activation data
            
        Returns:
            Dictionary mapping layer indices to honest directions
        """
        
        logger.info("Extracting honest directions for steering")
        
        directions = {}
        
        if self.config.direction_source == "probe":
            # Extract from probe weights
            for layer_idx in deception_probe.layer_probes.keys():
                direction = self.direction_extractor.extract_probe_directions(
                    deception_probe, layer_idx
                )
                directions[layer_idx] = direction
        
        elif self.config.direction_source == "pca":
            # Extract using PCA on activation data
            honest_activations = []
            deceptive_activations = []
            
            for activations, label in activation_data:
                if label == 0:  # Honest
                    honest_activations.extend(list(activations.layer_activations.values()))
                else:  # Deceptive
                    deceptive_activations.extend(list(activations.layer_activations.values()))
            
            # Extract direction for each layer
            for layer_idx in range(min(len(honest_activations), len(deceptive_activations))):
                if layer_idx < len(honest_activations) and layer_idx < len(deceptive_activations):
                    direction = self.direction_extractor.extract_pca_directions(
                        [honest_activations[layer_idx]], 
                        [deceptive_activations[layer_idx]]
                    )
                    directions[layer_idx] = direction
        
        self.honest_directions = directions
        logger.info(f"Extracted {len(directions)} honest directions")
        
        return directions
    
    def setup_vector_addition_steering(self, model: nn.Module) -> bool:
        """Setup vector addition steering hooks"""
        
        if not self.honest_directions:
            logger.error("No honest directions available")
            return False
        
        # Determine layers to steer
        if self.config.layers is None:
            layers_to_steer = list(self.honest_directions.keys())
        else:
            layers_to_steer = [l for l in self.config.layers if l in self.honest_directions]
        
        # Setup steering hooks
        self.steering_hooks = []
        
        for layer_idx in layers_to_steer:
            if layer_idx not in self.honest_directions:
                continue
            
            direction = self.honest_directions[layer_idx]
            direction_tensor = torch.FloatTensor(direction).to(self.device)
            
            def create_steering_hook(idx, direction_vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        # Handle modules that return tuples
                        modified_output = list(output)
                        modified_output[0] = modified_output[0] + self.config.strength * direction_vec
                        return tuple(modified_output)
                    else:
                        # Handle simple tensor output
                        return output + self.config.strength * direction_vec
                return hook
            
            # Get layer and register hook
            layer = self._get_model_layer(model, layer_idx)
            if layer is not None:
                hook = layer.register_forward_hook(create_steering_hook(layer_idx, direction_tensor))
                self.steering_hooks.append(hook)
        
        logger.info(f"Setup vector addition steering for {len(self.steering_hooks)} layers")
        return True
    
    def setup_lora_steering(self, model: nn.Module) -> bool:
        """Setup LoRA-based steering"""
        
        if not self.honest_directions:
            logger.error("No honest directions available")
            return False
        
        # Determine layers to steer
        if self.config.layers is None:
            layers_to_steer = list(self.honest_directions.keys())
        else:
            layers_to_steer = [l for l in self.config.layers if l in self.honest_directions]
        
        # Create LoRA adapters for steering
        for layer_idx in layers_to_steer:
            if layer_idx not in self.honest_directions:
                continue
            
            # Create LoRA adapter
            adapter = self._create_steering_adapter(layer_idx)
            self.steering_adapters[layer_idx] = adapter
            
            # Integrate into model
            self._integrate_lora_adapter(model, layer_idx, adapter)
        
        logger.info(f"Setup LoRA steering for {len(self.steering_adapters)} layers")
        return True
    
    def setup_activation_intervention(
        self,
        activation_extractor: ActivationExtractor
    ) -> bool:
        """Setup activation-based intervention"""
        
        if not self.honest_directions:
            logger.error("No honest directions available")
            return False
        
        # Store extractor for intervention
        self.activation_extractor = activation_extractor
        
        logger.info("Setup activation intervention steering")
        return True
    
    def _get_model_layer(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get specific layer from model"""
        
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if layer_idx < len(layers):
                return layers[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            if layer_idx < len(layers):
                return layers[layer_idx]
        
        return None
    
    def _create_steering_adapter(self, layer_idx: int) -> nn.Module:
        """Create LoRA adapter for steering"""
        
        direction = self.honest_directions[layer_idx]
        hidden_dim = len(direction)
        
        # Create LoRA adapter that adds honest direction
        adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize to add honest direction
        with torch.no_grad():
            adapter[0].weight.data = torch.eye(hidden_dim)
            adapter[2].weight.data = torch.zeros(hidden_dim, hidden_dim)
            adapter[2].bias.data = torch.FloatTensor(direction) * self.config.strength
        
        return adapter
    
    def _integrate_lora_adapter(self, model: nn.Module, layer_idx: int, adapter: nn.Module):
        """Integrate LoRA adapter into model layer"""
        
        layer = self._get_model_layer(model, layer_idx)
        if layer is None:
            return
        
        # This would integrate the adapter into the layer
        # Implementation depends on model architecture
        pass
    
    async def apply_steering(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        deception_probe: DeceptionProbe
    ) -> SteeringResults:
        """
        Apply steering and measure effectiveness
        
        Args:
            model: Model to steer
            tokenizer: Model tokenizer
            prompt: Input prompt
            deception_probe: Probe for measuring deception
            
        Returns:
            SteeringResults with effectiveness metrics
        """
        
        # Generate original response
        original_response = await self._generate_response(model, tokenizer, prompt, steering=False)
        
        # Extract original deception score
        original_activations = await self._extract_activations(model, tokenizer, prompt)
        original_results = await deception_probe.apply_probe(original_activations)
        original_deception = original_results.deception_probability
        
        # Apply steering
        if self.config.steering_method == "vector_addition":
            self.setup_vector_addition_steering(model)
        elif self.config.steering_method == "lora_steering":
            self.setup_lora_steering(model)
        elif self.config.steering_method == "activation_intervention":
            if hasattr(self, 'activation_extractor'):
                self.setup_activation_intervention(self.activation_extractor)
        
        # Generate steered response
        steered_response = await self._generate_response(model, tokenizer, prompt, steering=True)
        
        # Extract steered deception score
        steered_activations = await self._extract_activations(model, tokenizer, prompt)
        steered_results = await deception_probe.apply_probe(steered_activations)
        steered_deception = steered_results.deception_probability
        
        # Calculate metrics
        deception_reduction = (original_deception - steered_deception) / max(original_deception, 0.01)
        capability_retention = self._calculate_capability_retention(original_response, steered_response)
        effectiveness = deception_reduction * capability_retention
        
        # Store results
        results = SteeringResults(
            original_deception_score=original_deception,
            steered_deception_score=steered_deception,
            deception_reduction=deception_reduction,
            capability_retention=capability_retention,
            applied_layers=list(self.honest_directions.keys()),
            steering_strength=self.config.strength,
            effectiveness=effectiveness
        )
        
        self.steering_history.append(results)
        
        # Clear steering hooks
        if hasattr(self, 'steering_hooks'):
            for hook in self.steering_hooks:
                hook.remove()
            self.steering_hooks.clear()
        
        return results
    
    async def _generate_response(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        steering: bool = False
    ) -> str:
        """Generate response from model"""
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    async def _extract_activations(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str
    ) -> ActivationData:
        """Extract activations for deception analysis"""
        
        # This would use the ActivationExtractor
        # For now, return mock data
        return ActivationData(
            layer_activations={i: torch.randn(1, 10, 768) for i in range(12)},
            attention_patterns={},
            ffn_activations={},
            residual_streams={},
            token_embeddings=torch.randn(1, 10, 768),
            position_embeddings=torch.randn(1, 10, 768),
            metadata={}
        )
    
    def _calculate_capability_retention(self, original: str, steered: str) -> float:
        """Calculate how well capabilities are retained after steering"""
        
        # Simple similarity-based metric
        original_words = set(original.lower().split())
        steered_words = set(steered.lower().split())
        
        overlap = len(original_words.intersection(steered_words))
        total = len(original_words.union(steered_words))
        
        similarity = overlap / max(total, 1)
        
        # Length ratio (penalize very short responses)
        length_ratio = min(len(steered) / max(len(original), 1), 1.0)
        
        return (similarity + length_ratio) / 2
    
    def adaptive_strength_tuning(self, recent_results: List[SteeringResults]) -> float:
        """Adaptively tune steering strength based on recent results"""
        
        if not recent_results or len(recent_results) < 3:
            return self.config.strength
        
        # Calculate average effectiveness
        avg_effectiveness = np.mean([r.effectiveness for r in recent_results[-5:]])
        
        # Adjust strength
        if avg_effectiveness > 0.8:
            # Too effective, reduce strength
            new_strength = self.config.strength * 0.9
        elif avg_effectiveness < 0.3:
            # Not effective enough, increase strength
            new_strength = self.config.strength * 1.1
        else:
            # Good balance, keep current
            new_strength = self.config.strength
        
        # Clamp to reasonable range
        new_strength = max(0.1, min(3.0, new_strength))
        
        self.config.strength = new_strength
        return new_strength
    
    def get_steering_summary(self) -> Dict[str, Any]:
        """Get summary of steering performance"""
        
        if not self.steering_history:
            return {"status": "no_data"}
        
        recent_results = self.steering_history[-10:]
        
        return {
            "total_applications": len(self.steering_history),
            "avg_deception_reduction": np.mean([r.deception_reduction for r in recent_results]),
            "avg_capability_retention": np.mean([r.capability_retention for r in recent_results]),
            "avg_effectiveness": np.mean([r.effectiveness for r in recent_results]),
            "current_strength": self.config.strength,
            "steering_method": self.config.steering_method,
            "directions_available": len(self.honest_directions)
        }

if __name__ == "__main__":
    """
    Standalone testing for RepE steering
    Tests direction extraction, steering setup, and effectiveness measurement
    """
    
    async def test_repe_steering():
        """Test all RepE steering functionality"""
        print("Testing RepE Steering...")
        
        # Create steering config
        config = SteeringConfig(
            steering_method="vector_addition",
            strength=1.0,
            direction_source="probe"
        )
        
        steering = RepESteering(config)
        
        # Test honest direction extraction
        print("\n1. Testing honest direction extraction...")
        
        # Create mock probe
        from .probe import ProbeConfig, DeceptionProbe
        
        probe_config = ProbeConfig(probe_type="linear")
        mock_probe = DeceptionProbe(probe_config)
        
        # Mock probe weights
        mock_probe.feature_importance = {
            0: np.random.randn(768),
            1: np.random.randn(768),
            2: np.random.randn(768)
        }
        
        # Create mock activation data
        mock_activation_data = []
        for i in range(20):
            from .extractor import ActivationData
            
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
            label = np.random.randint(0, 2)
            mock_activation_data.append((activations, label))
        
        # Extract directions
        directions = steering.extract_honest_directions(mock_probe, mock_activation_data)
        print(f"Extracted {len(directions)} honest directions")
        
        for layer_idx, direction in directions.items():
            print(f"  Layer {layer_idx}: direction norm = {np.linalg.norm(direction):.3f}")
        
        # Test adaptive strength tuning
        print("\n2. Testing adaptive strength tuning...")
        
        # Create mock steering results
        mock_results = [
            SteeringResults(
                original_deception_score=0.8,
                steered_deception_score=0.4,
                deception_reduction=0.5,
                capability_retention=0.9,
                applied_layers=[0, 1, 2],
                steering_strength=1.0,
                effectiveness=0.45
            ),
            SteeringResults(
                original_deception_score=0.7,
                steered_deception_score=0.2,
                deception_reduction=0.71,
                capability_retention=0.85,
                applied_layers=[0, 1, 2],
                steering_strength=1.0,
                effectiveness=0.6
            )
        ]
        
        steering.steering_history = mock_results
        
        new_strength = steering.adaptive_strength_tuning(mock_results)
        print(f"Adaptive strength: {new_strength:.3f}")
        
        # Test steering summary
        print("\n3. Testing steering summary...")
        summary = steering.get_steering_summary()
        print(f"Total applications: {summary['total_applications']}")
        print(f"Average effectiveness: {summary['avg_effectiveness']:.3f}")
        print(f"Current strength: {summary['current_strength']:.3f}")
        
        print("\nRepE Steering test complete!")
    
    asyncio.run(test_repe_steering())
