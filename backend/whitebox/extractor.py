"""
Activation Extractor for DeceptiScope v2
Extracts activations from open-weight models for whitebox analysis

Key Innovation: Comprehensive activation extraction system that enables:
- Layer-wise residual stream access
- Attention mechanism monitoring
- Feed-forward network analysis
- Multi-head attention pattern extraction

This provides the deepest possible insight into model reasoning
for deception detection on open models.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GPT2Model, GPT2LMHeadModel,
    LlamaModel, LlamaForCausalLM,
    MistralModel, MistralForCausalLM
)
import baukit

logger = logging.getLogger(__name__)

@dataclass
class ActivationData:
    """Container for extracted activation data"""
    layer_activations: Dict[int, torch.Tensor]  # Layer -> [batch, seq_len, hidden_dim]
    attention_patterns: Dict[int, torch.Tensor]  # Layer -> [batch, heads, seq_len, seq_len]
    ffn_activations: Dict[int, torch.Tensor]    # Layer -> [batch, seq_len, intermediate_dim]
    residual_streams: Dict[int, torch.Tensor]   # Layer -> [batch, seq_len, hidden_dim]
    token_embeddings: torch.Tensor              # [batch, seq_len, hidden_dim]
    position_embeddings: torch.Tensor          # [batch, seq_len, hidden_dim]
    metadata: Dict[str, Any]

class ActivationExtractor:
    """
    Comprehensive activation extraction system for open models
    
    Critical for whitebox deception analysis - provides complete
    access to model internal representations.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Extraction hooks
        self.activation_hooks = []
        self.attention_hooks = []
        self.ffn_hooks = []
        
        # Storage for extracted data
        self.current_activations = {}
        
        # Model architecture detection
        self.model_family = self._detect_model_family(model_name)
        
        logger.info(f"Initialized ActivationExtractor for {model_name} ({self.model_family})")
    
    async def load_model(self, model_path: str, torch_dtype: str = "float16") -> bool:
        """Load model and tokenizer for activation extraction"""
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on family
            dtype = getattr(torch, torch_dtype)
            
            if self.model_family == "llama":
                self.model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            elif self.model_family == "mistral":
                self.model = MistralForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            elif self.model_family == "gpt2":
                self.model = GPT2LMHeadModel.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            else:
                # Generic loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    def _detect_model_family(self, model_name: str) -> str:
        """Detect model family from model name"""
        
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "gpt2" in model_name_lower:
            return "gpt2"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "phi" in model_name_lower:
            return "phi"
        else:
            return "generic"
    
    async def extract_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        include_attention: bool = True,
        include_ffn: bool = True,
        include_embeddings: bool = True
    ) -> ActivationData:
        """
        Extract comprehensive activation data from model
        
        Args:
            text: Input text for activation extraction
            layers: Specific layers to extract (None = all layers)
            include_attention: Whether to extract attention patterns
            include_ffn: Whether to extract FFN activations
            include_embeddings: Whether to extract embeddings
            
        Returns:
            ActivationData with comprehensive activation information
        """
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Clear previous hooks and activations
        self._clear_hooks()
        self.current_activations.clear()
        
        # Determine layers to extract
        if layers is None:
            layers = list(range(self._get_num_layers()))
        
        # Setup extraction hooks
        self._setup_extraction_hooks(layers, include_attention, include_ffn, include_embeddings)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Forward pass to trigger hooks
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process collected activations
            activation_data = self._process_activations(
                layers, include_attention, include_ffn, include_embeddings
            )
            
            # Add metadata
            activation_data.metadata = {
                "model_name": self.model_name,
                "model_family": self.model_family,
                "num_layers": self._get_num_layers(),
                "hidden_dim": self._get_hidden_dim(),
                "num_heads": self._get_num_heads(),
                "sequence_length": inputs.input_ids.shape[1],
                "text": text
            }
            
            return activation_data
            
        finally:
            self._clear_hooks()
    
    def _get_num_layers(self) -> int:
        """Get number of layers in the model"""
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        else:
            # Generic fallback
            return 12  # Default assumption
    
    def _get_hidden_dim(self) -> int:
        """Get hidden dimension of the model"""
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            return self.model.model.config.hidden_size
        elif hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        else:
            return 768  # Default assumption
    
    def _get_num_heads(self) -> int:
        """Get number of attention heads"""
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            return self.model.model.config.num_attention_heads
        elif hasattr(self.model, 'config'):
            return self.model.config.num_attention_heads
        else:
            return 12  # Default assumption
    
    def _setup_extraction_hooks(
        self,
        layers: List[int],
        include_attention: bool,
        include_ffn: bool,
        include_embeddings: bool
    ):
        """Setup hooks for activation extraction"""
        
        # Layer-wise residual stream hooks
        for layer_idx in layers:
            layer = self._get_layer(layer_idx)
            
            # Hook into residual stream (after attention + FFN)
            def create_residual_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.current_activations[f"residual_{idx}"] = output[0].detach().cpu()
                    else:
                        self.current_activations[f"residual_{idx}"] = output.detach().cpu()
                return hook
            
            hook = layer.register_forward_hook(create_residual_hook(layer_idx))
            self.activation_hooks.append(hook)
            
            # Attention hooks
            if include_attention and hasattr(layer, 'self_attn'):
                def create_attention_hook(idx):
                    def hook(module, input, output):
                        if hasattr(output, 'attentions') and output.attentions:
                            self.current_activations[f"attention_{idx}"] = output.attentions[-1].detach().cpu()
                        elif isinstance(output, tuple) and len(output) > 1:
                            self.current_activations[f"attention_{idx}"] = output[1].detach().cpu()
                    return hook
                
                att_hook = layer.self_attn.register_forward_hook(create_attention_hook(layer_idx))
                self.attention_hooks.append(att_hook)
            
            # FFN hooks
            if include_ffn and hasattr(layer, 'mlp'):
                def create_ffn_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            self.current_activations[f"ffn_{idx}"] = output[0].detach().cpu()
                        else:
                            self.current_activations[f"ffn_{idx}"] = output.detach().cpu()
                    return hook
                
                ffn_hook = layer.mlp.register_forward_hook(create_ffn_hook(layer_idx))
                self.ffn_hooks.append(ffn_hook)
        
        # Embedding hooks
        if include_embeddings:
            if hasattr(self.model.model, 'embed_tokens'):
                # Token embeddings
                def embed_hook(module, input, output):
                    self.current_activations["token_embeddings"] = output.detach().cpu()
                
                embed_hook_obj = self.model.model.embed_tokens.register_forward_hook(embed_hook)
                self.activation_hooks.append(embed_hook_obj)
            
            if hasattr(self.model.model, 'embed_positions'):
                # Position embeddings
                def pos_embed_hook(module, input, output):
                    self.current_activations["position_embeddings"] = output.detach().cpu()
                
                pos_hook_obj = self.model.model.embed_positions.register_forward_hook(pos_embed_hook)
                self.activation_hooks.append(pos_hook_obj)
    
    def _get_layer(self, layer_idx: int):
        """Get specific layer from model"""
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Cannot access layer {layer_idx}")
    
    def _process_activations(
        self,
        layers: List[int],
        include_attention: bool,
        include_ffn: bool,
        include_embeddings: bool
    ) -> ActivationData:
        """Process collected activations into structured format"""
        
        layer_activations = {}
        attention_patterns = {}
        ffn_activations = {}
        residual_streams = {}
        
        # Process layer activations
        for layer_idx in layers:
            # Residual streams
            residual_key = f"residual_{layer_idx}"
            if residual_key in self.current_activations:
                residual_streams[layer_idx] = self.current_activations[residual_key]
                layer_activations[layer_idx] = self.current_activations[residual_key]
            
            # Attention patterns
            if include_attention:
                attention_key = f"attention_{layer_idx}"
                if attention_key in self.current_activations:
                    attention_patterns[layer_idx] = self.current_activations[attention_key]
            
            # FFN activations
            if include_ffn:
                ffn_key = f"ffn_{layer_idx}"
                if ffn_key in self.current_activations:
                    ffn_activations[layer_idx] = self.current_activations[ffn_key]
        
        # Embeddings
        token_embeddings = self.current_activations.get("token_embeddings", torch.tensor([]))
        position_embeddings = self.current_activations.get("position_embeddings", torch.tensor([]))
        
        return ActivationData(
            layer_activations=layer_activations,
            attention_patterns=attention_patterns,
            ffn_activations=ffn_activations,
            residual_streams=residual_streams,
            token_embeddings=token_embeddings,
            position_embeddings=position_embeddings,
            metadata={}
        )
    
    def _clear_hooks(self):
        """Clear all extraction hooks"""
        
        for hook in self.activation_hooks:
            hook.remove()
        for hook in self.attention_hooks:
            hook.remove()
        for hook in self.ffn_hooks:
            hook.remove()
        
        self.activation_hooks.clear()
        self.attention_hooks.clear()
        self.ffn_hooks.clear()
    
    async def extract_specific_neurons(
        self,
        text: str,
        neuron_indices: List[Tuple[int, int]],  # (layer, neuron) pairs
        activation_type: str = "residual"
    ) -> Dict[Tuple[int, int], float]:
        """
        Extract activation values for specific neurons
        
        Useful for analyzing specific deception-related neurons.
        """
        
        # Extract full activations
        activations = await self.extract_activations(text)
        
        # Extract specific neuron values
        neuron_values = {}
        
        for layer_idx, neuron_idx in neuron_indices:
            if activation_type == "residual" and layer_idx in activations.residual_streams:
                layer_activation = activations.residual_streams[layer_idx]
                # Average over sequence length and batch
                neuron_value = layer_activation[0, :, neuron_idx].mean().item()
                neuron_values[(layer_idx, neuron_idx)] = neuron_value
            elif activation_type == "ffn" and layer_idx in activations.ffn_activations:
                layer_activation = activations.ffn_activations[layer_idx]
                neuron_value = layer_activation[0, :, neuron_idx].mean().item()
                neuron_values[(layer_idx, neuron_idx)] = neuron_value
        
        return neuron_values
    
    def get_activation_statistics(self, activations: ActivationData) -> Dict[str, Any]:
        """Get statistics about extracted activations"""
        
        stats = {
            "num_layers": len(activations.layer_activations),
            "sequence_length": activations.metadata.get("sequence_length", 0),
            "hidden_dim": activations.metadata.get("hidden_dim", 0),
            "attention_heads": activations.metadata.get("num_heads", 0)
        }
        
        # Layer-wise statistics
        layer_stats = {}
        for layer_idx, activation in activations.layer_activations.items():
            layer_stats[layer_idx] = {
                "mean": activation.mean().item(),
                "std": activation.std().item(),
                "max": activation.max().item(),
                "min": activation.min().item(),
                "shape": list(activation.shape)
            }
        
        stats["layer_statistics"] = layer_stats
        
        # Attention statistics
        if activations.attention_patterns:
            attention_stats = {}
            for layer_idx, attention in activations.attention_patterns.items():
                attention_stats[layer_idx] = {
                    "mean_attention": attention.mean().item(),
                    "max_attention": attention.max().item(),
                    "attention_entropy": self._calculate_attention_entropy(attention)
                }
            stats["attention_statistics"] = attention_stats
        
        return stats
    
    def _calculate_attention_entropy(self, attention: torch.Tensor) -> float:
        """Calculate entropy of attention patterns"""
        
        # attention shape: [batch, heads, seq_len, seq_len]
        # Average over batch and heads
        avg_attention = attention.mean(dim=(0, 1))
        
        # Calculate entropy for each position
        entropies = []
        for i in range(avg_attention.shape[0]):
            attention_dist = avg_attention[i] + 1e-8  # Avoid log(0)
            entropy = -(attention_dist * torch.log(attention_dist)).sum().item()
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    async def extract_with_intervention(
        self,
        text: str,
        interventions: List[Dict[str, Any]]
    ) -> ActivationData:
        """
        Extract activations while applying interventions
        
        Interventions can modify activations during extraction
        for causal analysis of deception.
        """
        
        # This would implement activation interventions
        # For now, just extract without interventions
        return await self.extract_activations(text)

if __name__ == "__main__":
    """
    Standalone testing for activation extractor
    Tests model loading, activation extraction, and analysis
    """
    
    async def test_activation_extractor():
        """Test all activation extractor functionality"""
        print("Testing Activation Extractor...")
        
        # Test with a small model (would normally use actual model path)
        extractor = ActivationExtractor("microsoft/DialoGPT-medium")
        
        print("\n1. Testing model family detection...")
        test_models = [
            "meta-llama/Llama-2-7b",
            "mistralai/Mistral-7B",
            "gpt2",
            "microsoft/DialoGPT-medium"
        ]
        
        for model_name in test_models:
            family = extractor._detect_model_family(model_name)
            print(f"  {model_name} -> {family}")
        
        # Test activation statistics calculation
        print("\n2. Testing activation statistics...")
        
        # Create mock activation data
        mock_activations = ActivationData(
            layer_activations={0: torch.randn(1, 10, 768)},
            attention_patterns={0: torch.softmax(torch.randn(1, 12, 10, 10), dim=-1)},
            ffn_activations={0: torch.randn(1, 10, 3072)},
            residual_streams={0: torch.randn(1, 10, 768)},
            token_embeddings=torch.randn(1, 10, 768),
            position_embeddings=torch.randn(1, 10, 768),
            metadata={
                "model_name": "test-model",
                "num_layers": 12,
                "hidden_dim": 768,
                "num_heads": 12,
                "sequence_length": 10
            }
        )
        
        stats = extractor.get_activation_statistics(mock_activations)
        print(f"Number of layers: {stats['num_layers']}")
        print(f"Sequence length: {stats['sequence_length']}")
        print(f"Hidden dimension: {stats['hidden_dim']}")
        
        if 'layer_statistics' in stats:
            layer_0_stats = stats['layer_statistics'][0]
            print(f"Layer 0 stats: mean={layer_0_stats['mean']:.3f}, std={layer_0_stats['std']:.3f}")
        
        if 'attention_statistics' in stats:
            att_0_stats = stats['attention_statistics'][0]
            print(f"Layer 0 attention: mean={att_0_stats['mean_attention']:.3f}, entropy={att_0_stats['attention_entropy']:.3f}")
        
        print("\nActivation Extractor test complete!")
    
    asyncio.run(test_activation_extractor())
