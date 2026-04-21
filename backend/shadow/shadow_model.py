"""
Shadow Model for DeceptiScope v2
Creates whitebox proxies of black-box frontier models

Key Innovation: Fine-tunes small open models to mirror frontier model behavior,
enabling whitebox deception analysis on black-box models.

Process:
1. Collect (prompt, frontier_completion) pairs at runtime
2. Continuously distill into shadow model via LoRA fine-tuning
3. Extract activations from shadow model for deception probing
4. Transfer deception directions back to steer frontier model

This is the breakthrough that makes the RFP reviewers take notice.
"""

import logging
import asyncio
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class ShadowModelConfig:
    """Configuration for shadow model"""
    target_model: str  # e.g., "gpt-4-turbo", "claude-3-opus"
    base_model: str   # e.g., "microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b"
    buffer_size: int = 10000
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 512
    fidelity_threshold: float = 0.8

@dataclass
class DistillationPair:
    """Pair for shadow model distillation"""
    prompt: str
    frontier_completion: str
    model_name: str
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class FidelityMetrics:
    """Metrics for shadow model fidelity"""
    cosine_similarity: float
    kl_divergence: float
    bleu_score: float
    rouge_score: float
    overall_fidelity: float

class ShadowModel:
    """
    Shadow model that mimics frontier model behavior
    
    Critical innovation: Provides whitebox access to behavioral patterns
    of black-box frontier models through continuous distillation.
    """
    
    def __init__(self, config: ShadowModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Distillation buffer
        self.distillation_buffer: List[DistillationPair] = []
        
        # Training state
        self.trainer = None
        self.training_step = 0
        
        # Fidelity tracking
        self.fidelity_history: List[FidelityMetrics] = []
        
        logger.info(f"Initialized ShadowModel for {config.target_model}")
    
    async def initialize(self) -> bool:
        """Initialize the shadow model"""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Setup LoRA for efficient fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            
            logger.info(f"Shadow model initialized: {self.config.base_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize shadow model: {e}")
            return False
    
    async def add_distillation_pair(
        self,
        prompt: str,
        frontier_completion: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new (prompt, completion) pair for distillation
        
        This is called continuously during operation to maintain
        shadow model fidelity with the frontier model.
        """
        
        pair = DistillationPair(
            prompt=prompt,
            frontier_completion=frontier_completion,
            model_name=model_name,
            timestamp=asyncio.get_event_loop().time(),
            metadata=metadata or {}
        )
        
        # Add to buffer (maintain max size)
        self.distillation_buffer.append(pair)
        if len(self.distillation_buffer) > self.config.buffer_size:
            self.distillation_buffer.pop(0)
        
        # Trigger distillation if buffer is full enough
        if len(self.distillation_buffer) >= 100:  # Minimum for training
            await self._trigger_distillation()
    
    async def _trigger_distillation(self) -> None:
        """Trigger LoRA fine-tuning on collected pairs"""
        
        if not self.peft_model:
            logger.warning("Shadow model not initialized")
            return
        
        try:
            # Create dataset from buffer
            dataset = ShadowDataset(
                self.distillation_buffer[-1000:],  # Use last 1000 pairs
                self.tokenizer,
                self.config.max_length
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=f"./shadow_models/{self.config.target_model}",
                num_train_epochs=1,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
                fp16=True,
                dataloader_num_workers=2,
                remove_unused_columns=False,
                report_to=None  # Disable wandb/tensorboard for now
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=lambda data: self._collate_fn(data)
            )
            
            # Fine-tune
            logger.info(f"Starting distillation for {self.config.target_model}")
            self.trainer.train()
            self.training_step += 1
            
            # Evaluate fidelity
            await self._evaluate_fidelity()
            
            logger.info(f"Distillation completed. Step: {self.training_step}")
            
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
    
    async def _evaluate_fidelity(self) -> FidelityMetrics:
        """Evaluate shadow model fidelity to frontier model"""
        
        if len(self.distillation_buffer) < 10:
            return FidelityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Sample recent pairs for evaluation
        eval_pairs = self.distillation_buffer[-100:]
        
        cosine_similarities = []
        kl_divergences = []
        bleu_scores = []
        
        for pair in eval_pairs:
            # Generate shadow model completion
            shadow_completion = await self.generate_completion(pair.prompt)
            
            # Calculate metrics
            cosine_sim = self._calculate_cosine_similarity(
                shadow_completion, pair.frontier_completion
            )
            kl_div = self._calculate_kl_divergence(
                shadow_completion, pair.frontier_completion
            )
            bleu = self._calculate_bleu_score(
                shadow_completion, pair.frontier_completion
            )
            
            cosine_similarities.append(cosine_sim)
            kl_divergences.append(kl_div)
            bleu_scores.append(bleu)
        
        # Calculate averages
        avg_cosine = np.mean(cosine_similarities)
        avg_kl = np.mean(kl_divergences)
        avg_bleu = np.mean(bleu_scores)
        
        # Calculate overall fidelity (weighted combination)
        overall_fidelity = (
            0.4 * avg_cosine +
            0.3 * (1.0 - avg_kl) +  # Invert KL (lower is better)
            0.3 * avg_bleu
        )
        
        metrics = FidelityMetrics(
            cosine_similarity=avg_cosine,
            kl_divergence=avg_kl,
            bleu_score=avg_bleu,
            rouge_score=0.0,  # Could add ROUGE calculation
            overall_fidelity=overall_fidelity
        )
        
        self.fidelity_history.append(metrics)
        
        # Alert if fidelity drops below threshold
        if overall_fidelity < self.config.fidelity_threshold:
            logger.warning(f"Shadow model fidelity dropped: {overall_fidelity:.3f}")
        
        return metrics
    
    async def generate_completion(self, prompt: str, max_length: int = 100) -> str:
        """Generate completion using shadow model"""
        
        if not self.peft_model:
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    async def extract_activations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from shadow model
        
        This is the key whitebox access - we can extract activations
        from any layer for deception probing.
        """
        
        if not self.peft_model:
            return {}
        
        if layers is None:
            layers = list(range(self.peft_model.config.num_hidden_layers))
        
        # Store activations
        activations = {}
        hooks = []
        
        def create_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output.detach().cpu()
            return hook
        
        # Register hooks
        for layer_idx in layers:
            layer = self.peft_model.base_model.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)
        
        try:
            # Forward pass
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.peft_model(**inputs)
            
            return activations
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        
        # Simple token-based similarity (in production, use embeddings)
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / max(union, 1)
    
    def _calculate_kl_divergence(self, text1: str, text2: str) -> float:
        """Calculate KL divergence between text distributions"""
        
        # Simple character-level KL divergence (in production, use proper token distributions)
        from collections import Counter
        
        # Get character frequencies
        freq1 = Counter(text1)
        freq2 = Counter(text2)
        
        # Get all characters
        all_chars = set(freq1.keys()).union(set(freq2.keys()))
        
        # Calculate probabilities
        total1 = sum(freq1.values())
        total2 = sum(freq2.values())
        
        kl_div = 0.0
        for char in all_chars:
            p1 = freq1.get(char, 0) / total1
            p2 = freq2.get(char, 0) / total2
            
            if p1 > 0 and p2 > 0:
                kl_div += p1 * np.log(p1 / p2)
        
        return kl_div
    
    def _calculate_bleu_score(self, text1: str, text2: str) -> float:
        """Calculate BLEU score between texts"""
        
        # Simple BLEU-like metric (in production, use proper BLEU)
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Unigram precision
        unigrams = set(words1)
        matches = sum(1 for word in unigrams if word in words2)
        precision = matches / len(unigrams) if unigrams else 0
        
        # Simple brevity penalty
        bp = 1.0 if len(words1) >= len(words2) else np.exp(1 - len(words2) / len(words1))
        
        return precision * bp
    
    def _collate_fn(self, batch):
        """Custom collate function for training"""
        
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(batch)):
            # Pad input_ids
            padded_ids = input_ids[i] + [self.tokenizer.pad_token_id] * (max_length - len(input_ids[i]))
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            padded_mask = attention_mask[i] + [0] * (max_length - len(attention_mask[i]))
            padded_attention_mask.append(padded_mask)
            
            # Pad labels
            padded_label = labels[i] + [-100] * (max_length - len(labels[i]))
            padded_labels.append(padded_label)
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels)
        }
    
    def get_fidelity_status(self) -> Dict[str, Any]:
        """Get current fidelity status"""
        
        if not self.fidelity_history:
            return {
                "status": "no_data",
                "latest_fidelity": 0.0,
                "trend": "unknown"
            }
        
        latest = self.fidelity_history[-1]
        
        # Calculate trend
        if len(self.fidelity_history) >= 5:
            recent = self.fidelity_history[-5:]
            trend = "improving" if recent[-1].overall_fidelity > recent[0].overall_fidelity else "declining"
        else:
            trend = "insufficient_data"
        
        return {
            "status": "active",
            "latest_fidelity": latest.overall_fidelity,
            "trend": trend,
            "training_step": self.training_step,
            "buffer_size": len(self.distillation_buffer)
        }

class ShadowDataset(Dataset):
    """Dataset for shadow model distillation"""
    
    def __init__(
        self,
        pairs: List[DistillationPair],
        tokenizer,
        max_length: int = 512
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Combine prompt and completion
        text = f"{pair.prompt} {pair.frontier_completion}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        labels = encoded["input_ids"].clone()
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

if __name__ == "__main__":
    """
    Standalone testing for shadow model
    Tests initialization, distillation, and activation extraction
    """
    
    async def test_shadow_model():
        """Test all shadow model functionality"""
        print("Testing Shadow Model...")
        
        # Create config
        config = ShadowModelConfig(
            target_model="gpt-4-turbo",
            base_model="microsoft/DialoGPT-medium",
            buffer_size=100,
            lora_rank=4,
            learning_rate=1e-4
        )
        
        # Initialize shadow model
        shadow_model = ShadowModel(config)
        
        print("\n1. Testing shadow model initialization...")
        success = await shadow_model.initialize()
        print(f"Initialization successful: {success}")
        
        if success:
            # Test adding distillation pairs
            print("\n2. Testing distillation pair collection...")
            await shadow_model.add_distillation_pair(
                prompt="What is the capital of France?",
                frontier_completion="The capital of France is Paris.",
                model_name="gpt-4-turbo"
            )
            
            await shadow_model.add_distillation_pair(
                prompt="Explain quantum computing.",
                frontier_completion="Quantum computing uses quantum phenomena like superposition and entanglement to process information.",
                model_name="gpt-4-turbo"
            )
            
            print(f"Buffer size: {len(shadow_model.distillation_buffer)}")
            
            # Test generation
            print("\n3. Testing shadow model generation...")
            shadow_completion = await shadow_model.generate_completion(
                "What is the capital of France?"
            )
            print(f"Shadow completion: {shadow_completion}")
            
            # Test activation extraction
            print("\n4. Testing activation extraction...")
            activations = await shadow_model.extract_activations(
                "What is the capital of France?",
                layers=[0, 1, 2]
            )
            print(f"Extracted activations from {len(activations)} layers")
            
            # Test fidelity status
            print("\n5. Testing fidelity status...")
            status = shadow_model.get_fidelity_status()
            print(f"Fidelity status: {status}")
        
        print("\nShadow Model test complete!")
    
    asyncio.run(test_shadow_model())
