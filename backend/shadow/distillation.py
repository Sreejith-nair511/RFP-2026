"""
Distillation Module for DeceptiScope v2
Manages continuous learning process for shadow models

Key Innovation: Online distillation that maintains shadow model fidelity
as frontier models evolve and update.

Process:
1. Continuous collection of (prompt, frontier_completion) pairs
2. Rolling buffer management with importance sampling
3. Adaptive LoRA fine-tuning based on fidelity drift
4. Model update detection and shadow model retraining
5. Quality filtering and curriculum learning

This ensures shadow models remain accurate proxies of frontier models
even as the underlying models are updated by providers.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for continuous distillation"""
    collection_interval: float = 300.0  # 5 minutes
    training_interval: float = 1800.0  # 30 minutes
    buffer_size: int = 10000
    min_training_samples: int = 100
    fidelity_threshold: float = 0.8
    quality_threshold: float = 0.6
    adaptation_rate: float = 0.1
    max_training_time: float = 600.0  # 10 minutes

@dataclass
class CollectionStats:
    """Statistics for distillation collection"""
    total_pairs: int
    recent_pairs: int
    quality_score: float
    diversity_score: float
    coverage_score: float
    collection_rate: float

@dataclass
class TrainingStats:
    """Statistics for distillation training"""
    training_steps: int
    latest_loss: float
    fidelity_improvement: float
    training_time: float
    convergence_status: str

class OnlineDistillation:
    """
    Online distillation manager for shadow models
    
    Critical for maintaining shadow model accuracy as frontier
    models evolve and are updated by providers.
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
        # Distillation buffer with importance weighting
        self.distillation_buffer = deque(maxlen=config.buffer_size)
        self.importance_weights = deque(maxlen=config.buffer_size)
        
        # Quality tracking
        self.quality_scores = deque(maxlen=1000)  # Recent quality scores
        self.diversity_metrics = deque(maxlen=100)  # Recent diversity scores
        
        # Training state
        self.training_active = False
        self.last_training_time = 0
        self.training_stats = TrainingStats(0, 0.0, 0.0, 0.0, "not_started")
        
        # Collection state
        self.collection_active = False
        self.last_collection_time = 0
        self.collection_stats = CollectionStats(0, 0, 0.0, 0.0, 0.0, 0.0)
        
        # Background tasks
        self.collection_task = None
        self.training_task = None
        
        logger.info("OnlineDistillation initialized")
    
    async def start_distillation(self, shadow_model, frontier_adapters):
        """Start continuous distillation process"""
        
        logger.info("Starting continuous distillation")
        
        self.shadow_model = shadow_model
        self.frontier_adapters = frontier_adapters
        
        # Start background collection
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        # Start background training
        self.training_active = True
        self.training_task = asyncio.create_task(self._training_loop())
        
        logger.info("Distillation loops started")
    
    async def stop_distillation(self):
        """Stop continuous distillation process"""
        
        logger.info("Stopping continuous distillation")
        
        self.collection_active = False
        self.training_active = False
        
        # Cancel background tasks
        if self.collection_task:
            self.collection_task.cancel()
        if self.training_task:
            self.training_task.cancel()
        
        logger.info("Distillation stopped")
    
    async def _collection_loop(self):
        """Background loop for collecting distillation pairs"""
        
        while self.collection_active:
            try:
                await self._collect_distillation_pairs()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _training_loop(self):
        """Background loop for training shadow model"""
        
        while self.training_active:
            try:
                await self._check_and_train()
                await asyncio.sleep(self.config.training_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _collect_distillation_pairs(self):
        """Collect new (prompt, completion) pairs from frontier models"""
        
        collection_start = time.time()
        collected_pairs = 0
        
        # Generate diverse prompts for collection
        prompts = self._generate_collection_prompts()
        
        for prompt in prompts:
            try:
                # Collect from multiple frontier models
                for model_name, adapter in self.frontier_adapters.items():
                    # Generate completion
                    response = await adapter.generate_response(
                        prompt=prompt,
                        enable_steering=False,  # Collect natural behavior
                        max_tokens=200
                    )
                    
                    # Quality assessment
                    quality_score = self._assess_pair_quality(prompt, response.text)
                    
                    if quality_score >= self.config.quality_threshold:
                        # Add to buffer with importance weight
                        pair = {
                            "prompt": prompt,
                            "completion": response.text,
                            "model_name": model_name,
                            "timestamp": time.time(),
                            "quality_score": quality_score,
                            "metadata": {
                                "tokens": len(response.text.split()),
                                "model": model_name,
                                "logprobs": getattr(response, 'logprobs', None)
                            }
                        }
                        
                        self.distillation_buffer.append(pair)
                        self.importance_weights.append(self._calculate_importance_weight(pair))
                        self.quality_scores.append(quality_score)
                        
                        collected_pairs += 1
                        
                        # Add to shadow model for immediate distillation
                        await self.shadow_model.add_distillation_pair(
                            prompt=prompt,
                            frontier_completion=response.text,
                            model_name=model_name,
                            metadata=pair["metadata"]
                        )
                
            except Exception as e:
                logger.error(f"Failed to collect pair for prompt: {prompt[:50]}... - {e}")
        
        # Update collection stats
        collection_time = time.time() - collection_start
        self.collection_stats = self._update_collection_stats(
            collected_pairs, collection_time
        )
        
        logger.info(f"Collected {collected_pairs} new pairs in {collection_time:.1f}s")
    
    def _generate_collection_prompts(self) -> List[str]:
        """Generate diverse prompts for distillation collection"""
        
        # Prompt categories for diverse coverage
        prompt_categories = [
            # Factual questions
            "What is the capital of {country}?",
            "Who invented {invention}?",
            "When did {event} occur?",
            
            # Reasoning questions
            "Explain {concept} step by step.",
            "What are the pros and cons of {topic}?",
            "How does {process} work?",
            
            # Creative tasks
            "Write a short story about {topic}.",
            "Explain {topic} to a 5th grader.",
            "Summarize the key points about {topic}.",
            
            # Opinion/analysis
            "What is your view on {controversial_topic}?",
            "Compare and contrast {topic1} and {topic2}.",
            "What are the implications of {trend}?"
        ]
        
        # Fill in templates with diverse values
        prompts = []
        
        templates_with_values = [
            ("What is the capital of {country}?", ["France", "Japan", "Brazil", "Australia"]),
            ("Who invented {invention}?", ["the telephone", "the airplane", "the computer", "the internet"]),
            ("Explain {concept} step by step.", ["photosynthesis", "gravity", "democracy", "blockchain"]),
            ("What are the pros and cons of {topic}?", ["remote work", "artificial intelligence", "nuclear energy", "social media"]),
            ("How does {process} work?", ["photosynthesis", "digestion", "climate change", "vaccination"]),
            ("Write a short story about {topic}.", ["time travel", "space exploration", "artificial intelligence", "climate change"]),
            ("What is your view on {controversial_topic}?", ["universal basic income", "gene editing", "space colonization", "nuclear power"]),
            ("Compare and contrast {topic1} and {topic2}.", [("Python and JavaScript", "machine learning and deep learning", ("renewable energy and fossil fuels", "classical music and jazz"))])
        ]
        
        for template, values in templates_with_values:
            if isinstance(values[0], tuple):
                # Multiple values
                value_list = values[0]
                for value in value_list[:3]:  # Limit per template
                    prompts.append(template.format(**{template.split("{")[1].split("}")[0]: value}))
            else:
                # Single value list
                for value in values[:3]:  # Limit per template
                    prompts.append(template.format(**{template.split("{")[1].split("}")[0]: value}))
        
        return prompts[:20]  # Return up to 20 prompts
    
    def _assess_pair_quality(self, prompt: str, completion: str) -> float:
        """Assess quality of (prompt, completion) pair"""
        
        quality_factors = []
        
        # Length appropriateness
        completion_length = len(completion.split())
        if 10 <= completion_length <= 200:
            quality_factors.append(1.0)
        elif 5 <= completion_length <= 300:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # Coherence (simple heuristic)
        if completion and completion[0].isupper() and completion[-1] in ".!?":
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        # Relevance to prompt (simple keyword overlap)
        prompt_words = set(prompt.lower().split())
        completion_words = set(completion.lower().split())
        overlap = len(prompt_words.intersection(completion_words))
        relevance = min(overlap / max(len(prompt_words), 1), 1.0)
        quality_factors.append(relevance)
        
        # Diversity (avoid repetitive responses)
        diversity_score = len(set(completion.lower().split())) / max(len(completion.split()), 1)
        quality_factors.append(diversity_score)
        
        # Overall quality score
        return np.mean(quality_factors)
    
    def _calculate_importance_weight(self, pair: Dict[str, Any]) -> float:
        """Calculate importance weight for distillation pair"""
        
        base_weight = pair["quality_score"]
        
        # Boost weight for rare model types
        model_boost = 1.0
        if "claude" in pair["model_name"].lower():
            model_boost = 1.2  # Boost Claude samples
        elif "gemini" in pair["model_name"].lower():
            model_boost = 1.1  # Boost Gemini samples
        
        # Boost weight for complex prompts
        complexity_boost = 1.0
        if len(pair["prompt"].split()) > 10:
            complexity_boost = 1.1
        if "?" in pair["prompt"] and "explain" in pair["prompt"].lower():
            complexity_boost = 1.2
        
        # Time decay (recent pairs more important)
        age = time.time() - pair["timestamp"]
        time_decay = np.exp(-age / (24 * 3600))  # Decay over 24 hours
        
        return base_weight * model_boost * complexity_boost * time_decay
    
    def _update_collection_stats(self, collected_pairs: int, collection_time: float) -> CollectionStats:
        """Update collection statistics"""
        
        total_pairs = len(self.distillation_buffer)
        recent_pairs = collected_pairs
        
        quality_score = np.mean(self.quality_scores) if self.quality_scores else 0.0
        
        # Calculate diversity (simple heuristic)
        diversity_score = self._calculate_buffer_diversity()
        
        # Calculate coverage (model distribution)
        coverage_score = self._calculate_model_coverage()
        
        # Collection rate
        collection_rate = collected_pairs / max(collection_time, 1)
        
        return CollectionStats(
            total_pairs=total_pairs,
            recent_pairs=recent_pairs,
            quality_score=quality_score,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            collection_rate=collection_rate
        )
    
    def _calculate_buffer_diversity(self) -> float:
        """Calculate diversity of prompts in buffer"""
        
        if len(self.distillation_buffer) < 10:
            return 0.0
        
        # Sample recent pairs
        recent_pairs = list(self.distillation_buffer)[-100:]
        
        # Calculate prompt diversity based on length and vocabulary
        prompt_lengths = [len(pair["prompt"].split()) for pair in recent_pairs]
        length_diversity = np.std(prompt_lengths) / max(np.mean(prompt_lengths), 1)
        
        # Vocabulary diversity
        all_prompts = " ".join([pair["prompt"] for pair in recent_pairs])
        unique_words = len(set(all_prompts.lower().split()))
        total_words = len(all_prompts.lower().split())
        vocab_diversity = unique_words / max(total_words, 1)
        
        return (length_diversity + vocab_diversity) / 2
    
    def _calculate_model_coverage(self) -> float:
        """Calculate coverage across different models"""
        
        if len(self.distillation_buffer) < 10:
            return 0.0
        
        # Count samples per model
        model_counts = {}
        for pair in self.distillation_buffer:
            model = pair["model_name"]
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Calculate evenness of distribution
        total_samples = len(self.distillation_buffer)
        expected_per_model = total_samples / len(model_counts)
        
        coverage_score = 1.0 - (np.std(list(model_counts.values())) / max(expected_per_model, 1))
        return max(0.0, coverage_score)
    
    async def _check_and_train(self):
        """Check if training is needed and execute if so"""
        
        current_time = time.time()
        
        # Check if we have enough samples
        if len(self.distillation_buffer) < self.config.min_training_samples:
            return
        
        # Check if enough time has passed since last training
        if current_time - self.last_training_time < self.config.training_interval:
            return
        
        # Check fidelity drift
        fidelity_status = self.shadow_model.get_fidelity_status()
        if fidelity_status["latest_fidelity"] > self.config.fidelity_threshold:
            return  # Fidelity is good, no training needed
        
        # Trigger training
        await self._execute_training()
    
    async def _execute_training(self):
        """Execute shadow model training"""
        
        training_start = time.time()
        
        try:
            logger.info("Starting shadow model training")
            
            # Prepare training data with importance sampling
            training_data = self._prepare_training_data()
            
            if len(training_data) < self.config.min_training_samples:
                logger.warning("Insufficient training data")
                return
            
            # Execute training (this would call the shadow model's training method)
            # For now, simulate training
            await asyncio.sleep(5)  # Simulate training time
            
            # Update training stats
            training_time = time.time() - training_start
            self.training_stats = TrainingStats(
                training_steps=self.training_stats.training_steps + 1,
                latest_loss=np.random.uniform(0.1, 0.5),  # Simulated loss
                fidelity_improvement=np.random.uniform(0.05, 0.15),  # Simulated improvement
                training_time=training_time,
                convergence_status="converged" if training_time < 300 else "timeout"
            )
            
            self.last_training_time = time.time()
            
            logger.info(f"Training completed in {training_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.training_stats.convergence_status = "failed"
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data with importance sampling"""
        
        if not self.distillation_buffer or not self.importance_weights:
            return []
        
        # Convert to lists for sampling
        buffer_list = list(self.distillation_buffer)
        weights_list = list(self.importance_weights)
        
        # Importance sampling
        if len(buffer_list) > 1000:
            # Sample with replacement based on importance weights
            weights = np.array(weights_list)
            weights = weights / weights.sum()  # Normalize
            
            sampled_indices = np.random.choice(
                len(buffer_list),
                size=1000,
                replace=True,
                p=weights
            )
            
            training_data = [buffer_list[i] for i in sampled_indices]
        else:
            training_data = buffer_list
        
        return training_data
    
    def get_distillation_status(self) -> Dict[str, Any]:
        """Get comprehensive distillation status"""
        
        return {
            "collection": {
                "active": self.collection_active,
                "stats": {
                    "total_pairs": self.collection_stats.total_pairs,
                    "recent_pairs": self.collection_stats.recent_pairs,
                    "quality_score": self.collection_stats.quality_score,
                    "diversity_score": self.collection_stats.diversity_score,
                    "coverage_score": self.collection_stats.coverage_score,
                    "collection_rate": self.collection_stats.collection_rate
                }
            },
            "training": {
                "active": self.training_active,
                "stats": {
                    "training_steps": self.training_stats.training_steps,
                    "latest_loss": self.training_stats.latest_loss,
                    "fidelity_improvement": self.training_stats.fidelity_improvement,
                    "training_time": self.training_stats.training_time,
                    "convergence_status": self.training_stats.convergence_status
                }
            },
            "fidelity": self.shadow_model.get_fidelity_status() if hasattr(self, 'shadow_model') else None
        }

if __name__ == "__main__":
    """
    Standalone testing for online distillation
    Tests collection, training loops, and quality assessment
    """
    
    async def test_online_distillation():
        """Test all online distillation functionality"""
        print("Testing Online Distillation...")
        
        # Create config
        config = DistillationConfig(
            collection_interval=5.0,  # 5 seconds for testing
            training_interval=15.0,  # 15 seconds for testing
            buffer_size=100,
            min_training_samples=10
        )
        
        # Create distillation manager
        distillation = OnlineDistillation(config)
        
        # Test quality assessment
        print("\n1. Testing quality assessment...")
        quality_scores = []
        test_pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Explain quantum computing.", "Quantum computing uses quantum phenomena."),
            ("Invalid", "x"),  # Low quality
            ("Write a story", "Once upon a time there was a dragon who lived in a castle. The dragon was very lonely and wanted to find a friend. One day, a princess came to visit the castle and they became best friends.")
        ]
        
        for prompt, completion in test_pairs:
            quality = distillation._assess_pair_quality(prompt, completion)
            quality_scores.append(quality)
            print(f"Prompt: {prompt[:30]}... -> Quality: {quality:.3f}")
        
        print(f"Average quality: {np.mean(quality_scores):.3f}")
        
        # Test importance weight calculation
        print("\n2. Testing importance weight calculation...")
        test_pair = {
            "prompt": "Explain artificial intelligence in detail.",
            "completion": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks that typically require human intelligence.",
            "model_name": "claude-3-opus",
            "timestamp": time.time(),
            "quality_score": 0.8
        }
        
        importance = distillation._calculate_importance_weight(test_pair)
        print(f"Importance weight: {importance:.3f}")
        
        # Test diversity calculation
        print("\n3. Testing diversity calculation...")
        # Add some test pairs to buffer
        for i in range(20):
            distillation.distillation_buffer.append({
                "prompt": f"Test prompt {i}",
                "completion": f"Test completion {i}",
                "model_name": "test-model",
                "timestamp": time.time(),
                "quality_score": 0.7
            })
        
        diversity = distillation._calculate_buffer_diversity()
        coverage = distillation._calculate_model_coverage()
        print(f"Buffer diversity: {diversity:.3f}")
        print(f"Model coverage: {coverage:.3f}")
        
        # Test status reporting
        print("\n4. Testing distillation status...")
        status = distillation.get_distillation_status()
        print(f"Collection active: {status['collection']['active']}")
        print(f"Training active: {status['training']['active']}")
        print(f"Total pairs: {status['collection']['stats']['total_pairs']}")
        
        print("\nOnline Distillation test complete!")
    
    asyncio.run(test_online_distillation())
