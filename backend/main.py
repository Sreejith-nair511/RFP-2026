"""
DeceptiScope v2 - AI Interpretability System for Deception Detection
Main FastAPI application entry point

Core Innovation: Hybrid graybox-whitebox architecture that works on BOTH
closed/proprietary frontier models (GPT-5, Claude Opus 4.6, Gemini 2.5) 
AND open-weight models via shadow model proxying.

Architecture Flow:
User Prompt -> Frontier Model API -> Shadow Model -> Deception Probes -> 
Fusion Layer -> Deception Score + Steering -> Dashboard
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import all modules
from adapters.openai_adapter import OpenAIAdapter
from adapters.anthropic_adapter import AnthropicAdapter  
from adapters.gemini_adapter import GeminiAdapter
from graybox.behavioral_probe import BehavioralProbe
from shadow.shadow_model import ShadowModel
from whitebox.extractor import ActivationExtractor
from fusion.fusion_layer import FusionLayer
from eval.harness import EvaluationHarness as EvalHarness

# Configure logging for interpretability research
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for active connections and models
active_connections: Dict[str, WebSocket] = {}
frontier_adapters = {}
shadow_models = {}
behavioral_probe = None
fusion_layer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DeceptiScope components on startup"""
    logger.info("Initializing DeceptiScope v2...")
    
    # Initialize frontier model adapters
    frontier_adapters['openai'] = OpenAIAdapter()
    frontier_adapters['anthropic'] = AnthropicAdapter()
    frontier_adapters['gemini'] = GeminiAdapter()
    
    # Initialize behavioral probe (graybox analysis)
    global behavioral_probe
    behavioral_probe = BehavioralProbe()
    
    # Initialize fusion layer for combining signals
    global fusion_layer
    fusion_layer = FusionLayer()
    
    logger.info("DeceptiScope v2 initialization complete")
    yield
    
    logger.info("Shutting down DeceptiScope v2...")

app = FastAPI(
    title="DeceptiScope v2",
    description="AI Interpretability System for Deception Detection in Frontier Models",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "DeceptiScope v2",
        "innovation": "Hybrid graybox-whitebox deception detection for frontier models"
    }

@app.get("/models")
async def list_available_models():
    """List available frontier models and their capabilities"""
    return {
        "frontier_models": {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-5-preview"],
            "anthropic": ["claude-3-opus-4.6", "claude-3-sonnet-4.6"],
            "gemini": ["gemini-2.5-pro"]
        },
        "shadow_models": list(shadow_models.keys()),
        "capabilities": {
            "logprob_extraction": ["openai"],
            "cot_tokens": ["anthropic", "openai"],
            "streaming": ["openai", "anthropic", "gemini"]
        }
    }

@app.websocket("/ws/chat/{model_name}")
async def websocket_chat(websocket: WebSocket, model_name: str):
    """Real-time chat interface with deception analysis"""
    await websocket.accept()
    connection_id = f"{model_name}_{id(websocket)}"
    active_connections[connection_id] = websocket
    
    try:
        while True:
            # Receive user message
            data = await websocket.receive_json()
            user_prompt = data.get("message", "")
            enable_steering = data.get("enable_steering", True)
            
            logger.info(f"Received prompt for {model_name}: {user_prompt[:100]}...")
            
            # Route to appropriate frontier adapter
            adapter = frontier_adapters.get(model_name.split('_')[0])
            if not adapter:
                await websocket.send_json({
                    "error": f"Unknown model: {model_name}"
                })
                continue
            
            # Generate response from frontier model
            frontier_response = await adapter.generate_response(
                prompt=user_prompt,
                enable_steering=enable_steering
            )
            
            # Extract behavioral signals (graybox probing)
            behavioral_signals = await behavioral_probe.analyze_response(
                prompt=user_prompt,
                response=frontier_response,
                model_name=model_name
            )
            
            # Get shadow model analysis if available
            shadow_analysis = None
            shadow_model = shadow_models.get(model_name)
            if shadow_model:
                shadow_analysis = await shadow_model.analyze_deception(
                    prompt=user_prompt,
                    frontier_response=frontier_response
                )
            
            # Fuse all signals in fusion layer
            deception_result = await fusion_layer.fuse_signals(
                behavioral_signals=behavioral_signals,
                shadow_analysis=shadow_analysis,
                prompt=user_prompt,
                response=frontier_response
            )
            
            # Send comprehensive analysis back to client
            await websocket.send_json({
                "response": frontier_response.text,
                "deception_score": deception_result.score,
                "deception_type": deception_result.deception_type,
                "confidence": deception_result.confidence,
                "explanation": deception_result.explanation,
                "steering_applied": frontier_response.steering_applied,
                "token_analysis": {
                    "per_token_scores": deception_result.per_token_scores,
                    "high_risk_tokens": deception_result.high_risk_tokens
                },
                "behavioral_signals": {
                    "entropy": behavioral_signals.get("entropy"),
                    "consistency": behavioral_signals.get("consistency"),
                    "cot_contradiction": behavioral_signals.get("cot_contradiction")
                }
            })
            
    except WebSocketDisconnect:
        del active_connections[connection_id]
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})

@app.post("/api/shadow-model/{target_model}")
async def create_shadow_model(target_model: str, config: Dict[str, Any]):
    """Initialize shadow model for target frontier model"""
    try:
        shadow_model = ShadowModel(
            target_model=target_model,
            base_model=config.get("base_model", "microsoft/DialoGPT-medium"),
            **config
        )
        
        await shadow_model.initialize()
        shadow_models[target_model] = shadow_model
        
        return {
            "status": "success",
            "shadow_model_id": target_model,
            "base_model": shadow_model.base_model_name,
            "target_fidelity": shadow_model.target_fidelity
        }
    except Exception as e:
        logger.error(f"Failed to create shadow model: {e}")
        return {"error": str(e)}

@app.post("/api/evaluate")
async def run_evaluation(eval_config: Dict[str, Any]):
    """Run deception detection evaluation"""
    try:
        from eval.harness import EvalConfig
        config = EvalConfig(
            benchmarks=eval_config.get("benchmarks"),
            baseline_methods=eval_config.get("baseline_methods"),
            sample_sizes=eval_config.get("sample_sizes"),
            steering_enabled=eval_config.get("steering_enabled", True),
        )
        harness = EvalHarness(config)
        results = await harness.run_evaluation(None, frontier_adapters)
        # Serialise dataclass results
        return {
            k: {
                "benchmark_name": v.benchmark_name,
                "total_examples": v.total_examples,
                "auc_roc": v.auc_roc,
                "accuracy": v.accuracy,
                "f1_score": v.f1_score,
                "calibration_error": v.calibration_error,
                "steering_improvement": v.steering_improvement,
                "baseline_comparisons": v.baseline_comparisons,
            }
            for k, v in results.items()
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    """
    Standalone testing mode - run DeceptiScope locally
    
    This allows researchers to test the system without the full frontend.
    All modules include __main__ blocks for independent testing.
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
