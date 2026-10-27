"""FastAPI application setup and model lifecycle."""
import asyncio, os
from contextlib import asynccontextmanager
import easyocr
import torch
from torch import nn
from torchvision import models, transforms
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .state import ml_models
from .services.vision import load_whole_image_detector
from .routes.api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load multimodal AI models cleanly at startup; release resources on shutdown."""
    print("[INIT] Loading TruthShield misinformation and synthetic-media analysis models…")

    # 1. Text Fake News Classifier (Fine-tuned BERT)
    try:
        ml_models["text_classifier"] = pipeline(
            "text-classification",
            model="Pulk17/Fake-News-Detection",
        )
        print("[OK] 1. BERT Fake-News Classifier loaded.")
    except Exception as e:
        print(f"[WARN] Text classifier failed: {e}")

    # 2. Full-image AI detector. Its score is only a screening signal; it can misclassify real photos.
    try:
        ml_models["ai_image_detector"] = pipeline(
            "image-classification",
            model="dima806/ai_vs_real_image_detection",
        )
        print("[OK] 2. dima806/ai_vs_real_image_detection loaded.")
    except Exception as e:
        print(f"[WARN] AI image classifier failed: {e}")

    # Full-scene detector trained across contemporary generators; uses its
    # published calibrated threshold and checksum-verified ONNX artifact.
    try:
        ml_models["whole_image_detector"] = load_whole_image_detector()
        print("[OK] Community Forensics frontier whole-image detector loaded.")
    except Exception as e:
        print(f"[WARN] Whole-image AI detector failed: {e}")

    # 4. Facial Deepfake ViT (GAN face-swaps, DeepFaceLab, FaceForensics++)
    try:
        ml_models["deepfake_detector"] = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
        )
        print("[OK] 4. ViT Facial Deepfake Detector loaded.")
    except Exception as e:
        print(f"[WARN] Facial deepfake detector failed: {e}")

    # EfficientNet-B0 CNN trained on FaceForensics++ C23; use as a second
    # face-forgery signal alongside the ViT, never as a standalone proof.
    try:
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
            map_location="cpu", file_name="efficientnet_b0_ffpp_c23.pth", progress=False,
        )
        cnn = models.efficientnet_b0(weights=None)
        cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 2)
        cnn.load_state_dict(state)
        cnn.eval()
        ml_models["cnn_deepfake_detector"] = {
            "model": cnn,
            "preprocess": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        }
        print("[OK] EfficientNet-B0 FF++ CNN detector loaded.")
    except Exception as e:
        print(f"[WARN] EfficientNet CNN detector failed: {e}")

    # 4. EasyOCR reader for news screenshots
    try:
        ml_models["ocr_reader"] = easyocr.Reader(["en"], gpu=False)
        print("[OK] 4. EasyOCR loaded successfully.")
    except Exception as e:
        print(f"[WARN] EasyOCR init warning: {e}")

    print(f"[READY] Active models: {list(ml_models.keys())}")
    yield
    ml_models.clear()


app = FastAPI(
    title="TruthShield Multimodal Misinformation & Synthetic Media Analysis API",
    description="Multi-modal forensic analysis suite with live academic telemetry",
    version="2.1.0",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/health/live")
async def health_live():
    """Process liveness; does not imply model readiness."""
    return {"status": "alive"}

@app.get("/health/ready")
async def health_ready():
    """Expose per-model availability and gate readiness on the text classifier."""
    capabilities = {name: name in ml_models for name in (
        "text_classifier", "ai_image_detector", "whole_image_detector",
        "deepfake_detector", "cnn_deepfake_detector", "ocr_reader",
    )}
    if not capabilities["text_classifier"]:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "models": capabilities})
    return {"status": "ready", "models": capabilities}

@app.get("/")
async def serve_frontend():
    """Serve the dashboard on the same origin as the API."""
    return FileResponse(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )

app.include_router(router)
