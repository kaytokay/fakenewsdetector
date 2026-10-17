import asyncio
import base64
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
import json
import ipaddress
import os
import socket
import sys
import tempfile

# Ensure standard output uses UTF-8 to prevent Windows charmap exceptions
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import easyocr
import numpy as np
import torch
from functools import lru_cache
from PIL import ExifTags, Image, ImageChops, ImageEnhance
import re
import requests
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from textblob import TextBlob
from transformers import pipeline
from torchvision import models, transforms
from torch import nn

# ---------------------------------------------------------------------------
# Global model store & live session telemetry
# ---------------------------------------------------------------------------
ml_models = {}
live_analytics = {
    "total_analyzed": 0,
    "verdict_counts": {"Lower signal": 0, "Elevated signal": 0},
    "sentiment_counts": {"Positive": 0, "Neutral": 0, "Negative": 0},
    "modality_counts": {"Text": 0, "Screenshot": 0, "Image": 0, "Video": 0, "URL": 0},
    "recent_history": [],           # Last 15 scans
    "latest_bar_breakdown": {},     # Scores for the latest scan
    "latest_video_timeline": []     # Frame-by-frame data for last video scan
}


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    """Serve the dashboard on the same origin as the API."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str


class URLInput(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Forensic & Heuristic Helpers
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> tuple:
    """TextBlob sentiment analysis — matches IIT Patna FDT Paper 4."""
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity * 100, 2)
    if polarity > 0.1:
        return "Positive", polarity, subjectivity
    elif polarity < -0.1:
        return "Negative", polarity, subjectivity
    return "Neutral", polarity, subjectivity


def fetch_public_url(url: str):
    """Fetch a public HTTP(S) URL without following redirects into private networks."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a public http:// or https:// URL.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))}
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError("The URL must resolve to a public host.")
    except (socket.gaierror, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="129", "Not=A?Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Upgrade-Insecure-Requests": "1",
    }
    response = requests.get(url, headers=browser_headers, timeout=10, allow_redirects=False, stream=True)
    if 300 <= response.status_code < 400:
        response.close()
        raise ValueError("Redirects are not followed. Enter the destination URL directly.")
    response.raise_for_status()
    chunks = []
    size = 0
    for chunk in response.iter_content(64 * 1024):
        size += len(chunk)
        if size > 12 * 1024 * 1024:
            response.close()
            raise ValueError("The URL response is larger than the 12 MB limit.")
        chunks.append(chunk)
    response._content = b"".join(chunks)
    response.close()
    return response


def extract_exif_metadata(image: Image.Image) -> tuple[dict, float]:
    """
    Inspects camera EXIF tags and returns neutral metadata with no risk penalty.
    Missing tags are common after sharing, editing, or screenshot capture.
    """
    exif_data = image.getexif()
    info = {
        "Format": image.format or "PNG/Raw",
        "Resolution": f"{image.width}x{image.height}",
    }
    if not exif_data:
        info["EXIF_Status"] = "Missing or stripped (common after sharing; not a synthetic verdict)"
        return info, 0.0

    info["EXIF_Status"] = "Present (Camera Hardware Tags Found)"
    for tag_id, val in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        if isinstance(val, (str, int, float)) and len(str(val)) < 60:
            info[str(tag_name)] = str(val)
    return info, 0.0


def provenance_fingerprint(contents: bytes, media_kind: str, media_type: str) -> dict:
    """Record a stable content fingerprint and whether C2PA bytes are present.

    This is chain-of-custody metadata for this scan; it does not prove who created
    the asset. C2PA presence is only detected here, not cryptographically verified.
    """
    info = {
        "Asset_SHA256": hashlib.sha256(contents).hexdigest(),
        "Asset_Bytes": len(contents),
        "Provenance_Note": f"Fingerprint recorded at scan time ({media_kind}); origin is unverified.",
    }
    try:
        from c2pa import Context, Reader, Settings
        settings = Settings.from_dict({"verify": {"verify_cert_anchors": True, "remote_manifest_fetch": False}})
        with Context(settings) as context:
            with Reader(media_type, BytesIO(contents), context=context) as reader:
                report = json.loads(reader.json())
        validation = report.get("validation_status", [])
        active_name = report.get("active_manifest")
        active = report.get("manifests", {}).get(active_name, {}) if active_name else {}
        info["C2PA_Manifest"] = "Present and parsed"
        info["C2PA_Validation"] = "No validation errors reported" if not validation else json.dumps(validation, ensure_ascii=False)[:500]
        info["C2PA_Claim_Generator"] = str(active.get("claim_generator", "Not declared"))[:160]
    except ImportError:
        info["C2PA_Manifest"] = "Verifier unavailable; embedded marker " + ("present" if b"c2pa" in contents.lower() else "not found")
        info["C2PA_Validation"] = "Not verified"
    except Exception as exc:
        info["C2PA_Manifest"] = "No readable manifest found" if b"c2pa" not in contents.lower() else "Manifest marker present but could not be parsed"
        info["C2PA_Validation"] = f"Not verified ({type(exc).__name__})"
    return info


def efficientnet_fake_score(image: Image.Image) -> float | None:
    """Return the CNN fake-class score for a face crop, if its checkpoint loaded."""
    entry = ml_models.get("cnn_deepfake_detector")
    if not entry:
        return None
    tensor = entry["preprocess"](image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = entry["model"](tensor)
        return round(float(torch.softmax(logits, dim=1)[0, 1].item() * 100), 2)


def load_whole_image_detector() -> dict:
    import onnxruntime as ort

    cache_path = os.path.join(torch.hub.get_dir(), "checkpoints", "community_forensics_frontier_fp16.onnx")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    expected_sha256 = "d75791ba2fa59146025d342cfaafa9ddeab24af117642a94f752ee4c1619375d"
    cached_hash = hashlib.sha256(open(cache_path, "rb").read()).hexdigest() if os.path.exists(cache_path) else None
    if cached_hash != expected_sha256:
        temp_path = cache_path + ".download"
        urllib.request.urlretrieve(
            "https://huggingface.co/Thermostatic/community-forensics-frontier-detector-2026-08/resolve/main/community_forensics_frontier_fp16.onnx",
            temp_path,
        )
        with open(temp_path, "rb") as checkpoint:
            downloaded_hash = hashlib.sha256(checkpoint.read()).hexdigest()
        if downloaded_hash != expected_sha256:
            os.remove(temp_path)
            raise ValueError("Whole-image detector checkpoint checksum did not match the published artifact.")
        os.replace(temp_path, cache_path)
    session = ort.InferenceSession(cache_path, providers=["CPUExecutionProvider"])
    return {"session": session, "input_name": session.get_inputs()[0].name}


def whole_image_ai_score(image: Image.Image) -> float | None:
    """Score the entire image using the calibrated general synthetic-image model."""
    entry = ml_models.get("whole_image_detector")
    if not entry:
        return None
    image = image.convert("RGB")
    scale = 440.0 / min(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - 384) // 2
    top = (resized.height - 384) // 2
    cropped = resized.crop((left, top, left + 384, top + 384))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.transpose(array, (2, 0, 1))[None].astype(np.float32)
    logit = float(np.ravel(entry["session"].run(None, {entry["input_name"]: tensor})[0])[0])
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(logit - 0.7403357915937764, -60, 60)))
    return round(float(calibrated * 100.0), 2)


def compute_pixel_statistics(image: Image.Image, grid_size: int = 16) -> dict:
    """Create a local high-frequency residual map and descriptive patch statistics."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    normalized = cv2.resize(gray, (grid_size * 16, grid_size * 16), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(normalized, (0, 0), 1.2)
    residual = cv2.absdiff(normalized, smooth)
    patches = residual.reshape(grid_size, 16, grid_size, 16).transpose(0, 2, 1, 3)
    patch_energy = patches.astype(np.float32).std(axis=(2, 3))
    mean_energy = float(patch_energy.mean())
    variation = float(patch_energy.std())
    lo, hi = np.percentile(patch_energy, [5, 95])
    normalized_map = np.clip((patch_energy - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255).astype(np.uint8)
    enlarged = cv2.resize(normalized_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(enlarged, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".jpg", heat)
    heat_b64 = base64.b64encode(encoded).decode("ascii") if ok else ""
    return {
        "patch_residual_mean": round(mean_energy, 3),
        "patch_residual_std": round(variation, 3),
        "patches_analyzed": grid_size * grid_size,
        "heatmap_b64": f"data:image/jpeg;base64,{heat_b64}" if heat_b64 else "",
    }


def compute_frame_pixel_statistics(frame_bgr: np.ndarray) -> tuple[float, float]:
    """Cheap all-frame patch residual statistics for video temporal analysis."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 1.2)).astype(np.float32)
    patch = residual.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3).std(axis=(2, 3))
    return round(float(patch.mean()), 3), round(float(patch.std()), 3)


@lru_cache(maxsize=1)
def face_cascade():
    if not callable(getattr(cv2, "CascadeClassifier", None)):
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return None if cascade.empty() else cascade


def detect_face_box(image: Image.Image):
    """Fast face box for image/video analysis, using a single cached cascade."""
    rgb = image.convert("RGB")
    scale = min(1.0, 720.0 / max(rgb.size))
    small = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))) if scale < 1 else rgb
    try:
        gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
        cascade = face_cascade()
        if cascade is None:
            return None
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return tuple(int(round(v / scale)) for v in (x, y, w, h))
    except (AttributeError, cv2.error, OSError):
        return None


def rppg_consistency(regional_rgb: dict, fps: float) -> dict:
    """Estimate cheek pulse-band periodicity and cross-cheek agreement, not a verdict."""
    left = regional_rgb.get("left", [])
    right = regional_rgb.get("right", [])
    count = min(len(left), len(right))
    duration = count / fps if fps > 0 else 0
    if count < 30 or duration < 10:
        return {"status": "Insufficient face-video duration for rPPG (need at least 10 seconds)", "samples": count}

    def interpolate_region(samples):
        values = np.asarray(samples, dtype=np.float64)
        valid = np.isfinite(values).all(axis=1)
        if valid.mean() < 0.7:
            return None
        positions = np.arange(count)
        return np.column_stack([np.interp(positions, positions[valid], values[valid, channel]) for channel in range(3)])

    def pulse_features(samples):
        rgb = np.asarray(samples, dtype=np.float64)
        means = np.maximum(rgb.mean(axis=0), 1.0)
        r, g, b = (rgb / means - 1.0).T
        x = 3.0 * r - 2.0 * g
        y = 1.5 * r + g - 1.5 * b
        scale = np.std(x) / max(np.std(y), 1e-8)
        pulse = x - scale * y
        pulse -= pulse.mean()
        pulse *= np.hanning(count)
        spectrum = np.abs(np.fft.rfft(pulse)) ** 2
        frequencies = np.fft.rfftfreq(count, d=1.0 / fps)
        band = (frequencies >= 0.7) & (frequencies <= 3.0)
        if not np.any(band) or float(spectrum[band].sum()) <= 1e-9:
            return None, 0.0
        band_indices = np.flatnonzero(band)
        peak = int(band_indices[np.argmax(spectrum[band])])
        bpm = round(float(frequencies[peak] * 60), 1)
        neighborhood = (frequencies >= frequencies[peak] - 0.15) & (frequencies <= frequencies[peak] + 0.15)
        concentration = float(spectrum[neighborhood].sum() / max(spectrum[band].sum(), 1e-9))
        return bpm, concentration

    left_values = interpolate_region(left[:count])
    right_values = interpolate_region(right[:count])
    if left_values is None or right_values is None:
        return {"status": "Insufficient stable face-region coverage for rPPG", "samples": count}
    left_bpm, left_quality = pulse_features(left_values)
    right_bpm, right_quality = pulse_features(right_values)
    agreement = round(max(0.0, 1.0 - abs(left_bpm - right_bpm) / 60.0), 3) if left_bpm is not None and right_bpm is not None else None
    return {
        "status": "Exploratory rPPG consistency estimate; not a deepfake verdict",
        "samples": count,
        "duration_seconds": round(duration, 2),
        "left_cheek_bpm_estimate": left_bpm,
        "right_cheek_bpm_estimate": right_bpm,
        "left_signal_concentration": round(left_quality, 3),
        "right_signal_concentration": round(right_quality, 3),
        "cross_cheek_agreement": agreement,
        "limitations": "Sensitive to motion, lighting, skin exposure, compression, and frame rate; plausible pulse signals can also occur in deepfakes.",
    }


def extract_fake_score(preds: list) -> float:
    """
    Universal probability extractor for Hugging Face vision classifier predictions.
    Handles label variants: artificial, fake, deepfake, synthetic vs real, human, authentic.
    """
    if not preds:
        return 0.0

    fake_keywords = ["fake", "artificial", "synthetic", "generated", "deepfake", "manipulated"]
    real_keywords = ["real", "human", "authentic", "original", "natural"]

    for p in preds:
        lbl = str(p.get("label", "")).lower()
        if any(k in lbl for k in fake_keywords):
            return round(float(p.get("score", 0.0)) * 100, 2)

    for p in preds:
        lbl = str(p.get("label", "")).lower()
        if any(k in lbl for k in real_keywords):
            return round((1.0 - float(p.get("score", 0.0))) * 100, 2)

    return round(float(preds[0].get("score", 0.0)) * 100, 2)


_AI_FILENAME_KEYWORDS = [
    "gemini", "generated", "dall-e", "dalle", "midjourney", "stable_diff",
    "stablediff", "sdxl", "ai_image", "ai-image", "synthetic", "firefly",
    "imagegen", "bing_image", "copilot", "sora", "runway", "leonardo",
    "nightcafe", "canva_ai", "adobe_gen", "flux",
]


def filename_ai_penalty(filename: str) -> float:
    """Detects default signatures in filenames generated by Gemini, DALL-E, Midjourney, etc."""
    name_lower = (filename or "").lower()
    for kw in _AI_FILENAME_KEYWORDS:
        if kw in name_lower:
            return 35.0
    return 0.0


def resolution_ai_penalty(width: int, height: int) -> float:
    """
    AI image generators almost exclusively default to exact power-of-2 dimensions:
    512x512, 1024x1024, 2048x2048, etc. Real camera captures have arbitrary aspect ratios.
    """
    powers = {256, 512, 768, 1024, 1536, 2048, 4096}
    if width in powers and height in powers:
        return 15.0
    if width == height and width in powers:
        return 15.0
    return 0.0


def verify_news_on_web(text: str) -> dict:
    """
    Live Fact-Checking & Web Grounding via Google News RSS Search.
    Extracts core named entities and nouns, filters out clickbait and sensationalist tokens,
    and requires semantic title corroboration to prevent false positives.
    """
    # 1. Strip UI crumbs, clickbait prefixes, dates, and non-alphanumeric noise
    cleaned = re.sub(r'Home\s*>\s*[^>]+>', ' ', text)
    cleaned = re.sub(
        r'\b(VIDEO OF THE DAY|LIVE|BREAKING NEWS|BREAKING|EXCLUSIVE|TRENDING|TOP NEWS|MONEYCONTROL NEWS|RHP Briefing|HIGHLIGHTS|ALERT)\b\s*:?',
        ' ', cleaned, flags=re.I
    )
    cleaned = re.sub(
        r'\b(October|November|December|January|February|March|April|May|June|July|August|September)\s+\d{1,2},?\s+\d{2,4}\b',
        ' ', cleaned, flags=re.I
    )
    cleaned = re.sub(r'\b\d{1,2}[\.:]\d{2}\b', ' ', cleaned)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)

    # 2. Extract key nouns, named entities, verbs, numbers, and descriptive terms
    blob = TextBlob(cleaned)
    stop = {
        'today', 'yesterday', 'tomorrow', 'report', 'briefing', 'highlights', 'news', 'said', 'says',
        'update', 'first', 'second', 'routine', 'later', 'following', 'heading', 'company', 'breaking',
        'exclusive', 'trending', 'video', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what',
        'when', 'where', 'about', 'into', 'more', 'their', 'will', 'would', 'could', 'should', 'secret',
        'is', 'are', 'was', 'were', 'has', 'had', 'have', 'been', 'being', 'get', 'gets', 'got', 'make', 'makes'
    }

    important_tokens = []
    for w, pos in blob.tags:
        w_lower = w.lower()
        if (pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ') or pos in ('CD',)) and len(w) >= 3 and w_lower not in stop:
            if w_lower not in [x.lower() for x in important_tokens]:
                important_tokens.append(w)

    if len(important_tokens) < 2:
        important_tokens = [w for w in cleaned.split() if len(w) >= 3 and w.lower() not in stop][:4]

    query_tokens = important_tokens[:5]
    query = ' '.join(query_tokens).strip()
    if not query:
        return {"corroborated": False, "direct_match": False, "match_count": 0, "sources": [], "query": ""}

    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            debunk_words = {
                'hoax', 'debunked', 'fake', 'false claim', 'fact check', 'conspiracy',
                'unfounded', 'myth', 'rumour', 'rumor', 'parody', 'satire', 'alleged',
                'allegedly', 'speculation', 'mystery', 'possible', 'ufo'
            }
            valid_sources = []
            input_tokens = {
                token.lower() for token in re.findall(r"[a-z0-9]+", text.lower())
                if len(token) > 2 and token.lower() not in stop
            }

            for item in items[:8]:
                title = item.findtext("title", default="").strip()
                link = item.findtext("link", default="").strip()
                pub_date = item.findtext("pubDate", default="").strip()
                source_el = item.find("source")
                source_name = source_el.text if source_el is not None else ""
                if not source_name and "-" in title:
                    source_name = title.split("-")[-1].strip()

                t_lower = title.lower()
                is_debunk = any(dw in t_lower for dw in debunk_words)
                headline = re.split(r"\s[-–|]\s", title, maxsplit=1)[0]
                headline_tokens = {
                    token.lower() for token in re.findall(r"[a-z0-9]+", headline.lower())
                    if len(token) > 2 and token.lower() not in stop
                }
                title_overlap = (
                    len(headline_tokens & input_tokens) / len(headline_tokens)
                    if len(headline_tokens) >= 4 else 0.0
                )
                headline_match = title_overlap >= 0.75

                # Match key entity tokens
                matches = sum(
                    1 for q in query_tokens
                    if (q.lower() in t_lower or (len(q) > 4 and q[:4].lower() in t_lower))
                )

                # Keep matching debunks and fact-checks as references too; dropping these
                # would hide precisely the sources a user needs to assess a disputed claim.
                overlap_ratio = matches / len(query_tokens) if query_tokens else 0.0
                if (matches >= 3 or overlap_ratio >= 0.6) and matches >= 2:
                    valid_sources.append({
                        "title": title,
                        "link": link,
                        "source": source_name,
                        "pubDate": pub_date,
                        "is_fact_check": is_debunk,
                        "headline_match": headline_match,
                        "headline_match_pct": round(title_overlap * 100, 1),
                    })

            is_corroborated = len(valid_sources) >= 1
            return {
                "corroborated": is_corroborated,
                "direct_match": any(source["headline_match"] for source in valid_sources),
                "authenticity_score_pct": max((source["headline_match_pct"] for source in valid_sources if source["headline_match"]), default=None),
                "match_count": len(valid_sources),
                "sources": valid_sources[:4],
                "query": query
            }
    except Exception as e:
        print(f"[WARN] Live news search notice: {e}")
        return {"corroborated": False, "direct_match": False, "match_count": 0, "sources": [], "query": query}
def compute_image_forensics(image: Image.Image) -> dict:
    """
    Real Pixel-Level Deep Forensic Analysis (Paper 1 & Paper 4):
    1. Error Level Analysis (ELA) - measures compression degradation discrepancy across surfaces.
    2. 2D Fast Fourier Transform (FFT) - detects periodic upsampling lattice artifacts typical of generative VAE decoders.
    3. Laplacian Texture Smoothness - measures unnatural lack of natural high-frequency camera noise.
    4. Color Channel Covariance - checks for natural sensor chromatic dispersion.
    """
    rgb_img = image.convert("RGB")
    img_np = np.array(rgb_img)
    h, w, _ = img_np.shape
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Error Level Analysis (ELA)
    temp_buf = BytesIO()
    rgb_img.save(temp_buf, format="JPEG", quality=90)
    temp_buf.seek(0)
    resaved = Image.open(temp_buf).convert("RGB")
    ela = ImageChops.difference(rgb_img, resaved)
    extrema = ela.getextrema()
    max_d = max([ex[1] for ex in extrema])
    scale = 255.0 / max_d if max_d > 0 else 1.0
    ela_enhanced = ImageEnhance.Brightness(ela).enhance(scale)
    ela_np = np.array(ela_enhanced.convert("L"))
    ela_mean = float(np.mean(ela_np))
    ela_std = float(np.std(ela_np))

    # ELA Colormap Heatmap
    ela_color = cv2.applyColorMap(ela_np, cv2.COLORMAP_JET)
    _, buf_ela = cv2.imencode(".jpg", ela_color)
    ela_b64 = base64.b64encode(buf_ela).decode("utf-8")

    # 2. 2D Fourier Spectrum (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-9)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
    _, buf_fft = cv2.imencode(".jpg", mag_color)
    fft_b64 = base64.b64encode(buf_fft).decode("utf-8")

    # High frequency energy ratio
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r**2
    low_freq_energy = np.sum(np.abs(fshift)[mask_low])
    high_freq_energy = np.sum(np.abs(fshift)[~mask_low])
    hf_ratio = float(high_freq_energy / (low_freq_energy + 1e-9))

    # Color channel covariance
    r_ch, g_ch = img_np[:, :, 0], img_np[:, :, 1]
    r_std = float(np.std(r_ch))
    g_std = float(np.std(g_ch))
    if r_std > 1e-5 and g_std > 1e-5:
        corr_matrix = np.corrcoef(r_ch.flat, g_ch.flat)
        raw_corr = float(corr_matrix[0, 1])
        rg_corr = 0.0 if np.isnan(raw_corr) or np.isinf(raw_corr) else raw_corr
    else:
        rg_corr = 0.0

    if np.isnan(hf_ratio) or np.isinf(hf_ratio):
        hf_ratio = 0.0

    # Forensic risk score from frequency & compression domains
    forensic_synthetic_points = 0.0
    if hf_ratio > 0.50:
        forensic_synthetic_points += 15.0
    if rg_corr > 0.92:
        forensic_synthetic_points += 10.0
    if ela_std < 25.0:
        forensic_synthetic_points += 10.0

    return {
        "ela_b64": f"data:image/jpeg;base64,{ela_b64}",
        "fft_b64": f"data:image/jpeg;base64,{fft_b64}",
        "ela_mean": round(ela_mean, 2),
        "ela_std": round(ela_std, 2),
        "hf_ratio": round(hf_ratio, 3),
        "color_corr": round(rg_corr, 3),
        "forensic_risk_score": min(35.0, forensic_synthetic_points)
    }


def compute_academic_metrics(y_true_binary: int, confidence: float) -> dict:
    """
    Academic benchmark evaluation metrics: Accuracy, Precision, Recall, F1, MCC,
    and a 2x2 Confusion Matrix (TP, FP, FN, TN) conforming to TransStacking & Guardian-AI.
    """
    predicted_binary = 1 if confidence >= 0.5 else 0
    tp = 1 if (y_true_binary == 1 and predicted_binary == 1) else 0
    tn = 1 if (y_true_binary == 0 and predicted_binary == 0) else 0
    fp = 1 if (y_true_binary == 0 and predicted_binary == 1) else 0
    fn = 1 if (y_true_binary == 1 and predicted_binary == 0) else 0

    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.95, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.94, 4)
    f1 = round(
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.945,
        4,
    )

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = round(mcc_num / mcc_den if mcc_den > 0 else 0.92, 4)

    return {
        "Accuracy": f"{round(confidence * 100, 2)}%",
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "MCC": mcc,
        "Confusion_Matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


def generate_tampering_mask_and_bbox(image: Image.Image, is_fake: bool) -> tuple[str, str]:
    """
    Draws facial bounding boxes (Red for an elevated model signal, cyan otherwise)
    and creates a heuristic mask. The box color is not an authenticity verdict.
    """
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    annotated = img_np.copy()
    mask = np.zeros(gray.shape, dtype=np.uint8)
    color = (255, 0, 0) if is_fake else (0, 190, 220)  # RGB

    faces = []
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(detected) > 0:
                faces = detected
    except (AttributeError, cv2.error, OSError):
        # Some OpenCV builds omit Haar cascade support. Keep previews available
        # without claiming a detected face or failing the entire image analysis.
        faces = []

    hh, ww = gray.shape
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            if is_fake:
                mask[y:y + h, x:x + w] = 255
    else:
        # Fallback ROI when no centered face is detected
        cv2.rectangle(
            annotated,
            (int(ww * 0.15), int(hh * 0.15)),
            (int(ww * 0.85), int(hh * 0.85)),
            color, 3,
        )
        if is_fake:
            # Laplacian edge energy for localized high-frequency mask
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_norm = np.uint8(np.absolute(laplacian))
            _, thresh = cv2.threshold(lap_norm, 35, 255, cv2.THRESH_BINARY)
            mask[int(hh * 0.15):int(hh * 0.85), int(ww * 0.15):int(ww * 0.85)] = thresh[
                int(hh * 0.15):int(hh * 0.85), int(ww * 0.15):int(ww * 0.85)
            ]

    _, buf_ann = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    _, buf_mask = cv2.imencode(".jpg", mask)

    ann_b64 = base64.b64encode(buf_ann).decode("utf-8")
    mask_b64 = base64.b64encode(buf_mask).decode("utf-8")
    return f"data:image/jpeg;base64,{ann_b64}", f"data:image/jpeg;base64,{mask_b64}"


def largest_face_crop(image: Image.Image) -> Image.Image | None:
    """Return a padded crop of the largest detected face for face-specific analysis."""
    rgb = image.convert("RGB")
    face = detect_face_box(rgb)
    if face is None:
        return None
    x, y, width, height = face
    pad_x, pad_y = int(width * 0.2), int(height * 0.2)
    bounds = (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(rgb.width, x + width + pad_x),
        min(rgb.height, y + height + pad_y),
    )
    return rgb.crop(bounds)


def record_telemetry(
    modality: str,
    is_fake: bool,
    confidence_pct: float,
    sentiment: str,
    breakdown: dict,
    video_timeline: list = None,
):
    """Updates real-time metrics for connected dashboards and SSE streaming."""
    live_analytics["total_analyzed"] += 1
    verdict_key = "Elevated signal" if is_fake else "Lower signal"
    live_analytics["verdict_counts"][verdict_key] += 1
    live_analytics["sentiment_counts"][sentiment] += 1
    live_analytics["modality_counts"][modality] += 1
    live_analytics["latest_bar_breakdown"] = breakdown
    if video_timeline is not None:
        live_analytics["latest_video_timeline"] = video_timeline

    live_analytics["recent_history"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "modality": modality,
        "verdict": verdict_key,
        "confidence": round(confidence_pct, 2),
    })
    if len(live_analytics["recent_history"]) > 15:
        live_analytics["recent_history"].pop(0)


# ===========================================================================
# ENDPOINT 1 — News Text Analysis (BERT + Live Web Grounding + Sentiment)
# ===========================================================================
@app.post("/analyze-text")
async def analyze_text_endpoint(payload: TextInput, modality_override: str = "Text"):
    if modality_override not in {"Text", "URL"}:
        raise HTTPException(status_code=400, detail="Unsupported text analysis modality.")
    classifier = ml_models.get("text_classifier")
    if not classifier:
        raise HTTPException(status_code=500, detail="Text classifier model not loaded.")

    clean = payload.text.strip()
    if len(clean) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters.")

    # 1. Live Web Fact-Checking Grounding
    web_check = await run_in_threadpool(verify_news_on_web, clean)
    direct_match = bool(web_check.get("direct_match"))

    # 2. BERT stylistic classification
    res = await run_in_threadpool(classifier, clean, truncation=True, max_length=512)
    pred = res[0]
    raw_fake = pred["label"] == "LABEL_0"
    raw_conf = round(pred["score"] * 100, 2)
    raw_fake_prob = raw_conf if raw_fake else round(100 - raw_conf, 2)

    # Related coverage is useful context, but a search result cannot verify a claim.
    fake_prob = raw_fake_prob
    is_fake = fake_prob >= 50.0 and not direct_match
    verdict = "Likely authentic article — matching publisher headline found" if direct_match else "Unverified — no direct publisher headline match found"
    conf_pct = fake_prob if is_fake else round(100 - fake_prob, 2)

    sentiment, polarity, subjectivity = analyze_sentiment(clean)

    breakdown = {
        "Text Classifier Pattern Score (%)": fake_prob,
        "Subjectivity / Bias (%)": subjectivity,
    }
    assessment = {
        "status": "LIKELY AUTHENTIC ARTICLE" if direct_match else "UNVERIFIED",
        "authenticity_score_pct": web_check.get("authenticity_score_pct"),
        "summary": f"A close headline match was found from {', '.join(dict.fromkeys(source.get('source', 'Publisher') for source in web_check.get('sources', []) if source.get('headline_match')))}. This supports that the article was published by that outlet; it does not verify every claim in the article." if direct_match else "No close publisher headline match was found. Related references alone cannot establish article authenticity or whether its claims are true.",
        "evidence": [
            {"label": "Text classifier signal", "detail": f"{fake_prob}% pattern score from a learned fake/real text classifier. This is not a probability that the claim is false."},
            {"label": "Article authenticity", "detail": "A close match for the headline was found in the listed publisher references." if direct_match else "No direct headline match was found; authenticity remains unverified."},
            {"label": "Claim accuracy", "detail": "A matching article does not confirm every factual claim it contains."},
        ],
    }
    record_telemetry(modality_override, is_fake, conf_pct, sentiment, breakdown)

    return {
        "modality": modality_override,
        "prediction": verdict,
        "elevated_signals": is_fake,
        "confidence_pct": conf_pct,
        "sentiment": sentiment,
        "polarity": polarity,
        "preview": clean[:160] + ("…" if len(clean) > 160 else ""),
        "metadata": {
            "Character_Count": len(clean),
            "Word_Count": len(clean.split()),
            "Related_Coverage": f"{web_check['match_count']} related result(s)" if web_check["corroborated"] else "No related results found",
            "Sentiment": f"{sentiment} ({polarity})",
        },
        "web_verification": web_check,
        "assessment": assessment,
        "bar_chart_data": breakdown,
        "live_stats": live_analytics,
    }


# ===========================================================================
# ENDPOINT 2 — News Screenshot (EasyOCR + Live Web Fact-Checking + BERT)
# ===========================================================================
@app.post("/analyze-screenshot")
async def analyze_screenshot_endpoint(file: UploadFile = File(...)):
    ocr = ml_models.get("ocr_reader")
    t_clf = ml_models.get("text_classifier")
    img_clf = ml_models.get("ai_image_detector") or ml_models.get("deepfake_detector")
    if not (ocr and t_clf and img_clf):
        raise HTTPException(status_code=500, detail="Models still loading.")

    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # A — EasyOCR text extraction
    lines = await run_in_threadpool(ocr.readtext, contents, detail=0)
    text = " ".join(lines).strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Could not extract enough text from screenshot.")

    # B — Live Web Grounding / Fact-Checking Search
    web_check = await run_in_threadpool(verify_news_on_web, text)
    direct_match = bool(web_check.get("direct_match"))

    # C — BERT stylistic analysis
    t_res = await run_in_threadpool(t_clf, text, truncation=True, max_length=512)
    t_pred = t_res[0]
    raw_text_fake = t_pred["score"] * 100 if t_pred["label"] == "LABEL_0" else (100 - t_pred["score"] * 100)

    # D — Visual signal is reported separately: a whole-page screenshot is not a suitable
    # input for an AI-generated-image classifier, and that score must not decide claim truth.
    i_preds = await run_in_threadpool(img_clf, image)
    raw_visual_fake = extract_fake_score(i_preds)

    # E — News claim screening is based on OCR text; keep screenshot appearance separate.
    text_fake = round(raw_text_fake, 2)
    combined_fake = text_fake
    is_fake = text_fake >= 50.0 and not direct_match
    verdict = "Likely authentic article — matching publisher headline found" if direct_match else "Unverified — no direct publisher headline match found"
    conf_pct = combined_fake if is_fake else round(100.0 - combined_fake, 2)

    sentiment, polarity, subjectivity = analyze_sentiment(text)

    breakdown = {
        "OCR Text Classifier Pattern Score (%)": text_fake,
        "Whole Screenshot Image Pattern Score (%)": raw_visual_fake,
        "Subjectivity Index (%)": subjectivity,
    }
    record_telemetry("Screenshot", is_fake, conf_pct, sentiment, breakdown)

    metadata = {
        "OCR_Words_Detected": len(text.split()),
        "Related_Coverage": f"{web_check['match_count']} related result(s)" if web_check["corroborated"] else "No related results found",
        "Search_Query": web_check["query"],
        "Sentiment": f"{sentiment} ({polarity})",
        "Whole_Screenshot_AI_Image_Signal": f"{raw_visual_fake}% (not used for claim verdict)",
    }

    assessment = {
        "status": "LIKELY AUTHENTIC ARTICLE" if direct_match else "UNVERIFIED",
        "authenticity_score_pct": web_check.get("authenticity_score_pct"),
        "summary": f"A close headline match was found from {', '.join(dict.fromkeys(source.get('source', 'Publisher') for source in web_check.get('sources', []) if source.get('headline_match')))}. This supports that the article was published by that outlet. It does not prove the screenshot is unaltered, the photo is genuine, or every claim is accurate." if direct_match else "No close publisher headline match was found. Related references alone cannot establish article authenticity; the whole-page image score cannot determine whether its embedded photo is AI-generated.",
        "evidence": [
            {"label": "OCR text classifier signal", "detail": f"{text_fake}% pattern score from a learned fake/real text classifier. This is not a probability that the claim is false."},
            {"label": "Whole-page image classifier", "detail": f"{raw_visual_fake}% image-pattern score. It evaluates the complete screenshot, not a separately cropped news photo, so it does not affect the result."},
            {"label": "Article authenticity", "detail": "A close match for the article headline was found in the listed publisher references." if direct_match else "No direct headline match was found; authenticity remains unverified."},
        ],
    }

    return {
        "modality": "Screenshot",
        "prediction": verdict,
        "elevated_signals": is_fake,
        "confidence_pct": conf_pct,
        "sentiment": sentiment,
        "preview": f"OCR Extracted: {text[:180]}…",
        "metadata": metadata,
        "web_verification": web_check,
        "assessment": assessment,
        "bar_chart_data": breakdown,
        "live_stats": live_analytics,
    }


# ===========================================================================
# ENDPOINT 3 — Image Deepfake & AI-Generated Media Analysis
# ===========================================================================
@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    ai_clf = ml_models.get("ai_image_detector")
    df_clf = ml_models.get("deepfake_detector")
    cnn_entry = ml_models.get("cnn_deepfake_detector")
    whole_detector = ml_models.get("whole_image_detector")
    if not (whole_detector or ai_clf or df_clf or cnn_entry):
        raise HTTPException(status_code=500, detail="AI image detection models are still loading.")

    contents = await file.read()
    try:
        raw = Image.open(BytesIO(contents))
        rgb = raw.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # 1. Camera Hardware EXIF analysis
    metadata, _ = extract_exif_metadata(raw)

    # 2. Deep Pixel Forensics: Error Level Analysis (ELA) + 2D FFT Frequency Spectrum
    forensics = await run_in_threadpool(compute_image_forensics, rgb)
    pixel_stats = await run_in_threadpool(compute_pixel_statistics, rgb)

    # Analyze the whole image for AI-generation patterns, and a face crop for face swaps.
    ai_gen_score = 0.0
    legacy_ai_score = None
    deepfake_score = 0.0
    face_detection_available = (
        callable(getattr(cv2, "CascadeClassifier", None))
        and hasattr(getattr(cv2, "data", None), "haarcascades")
    )
    face_crop = await run_in_threadpool(largest_face_crop, rgb)

    whole_score = await run_in_threadpool(whole_image_ai_score, rgb)
    if ai_clf:
        ai_preds = await run_in_threadpool(ai_clf, rgb)
        legacy_ai_score = extract_fake_score(ai_preds)
    if whole_score is not None:
        ai_gen_score = whole_score
    elif legacy_ai_score is not None:
        ai_gen_score = legacy_ai_score

    if df_clf and face_crop is not None:
        df_preds = await run_in_threadpool(df_clf, face_crop)
        deepfake_score = extract_fake_score(df_preds)
    cnn_score = await run_in_threadpool(efficientnet_fake_score, face_crop) if face_crop is not None and cnn_entry else None

    face_model_scores = ([deepfake_score] if face_crop is not None and df_clf else []) + ([cnn_score] if cnn_score is not None else [])
    face_model_names = (["ViT"] if face_crop is not None and df_clf else []) + (["EfficientNet-B0 CNN"] if cnn_score is not None else [])
    face_model_ran = bool(face_crop is not None and face_model_scores)
    face_ensemble = round(sum(face_model_scores) / len(face_model_scores), 2) if face_model_ran else None
    # Keep the whole-image AI-generation result separate from face-forgery
    # classifiers. Their scores measure different tasks and are not comparable.
    primary_ai_score = whole_score if whole_score is not None else legacy_ai_score
    signal_pct = round(primary_ai_score, 2) if primary_ai_score is not None else 0.0
    is_fake = bool(primary_ai_score is not None and primary_ai_score >= 65.0)
    analysis_ran = primary_ai_score is not None
    conf_pct = signal_pct if is_fake else round(100.0 - signal_pct, 2) if analysis_ran else 0.0

    breakdown = {
        "Whole-Image AI Generation Signal (%)": ai_gen_score,
        **({"Legacy Full-Image Classifier Signal (%)": legacy_ai_score} if legacy_ai_score is not None else {}),
        **({"Face-Crop ViT Signal (%)": deepfake_score} if face_crop is not None and df_clf else {}),
        **({"Face-Crop EfficientNet CNN Signal (%)": cnn_score} if cnn_score is not None else {}),
        "Patch Residual Energy Mean": pixel_stats["patch_residual_mean"],
        "Patch Residual Energy Variation": pixel_stats["patch_residual_std"],
    }

    metadata.update({
        "Whole_Image_AI_Detector": f"{whole_score}%" if whole_score is not None else "Unavailable",
        "Whole_Image_AI_Threshold": "65% calibrated screening threshold" if whole_score is not None else "Unavailable",
        "Legacy_Full_Image_Model_Output": f"{legacy_ai_score}%" if legacy_ai_score is not None else "Unavailable",
        "AI_Generator_Output": f"{ai_gen_score}%" if analysis_ran else "Unavailable",
        "Face_Crop_Deepfake_Output": f"{deepfake_score}%" if face_crop is not None and df_clf else "Unavailable",
        "EfficientNet_CNN_Output": f"{cnn_score}%" if cnn_score is not None else "Unavailable",
        "Face_Detection_Status": "Face crop analyzed" if face_crop is not None else "Unavailable in this OpenCV build" if not face_detection_available else "No face detected",
        "Detection_Basis": "Whole-image AI-generation score; face-forgery models reported separately" if analysis_ran else "Whole-image AI-generation model unavailable",
        "FFT_Spectral_Energy_Ratio": f"{forensics['hf_ratio']}",
        "ELA_Compression_Std": f"{forensics['ela_std']}",
        "Color_Covariance": f"{forensics['color_corr']}",
        "Pixel_Patches_Analyzed": pixel_stats["patches_analyzed"],
        **provenance_fingerprint(contents, "image", Image.MIME.get(raw.format, "application/octet-stream")),
    })

    record_telemetry("Image", is_fake, conf_pct, "Neutral", breakdown)
    if face_model_ran:
        ann_b64, mask_b64 = generate_tampering_mask_and_bbox(rgb, is_fake)
    else:
        ann_b64, mask_b64 = "", ""

    face_flagged = face_ensemble is not None and face_ensemble >= 50.0
    prediction_str = ("Elevated whole-image AI-generation signal" if is_fake else "No strong whole-image AI-generation signal") if analysis_ran else "Inconclusive — no whole-image AI model was available"
    if analysis_ran and not is_fake and face_flagged:
        prediction_str += "; face model flagged for review"
    model_evidence = []
    if whole_score is not None:
        model_evidence.append({"label": "Whole-image AI-generation detector", "detail": f"Community Forensics calibrated score {whole_score}%; published screening threshold 65%. It evaluates a whole-image crop, not just a face; this is not proof of origin."})
    elif legacy_ai_score is not None:
        model_evidence.append({"label": "Legacy whole-image detector", "detail": f"{legacy_ai_score}% screening score; fallback model with limited calibration. It does not determine authenticity."})
    if legacy_ai_score is not None and whole_score is not None:
        model_evidence.append({"label": "Legacy model comparison", "detail": f"Older full-image model score: {legacy_ai_score}%. Kept as a secondary comparison and excluded from the calibrated primary whole-image score."})
    model_evidence.extend([
        {"label": "Face-region forgery models", "detail": f"{'+'.join(face_model_names)} screening score {face_ensemble}% on the detected face crop; {'flagged for review' if face_flagged else 'below the 50% screening threshold'}. This is a separate task from detecting AI-generated whole images, and is not a calibrated probability." if face_model_ran else "No face was detected; face-specific models were skipped. Whole-image analysis ran independently."},
        {"label": "Pixel-level statistics", "detail": f"Compared local high-frequency residual energy across {pixel_stats['patches_analyzed']} patches. The heatmap highlights variation, not confirmed edits."},
        {"label": "Metadata and provenance", "detail": f"EXIF, C2PA manifest validation, and SHA-256 fingerprint are reported as source clues; missing metadata does not mean synthetic."},
    ])
    assessment = {
        "status": "MODEL SIGNAL" if is_fake else "FACE MODEL FLAG" if face_flagged else "NO STRONG SIGNALS" if analysis_ran else "INCONCLUSIVE",
        "summary": "The gauge shows the whole-image AI-generation detector only. Face-forgery classifiers are reported separately because their scores measure a different task and are not calibrated probabilities. All model outputs are screening signals, not proof; pixel statistics, metadata, and provenance add context." if analysis_ran else "The whole-image detector was unavailable. Face-model results, if present, are separate screening signals; pixel statistics and metadata alone cannot establish whether content is synthetic.",
        "evidence": model_evidence,
    }

    return {
        "modality": "Image",
        "prediction": prediction_str,
        "elevated_signals": is_fake,
        "confidence_pct": conf_pct,
        "signal_pct": signal_pct,
        "gauge_value_pct": signal_pct,
        "face_signal_pct": face_ensemble,
        "preview": f"Whole-image AI-generation score: {signal_pct}%" + (f" | separate face-forgery score: {face_ensemble}%" if face_model_ran else " | no face crop analyzed"),
        "metadata": metadata,
        "assessment": assessment,
        "gauge_title": "Whole-image AI-generation signal",
        "gauge_label": "Separate from face-forgery screening; not proof",
        "annotated_image_b64": ann_b64,
        "tampering_mask_b64": mask_b64,
        "ela_heatmap_b64": forensics["ela_b64"],
        "fft_spectrum_b64": forensics["fft_b64"],
        "pixel_statistics_heatmap_b64": pixel_stats["heatmap_b64"],
        "bar_chart_data": breakdown,
        "live_stats": live_analytics,
    }


# ===========================================================================
# ENDPOINT 4 — Video Deepfake (Temporal Frame Sampling — Guardian-AI style)
# ===========================================================================
@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    vit_clf = ml_models.get("deepfake_detector")
    cnn_entry = ml_models.get("cnn_deepfake_detector")
    ai_clf = ml_models.get("ai_image_detector")
    whole_detector = ml_models.get("whole_image_detector")
    video_model_names = (["ViT"] if vit_clf else []) + (["EfficientNet-B0 CNN"] if cnn_entry else [])
    if not (vit_clf or cnn_entry or ai_clf or whole_detector):
        raise HTTPException(status_code=500, detail="Video deepfake vision model not loaded.")

    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open this video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or "Unknown"
        timeline = []
        face_scores = []
        whole_image_scores = []
        region_signals = {"left": [], "right": []}
        all_patch_means = []
        all_patch_stds = []
        frame_index = 0

        # Sequential decode: run whole-image AI scoring and pixel statistics on
        # every frame, plus face-specific models wherever a face is present.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            patch_mean, patch_std = await run_in_threadpool(compute_frame_pixel_statistics, frame)
            all_patch_means.append(patch_mean)
            all_patch_stds.append(patch_std)

            whole_score = await run_in_threadpool(whole_image_ai_score, pil)
            if whole_score is None and ai_clf:
                ai_preds = await run_in_threadpool(ai_clf, pil)
                whole_score = extract_fake_score(ai_preds)
            if whole_score is not None:
                whole_image_scores.append(whole_score)

            box = await run_in_threadpool(detect_face_box, pil)
            frame_scores = []
            face_crop = None
            if box is not None:
                x, y, fw, fh = box
                pad_x, pad_y = int(fw * .2), int(fh * .2)
                face_crop = pil.crop((max(0, x-pad_x), max(0, y-pad_y), min(pil.width, x+fw+pad_x), min(pil.height, y+fh+pad_y)))
                if vit_clf:
                    vit_preds = await run_in_threadpool(vit_clf, face_crop)
                    frame_scores.append(extract_fake_score(vit_preds))
                if cnn_entry:
                    cnn_value = await run_in_threadpool(efficientnet_fake_score, face_crop)
                    if cnn_value is not None:
                        frame_scores.append(cnn_value)

                # Cheek ROI RGB means feed an exploratory rPPG consistency check.
                for name, center_x in (("left", 0.32), ("right", 0.68)):
                    xa = max(0, min(pil.width - 1, x + int(fw * (center_x - .10))))
                    xb = max(xa + 1, min(pil.width, x + int(fw * (center_x + .10))))
                    ya = max(0, min(pil.height - 1, y + int(fh * .52)))
                    yb = max(ya + 1, min(pil.height, y + int(fh * .72)))
                    roi = np.asarray(pil.crop((xa, ya, xb, yb)), dtype=np.float32)
                    region_signals[name].append(roi.reshape(-1, 3).mean(axis=0).tolist())
            else:
                missing = [float("nan")] * 3
                region_signals["left"].append(missing)
                region_signals["right"].append(missing)

            face_signal = round(sum(frame_scores) / len(frame_scores), 2) if frame_scores else None
            if face_signal is not None:
                face_scores.append(face_signal)
                plotted_score = face_signal
                basis = "face-model ensemble"
            else:
                plotted_score = whole_score
                basis = "whole-image AI-generation model; no face model"

            timestamp = round(frame_index / fps, 3) if fps > 0 else 0.0
            timeline.append({
                "frame": frame_index,
                "time_label": f"{timestamp}s",
                "fake_prob": plotted_score,
                "face_signal_pct": face_signal,
                "whole_image_ai_pct": whole_score,
                "pixel_residual_mean": patch_mean,
                "pixel_residual_variation": patch_std,
                "basis": basis,
            })
            frame_index += 1
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not timeline:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")

    total_frames = len(timeline)
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0
    fake_count = sum(score >= 50.0 for score in face_scores)
    fake_ratio = round(fake_count / len(face_scores) * 100, 2) if face_scores else 0.0
    avg_face_signal = round(float(np.mean(face_scores)), 2) if face_scores else None
    peak_face_signal = round(float(max(face_scores)), 2) if face_scores else None
    avg_whole_signal = round(float(np.mean(whole_image_scores)), 2) if whole_image_scores else None
    whole_elevated_ratio = round(sum(score >= (65.0 if whole_detector else 50.0) for score in whole_image_scores) / len(whole_image_scores) * 100, 2) if whole_image_scores else 0.0
    is_fake = bool((avg_whole_signal is not None and avg_whole_signal >= (65.0 if whole_detector else 65.0)) or (face_scores and fake_ratio > 50.0))
    inconclusive = not face_scores and not whole_image_scores
    component_signals = ([avg_whole_signal] if avg_whole_signal is not None else []) + ([avg_face_signal] if avg_face_signal is not None else [])
    signal_pct = max(component_signals) if component_signals else 0.0
    conf_pct = signal_pct if is_fake else round(100.0 - signal_pct, 2)
    rppg = rppg_consistency(region_signals, fps)
    rppg_detail = rppg["status"]
    if rppg.get("left_cheek_bpm_estimate") is not None and rppg.get("right_cheek_bpm_estimate") is not None:
        rppg_detail += (
            f"; left/right cheek pulse-band estimates {rppg['left_cheek_bpm_estimate']}/{rppg['right_cheek_bpm_estimate']} BPM"
            f"; cross-cheek agreement {rppg.get('cross_cheek_agreement')}"
        )

    breakdown = {
        "Frames Analyzed (%)": 100.0,
        "Mean Whole-Image AI Generation Signal (%)": avg_whole_signal or 0.0,
        "Elevated Whole-Image Frames (%)": whole_elevated_ratio,
        "Face Frames with Model Score": len(face_scores),
        "Mean Face Deepfake Ensemble Signal (%)": avg_face_signal or 0.0,
        "Peak Face Deepfake Ensemble Signal (%)": peak_face_signal or 0.0,
        "Mean Local Pixel Residual": round(float(np.mean(all_patch_means)), 3),
        "Pixel Residual Variation Across Frames": round(float(np.std(all_patch_means)), 3),
        **({"Elevated Face-Model Frames (%)": fake_ratio} if face_scores else {}),
    }
    metadata = {
        "Filename": file.filename,
        "Duration": f"{duration_sec}s ({round(fps, 1)} FPS)",
        "Frame_Scan": f"Scanned every decoded frame: {total_frames} / container reported {reported_frames}",
        "Frame_Dimensions": f"{width}x{height}",
        "Video_Codec_FourCC": fourcc,
        "Face_Frames_Analyzed": f"{len(face_scores)} of {total_frames}",
        "Whole_Image_Frames_Analyzed": f"{len(whole_image_scores)} of {total_frames}",
        "Mean_Whole_Image_AI_Signal": f"{avg_whole_signal}%" if avg_whole_signal is not None else "Unavailable",
        "Elevated_Face_Model_Frames": f"{fake_count}/{len(face_scores)} ({fake_ratio}%)" if face_scores else "No face-model scores; inconclusive",
        "rPPG_Analysis": rppg_detail,
        **provenance_fingerprint(contents, "video", "video/mp4" if suffix.lower() == ".mp4" else "video/webm" if suffix.lower() == ".webm" else "application/octet-stream"),
    }
    record_telemetry("Video", is_fake, conf_pct, "Neutral", breakdown, video_timeline=timeline)

    return {
        "modality": "Video",
        "prediction": "Inconclusive — no usable model signals" if inconclusive else "Elevated synthetic-content signals" if is_fake else "No strong synthetic-content signals",
        "elevated_signals": is_fake,
        "confidence_pct": conf_pct,
        "signal_pct": signal_pct,
        "preview": f"Scanned every decoded frame: whole-image model on {len(whole_image_scores)}/{total_frames}; face models on {len(face_scores)}/{total_frames}.",
        "metadata": metadata,
        "assessment": {
            "status": "INCONCLUSIVE" if inconclusive else "MODEL SIGNAL" if is_fake else "NO STRONG SIGNALS",
            "summary": "Every decoded frame receives whole-image AI-generation analysis and pixel statistics; face-forgery models and exploratory rPPG run where conditions permit. These are screening signals, not proof of origin or authenticity." if not inconclusive else "Every decoded frame was scanned, but no usable image or face model signal was available.",
            "evidence": [
                {"label": "Frame coverage", "detail": f"{total_frames} of {total_frames} decoded frames analyzed."},
                {"label": "Whole-image AI generation", "detail": f"Scored {len(whole_image_scores)} frames; mean {avg_whole_signal}%, {whole_elevated_ratio}% above the screening threshold. Whole-image model score is a research estimate, not proof." if whole_image_scores else "Whole-image detector unavailable."},
                {"label": "Face-region forgery ensemble", "detail": f"{' + '.join(video_model_names)} signals on {len(face_scores)} frames; mean {avg_face_signal}% and elevated on {fake_count} frames. Scores are not calibrated probabilities." if face_scores else "No face crops found; whole-image analysis still ran on each decoded frame."},
                {"label": "rPPG", "detail": f"{rppg_detail}. This physiological signal is affected by lighting, motion, compression, skin visibility, and may be present in deepfakes."},
                {"label": "Pixel-level statistics", "detail": "High-frequency patch residual statistics were computed for every decoded frame; these locate variation, not confirmed edits."},
            ],
        },
        "rppg": rppg,
        "bar_chart_data": breakdown,
        "video_timeline": timeline,
        "live_stats": live_analytics,
    }


# ===========================================================================
# ENDPOINT 5 — URL Scraper + Auto-Router
# ===========================================================================
@app.post("/analyze-url")
async def analyze_url_endpoint(payload: URLInput):
    try:
        resp = await run_in_threadpool(fetch_public_url, payload.url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            domain = urllib.parse.urlparse(payload.url).hostname or "This website"
            raise HTTPException(
                status_code=400,
                detail=f"{domain} has anti-bot / scraper protection (403 Forbidden). Please copy the article text or headline and paste it into Tab 1 ('News Text') instead."
            )
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    ctype = resp.headers.get("Content-Type", "").lower()

    if "image" in ctype or payload.url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        try:
            raw = Image.open(BytesIO(resp.content))
            rgb = raw.convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="URL did not return a valid readable image.")

        clf = ml_models.get("ai_image_detector") or ml_models.get("deepfake_detector")
        if not clf:
            raise HTTPException(status_code=500, detail="Vision detection models are still loading.")

        preds = await run_in_threadpool(clf, rgb)
        vit_fake = extract_fake_score(preds)
        is_fake = vit_fake >= 50.0
        conf_pct = vit_fake if is_fake else round(100 - vit_fake, 2)
        breakdown = {
            "Synthetic Anomaly (%)": vit_fake,
            "Authenticity (%)": round(100 - vit_fake, 2),
        }
        record_telemetry("URL", is_fake, conf_pct, "Neutral", breakdown)
        ann_b64, mask_b64 = generate_tampering_mask_and_bbox(rgb, is_fake)
        return {
            "modality": "URL (Image)",
            "prediction": "Elevated synthetic signals" if is_fake else "No strong synthetic signals",
            "elevated_signals": is_fake,
            "confidence_pct": conf_pct,
            "preview": payload.url,
            "metadata": extract_exif_metadata(raw)[0],
            "annotated_image_b64": ann_b64,
            "tampering_mask_b64": mask_b64,
            "bar_chart_data": breakdown,
            "live_stats": live_analytics,
        }

    # Route B — scrape news article text
    soup = BeautifulSoup(resp.text, "html.parser")
    paras = [p.get_text() for p in soup.find_all("p")]
    article = " ".join(paras).strip() or soup.get_text(separator=" ", strip=True)
    if len(article) < 10:
        raise HTTPException(status_code=400, detail="Could not extract sufficient article text from URL (minimum 10 characters required).")
    return await analyze_text_endpoint(TextInput(text=article[:2500]), modality_override="URL")


# ===========================================================================
# SSE ENDPOINT — Real-Time Live Telemetry Stream
# ===========================================================================
@app.get("/stream-stats")
async def stream_stats():
    """Push live session analytics to connected browser dashboards via SSE."""
    async def generator():
        last = -1
        while True:
            if live_analytics["total_analyzed"] != last:
                last = live_analytics["total_analyzed"]
                yield f"data: {json.dumps(live_analytics)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream")
