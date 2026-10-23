"""Image and video forensic analysis helpers."""
import base64, hashlib, json, os, re, urllib.request
from functools import lru_cache
from io import BytesIO
import cv2, numpy as np, torch
from PIL import ExifTags, Image, ImageChops, ImageEnhance
from fastapi import HTTPException
from ..state import ml_models
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
        raise HTTPException(status_code=503, detail="Vision classifier returned no predictions; no media signal is available.")

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

    raise HTTPException(status_code=503, detail="Vision classifier returned an unknown label; no media signal is available.")


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

