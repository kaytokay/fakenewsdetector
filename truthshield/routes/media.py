"""Image and video analysis endpoints."""
import os, tempfile, time
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from ..state import MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES, MAX_VIDEO_SECONDS, live_analytics, ml_models
from ..services.security import read_upload_limited
from ..services.telemetry import record_telemetry
from ..services.vision import (
    compute_frame_pixel_statistics, compute_image_forensics, compute_pixel_statistics,
    detect_face_box, efficientnet_fake_score, extract_exif_metadata, extract_fake_score,
    filename_ai_penalty, generate_tampering_mask_and_bbox, largest_face_crop,
    provenance_fingerprint, resolution_ai_penalty, rppg_consistency, whole_image_ai_score,
)
router = APIRouter()
@router.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    ai_clf = ml_models.get("ai_image_detector")
    df_clf = ml_models.get("deepfake_detector")
    cnn_entry = ml_models.get("cnn_deepfake_detector")
    whole_detector = ml_models.get("whole_image_detector")
    if not (whole_detector or ai_clf or df_clf or cnn_entry):
        raise HTTPException(status_code=500, detail="AI image detection models are still loading.")

    contents = await read_upload_limited(file)
    try:
        raw = Image.open(BytesIO(contents))
        if raw.width * raw.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds the 40 megapixel limit.")
        rgb = raw.convert("RGB")
    except HTTPException:
        raise
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


@router.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    vit_clf = ml_models.get("deepfake_detector")
    cnn_entry = ml_models.get("cnn_deepfake_detector")
    ai_clf = ml_models.get("ai_image_detector")
    whole_detector = ml_models.get("whole_image_detector")
    video_model_names = (["ViT"] if vit_clf else []) + (["EfficientNet-B0 CNN"] if cnn_entry else [])
    if not (vit_clf or cnn_entry or ai_clf or whole_detector):
        raise HTTPException(status_code=500, detail="Video deepfake vision model not loaded.")

    contents = await read_upload_limited(file)
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
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Video frame resolution exceeds the 40 megapixel limit.")
        duration_reported = reported_frames / fps if fps > 0 and reported_frames > 0 else 0
        if duration_reported > MAX_VIDEO_SECONDS:
            raise HTTPException(status_code=413, detail="Video exceeds the 120 second limit.")
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or "Unknown"
        timeline = []
        face_scores = []
        whole_image_scores = []
        region_signals = {"left": [], "right": []}
        all_patch_means = []
        all_patch_stds = []
        frame_index = 0
        sampled_frames = 0
        sample_interval = max(1, int(fps))  # one frame per second, capped below
        processing_started = time.monotonic()

        # Sample one frame per second and enforce duration, frame-count and wall-clock budgets.
        while sampled_frames < MAX_VIDEO_FRAMES:
            if time.monotonic() - processing_started > MAX_VIDEO_SECONDS:
                raise HTTPException(status_code=413, detail="Video analysis exceeded its 120 second processing budget.")
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
            frame_index += sample_interval
            sampled_frames += 1
            for _ in range(sample_interval - 1):
                if not cap.grab():
                    break
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
    duration_sec = round(reported_frames / fps, 2) if fps > 0 and reported_frames > 0 else round(total_frames * sample_interval / fps, 2) if fps > 0 else 0
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
        "Frame_Scan": f"Sampled {total_frames} frames at approximately 1 frame/second from {reported_frames} reported frames (maximum {MAX_VIDEO_FRAMES})",
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
        "preview": f"Sampled approximately 1 frame/second: whole-image model on {len(whole_image_scores)}/{total_frames}; face models on {len(face_scores)}/{total_frames}.",
        "metadata": metadata,
        "assessment": {
            "status": "INCONCLUSIVE" if inconclusive else "MODEL SIGNAL" if is_fake else "NO STRONG SIGNALS",
            "summary": "Frames are sampled at approximately 1 frame/second within duration, frame-count, resolution, upload-size, and processing-time limits. These are screening signals, not proof of origin or authenticity." if not inconclusive else "Frames were sampled within configured resource limits, but no usable image or face model signal was available.",
            "evidence": [
                {"label": "Frame coverage", "detail": f"{total_frames} sampled frames analyzed at approximately 1 frame/second."},
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

