"""Text and screenshot claim-analysis endpoints."""
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from ..state import MAX_IMAGE_PIXELS, live_analytics, ml_models
from ..schemas import TextInput
from ..services.security import read_upload_limited
from ..services.text import analyze_sentiment, text_fake_score, verify_news_on_web
from ..services.vision import extract_fake_score, whole_image_ai_score
from ..services.telemetry import record_telemetry
router = APIRouter()
@router.post("/analyze-text")
async def analyze_text_endpoint(payload: TextInput, modality_override: str = "Text", web_query_text: str | None = None):
    if modality_override not in {"Text", "URL"}:
        raise HTTPException(status_code=400, detail="Unsupported text analysis modality.")
    classifier = ml_models.get("text_classifier")
    if not classifier:
        raise HTTPException(status_code=500, detail="Text classifier model not loaded.")

    clean = payload.text.strip()
    if len(clean) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters.")

    # 1. Live Web Fact-Checking Grounding
    web_check = await run_in_threadpool(
        verify_news_on_web,
        web_query_text or clean,
        strict_relevance=(modality_override == "URL"),
    )

    # 2. BERT stylistic classification
    res = await run_in_threadpool(classifier, clean, truncation=True, max_length=512)
    if not res:
        raise HTTPException(status_code=503, detail="Text classifier returned no predictions; the result is inconclusive.")
    pred = res[0]
    raw_fake_prob = text_fake_score(pred, classifier)

    fake_prob = raw_fake_prob
    is_fake = fake_prob >= 50.0
    verdict = "Elevated misinformation signals" if is_fake else "No strong misinformation signals"
    conf_pct = fake_prob if is_fake else round(100 - fake_prob, 2)

    sentiment, polarity, subjectivity = analyze_sentiment(clean)

    breakdown = {
        "Text Classifier Pattern Score (%)": fake_prob,
        "Subjectivity / Bias (%)": subjectivity,
    }
    related_count = web_check.get("match_count", 0)
    trusted_count = web_check.get("trusted_relevant_source_count", 0)
    trusted_result_count = web_check.get("trusted_relevant_result_count", 0)
    if modality_override == "URL":
        related_summary = (
            f"Found {related_count} independent report(s) matching key details from this page, including {trusted_count} configured trusted publisher(s). Compare the reported facts and context; matching coverage does not verify the page or every claim in it."
            if related_count else
            "No sufficiently relevant independent reports were found for this page in this search. The page remains unverified; no match does not mean its claims are false."
        )
    else:
        related_summary = (
            f"Found {related_count} relevant result(s) matching key claim terms, including {trusted_count} configured trusted publisher(s). This makes the claim worth checking against those sources; it does not prove the claim."
            if related_count else
            "No relevant results matched enough of the claim's key terms in this search. That leaves the claim unverified; it does not mean the claim is false."
        )
    assessment = {
        "status": "ELEVATED_CLASSIFIER_SIGNAL" if is_fake else web_check.get("evidence_status", "NO_RELEVANT_REPORTING"),
        "related_result_count": related_count,
        "trusted_relevant_source_count": trusted_count,
        "trusted_relevant_result_count": trusted_result_count,
        "summary": related_summary,
        "evidence": [
            {"label": "Text classifier signal", "detail": f"{fake_prob}% pattern score from a learned fake/real text classifier. This is not a probability that the claim is false."},
            {"label": "Independent coverage" if modality_override == "URL" else "Related reporting", "detail": related_summary},
            {"label": "What this means", "detail": "Matching key terms identifies coverage worth comparing. It can support or challenge individual details, but does not establish whether the page or all its claims are true." if modality_override == "URL" else "Matching key terms identifies relevant coverage that may support or challenge the claim. Open the sources and compare the details before drawing a conclusion."},
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
        "preview": (web_query_text[:160] + ("…" if len(web_query_text) > 160 else "")) if modality_override == "URL" and web_query_text else clean[:160] + ("…" if len(clean) > 160 else ""),
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


@router.post("/analyze-screenshot")
async def analyze_screenshot_endpoint(file: UploadFile = File(...)):
    ocr = ml_models.get("ocr_reader")
    t_clf = ml_models.get("text_classifier")
    img_clf = ml_models.get("ai_image_detector")
    if not (ocr and t_clf):
        raise HTTPException(status_code=500, detail="Models still loading.")

    contents = await read_upload_limited(file)
    try:
        image = Image.open(BytesIO(contents))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds the 40 megapixel limit.")
        image = image.convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # A — EasyOCR text extraction
    lines = await run_in_threadpool(ocr.readtext, contents, detail=0)
    text = " ".join(lines).strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Could not extract enough text from screenshot.")

    # B — Live Web Grounding / Fact-Checking Search
    web_check = await run_in_threadpool(verify_news_on_web, text)

    # C — BERT stylistic analysis
    t_res = await run_in_threadpool(t_clf, text, truncation=True, max_length=512)
    if not t_res:
        raise HTTPException(status_code=503, detail="Text classifier returned no predictions; the screenshot result is inconclusive.")
    t_pred = t_res[0]
    raw_text_fake = text_fake_score(t_pred, t_clf)

    # D — Use a general whole-image detector first. Face-forgery models are
    # incompatible with a screenshot and must never be used as a substitute.
    raw_visual_fake = None
    visual_model = None
    try:
        raw_visual_fake = await run_in_threadpool(whole_image_ai_score, image)
        if raw_visual_fake is not None:
            visual_model = "Calibrated whole-image detector"
    except Exception:
        # A failed primary detector may still have the compatible legacy fallback.
        raw_visual_fake = None
    if raw_visual_fake is None and img_clf:
        try:
            i_preds = await run_in_threadpool(img_clf, image)
            raw_visual_fake = extract_fake_score(i_preds)
            visual_model = "Legacy whole-image detector"
        except Exception:
            # Keep OCR claim screening available while clearly marking this signal unavailable.
            raw_visual_fake = None

    # E — News claim screening is based on OCR text; keep screenshot appearance separate.
    text_fake = round(raw_text_fake, 2)
    combined_fake = text_fake
    is_fake = text_fake >= 50.0
    verdict = "Elevated misinformation signals" if is_fake else "No strong misinformation signals"
    conf_pct = combined_fake if is_fake else round(100.0 - combined_fake, 2)

    sentiment, polarity, subjectivity = analyze_sentiment(text)

    breakdown = {
        "OCR Text Classifier Pattern Score (%)": text_fake,
        "Subjectivity Index (%)": subjectivity,
    }
    if raw_visual_fake is not None:
        breakdown["Whole Screenshot Image Pattern Score (%)"] = raw_visual_fake
    record_telemetry("Screenshot", is_fake, conf_pct, sentiment, breakdown)

    metadata = {
        "OCR_Words_Detected": len(text.split()),
        "Related_Coverage": f"{web_check['match_count']} related result(s)" if web_check["corroborated"] else "No related results found",
        "Search_Query": web_check["query"],
        "Sentiment": f"{sentiment} ({polarity})",
        "Whole_Screenshot_AI_Image_Signal": f"{raw_visual_fake}% (not used for claim verdict)" if raw_visual_fake is not None else "Unavailable",
        "Whole_Screenshot_AI_Image_Model": visual_model or "Unavailable",
    }

    related_count = web_check.get("match_count", 0)
    trusted_count = web_check.get("trusted_relevant_source_count", 0)
    trusted_result_count = web_check.get("trusted_relevant_result_count", 0)
    related_summary = (
        f"Found {related_count} relevant result(s) matching key OCR claim terms, including {trusted_count} configured trusted publisher(s). This makes the claim worth checking against those sources; it does not prove the claim or the screenshot's authenticity."
        if related_count else
        "No relevant results matched enough of the extracted claim's key terms in this search. That leaves the claim unverified; it does not mean the claim or screenshot is false."
    )
    assessment = {
        "status": "ELEVATED_CLASSIFIER_SIGNAL" if is_fake else web_check.get("evidence_status", "NO_RELEVANT_REPORTING"),
        "related_result_count": related_count,
        "trusted_relevant_source_count": trusted_count,
        "trusted_relevant_result_count": trusted_result_count,
        "summary": related_summary,
        "evidence": [
            {"label": "OCR text classifier signal", "detail": f"{text_fake}% pattern score from a learned fake/real text classifier. This is not a probability that the claim is false."},
            {"label": "Whole-page image classifier", "detail": f"{raw_visual_fake}% image-pattern score from the {visual_model}. It evaluates the complete screenshot and does not affect the claim result." if raw_visual_fake is not None else "Unavailable. No compatible whole-image model produced a signal; face-forgery models are not used for screenshots."},
            {"label": "Related reporting", "detail": related_summary},
            {"label": "What this means", "detail": "Matching key terms identifies relevant coverage that may support or challenge the claim. Open the sources and compare the details before drawing a conclusion."},
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

