"""URL fetching and auto-routing endpoint."""
from io import BytesIO
import urllib.parse
import requests
from bs4 import BeautifulSoup
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from ..state import MAX_IMAGE_PIXELS, live_analytics, ml_models
from ..schemas import TextInput, URLInput
from ..services.security import fetch_public_url
from ..services.telemetry import record_telemetry
from ..services.vision import extract_exif_metadata, extract_fake_score, generate_tampering_mask_and_bbox
from .claims import analyze_text_endpoint
router = APIRouter()
@router.post("/analyze-url")
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
            if raw.width * raw.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image exceeds the 40 megapixel limit.")
            rgb = raw.convert("RGB")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="URL did not return a valid readable image.")

        clf = ml_models.get("ai_image_detector")
        if not clf:
            raise HTTPException(status_code=503, detail="Whole-image AI-generation detector is unavailable; face-forgery models are not compatible with this input.")

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
    page_title = ""
    title_meta = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "twitter:title"})
    if title_meta and title_meta.get("content"):
        page_title = title_meta["content"].strip()
    elif soup.title:
        page_title = soup.title.get_text(" ", strip=True)
    description_meta = (
        soup.find("meta", attrs={"property": "og:description"})
        or soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", attrs={"name": "twitter:description"})
    )
    page_description = description_meta.get("content", "").strip() if description_meta else ""
    paras = [p.get_text() for p in soup.find_all("p")]
    article = " ".join(paras).strip() or soup.get_text(separator=" ", strip=True)
    if len(article) < 10:
        raise HTTPException(status_code=400, detail="Could not extract sufficient article text from URL (minimum 10 characters required).")
    # Put publisher-written title and summary first so key terms for cross-source
    # discovery come from the article's central subject, not incidental body text.
    web_query_text = " ".join(part for part in (page_title, page_description, article[:700]) if part)
    result = await analyze_text_endpoint(
        TextInput(text=article[:2500]),
        modality_override="URL",
        web_query_text=web_query_text or article[:700],
    )
    result["metadata"]["Article_Title"] = page_title or "Title unavailable"
    result["metadata"]["Source_Domain"] = urllib.parse.urlparse(payload.url).hostname or "Unknown"
    return result

