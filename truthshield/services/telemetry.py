"""Session telemetry updates."""
from datetime import datetime
from ..state import live_analytics
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

