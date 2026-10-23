"""Shared runtime state and resource limits."""
import threading
import warnings
from PIL import Image

ml_models = {}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_VIDEO_SECONDS = 120
MAX_VIDEO_FRAMES = 120
MAX_TELEMETRY_CONNECTIONS = 8
telemetry_connections = threading.BoundedSemaphore(MAX_TELEMETRY_CONNECTIONS)

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

live_analytics = {
    "total_analyzed": 0,
    "verdict_counts": {"Lower signal": 0, "Elevated signal": 0},
    "sentiment_counts": {"Positive": 0, "Neutral": 0, "Negative": 0},
    "modality_counts": {"Text": 0, "Screenshot": 0, "Image": 0, "Video": 0, "URL": 0},
    "recent_history": [],
    "latest_bar_breakdown": {},
    "latest_video_timeline": [],
}
