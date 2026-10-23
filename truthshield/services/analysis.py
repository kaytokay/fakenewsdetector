"""Stable helper imports for callers; implementations live in focused modules."""
from .security import PinnedAddressAdapter, fetch_public_url, read_upload_limited
from .text import analyze_sentiment, text_fake_score, verify_news_on_web
from .vision import (
    compute_frame_pixel_statistics,
    compute_image_forensics,
    compute_pixel_statistics,
    detect_face_box,
    efficientnet_fake_score,
    extract_exif_metadata,
    extract_fake_score,
    face_cascade,
    filename_ai_penalty,
    generate_tampering_mask_and_bbox,
    largest_face_crop,
    load_whole_image_detector,
    provenance_fingerprint,
    resolution_ai_penalty,
    rppg_consistency,
    whole_image_ai_score,
)
from .metrics import compute_academic_metrics
from .telemetry import record_telemetry

__all__ = [
    "PinnedAddressAdapter", "fetch_public_url", "read_upload_limited",
    "analyze_sentiment", "text_fake_score", "verify_news_on_web",
    "compute_frame_pixel_statistics", "compute_image_forensics",
    "compute_pixel_statistics", "detect_face_box", "efficientnet_fake_score",
    "extract_exif_metadata", "extract_fake_score", "face_cascade",
    "filename_ai_penalty", "generate_tampering_mask_and_bbox", "largest_face_crop",
    "load_whole_image_detector", "provenance_fingerprint", "resolution_ai_penalty",
    "rppg_consistency", "whole_image_ai_score", "compute_academic_metrics", "record_telemetry",
]
