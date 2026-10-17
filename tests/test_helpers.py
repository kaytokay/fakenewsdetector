import unittest
import numpy as np
from PIL import Image
import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    analyze_sentiment,
    extract_exif_metadata,
    extract_fake_score,
    filename_ai_penalty,
    resolution_ai_penalty,
    compute_image_forensics,
    compute_academic_metrics,
    generate_tampering_mask_and_bbox,
    record_telemetry,
    live_analytics,
)


class TestForensicHelpers(unittest.TestCase):

    def test_sentiment_positive(self):
        text = "This is a wonderful, fantastic and highly successful breakthrough!"
        sentiment, polarity, subjectivity = analyze_sentiment(text)
        self.assertEqual(sentiment, "Positive")
        self.assertGreater(polarity, 0.1)
        self.assertGreaterEqual(subjectivity, 0.0)

    def test_sentiment_negative(self):
        text = "This is a terrible, horrible and disastrous failure of policy."
        sentiment, polarity, subjectivity = analyze_sentiment(text)
        self.assertEqual(sentiment, "Negative")
        self.assertLess(polarity, -0.1)

    def test_sentiment_neutral(self):
        text = "The meeting was held at 2:00 PM on Tuesday in conference room B."
        sentiment, polarity, subjectivity = analyze_sentiment(text)
        self.assertEqual(sentiment, "Neutral")
        self.assertTrue(-0.1 <= polarity <= 0.1)

    def test_extract_exif_metadata_stripped(self):
        # Programmatically generated image has no camera EXIF
        img = Image.new("RGB", (200, 200), color=(120, 150, 180))
        meta, penalty = extract_exif_metadata(img)
        self.assertEqual(penalty, 30.0)
        self.assertIn("Null / Stripped", meta.get("EXIF_Status", ""))
        self.assertEqual(meta.get("Resolution"), "200x200")

    def test_extract_exif_metadata_present(self):
        img = Image.new("RGB", (300, 200), color=(100, 100, 100))
        exif = img.getexif()
        exif[271] = "Canon"  # Make
        exif[272] = "EOS 80D"  # Model
        meta, penalty = extract_exif_metadata(img)
        self.assertEqual(penalty, 0.0)
        self.assertIn("Present", meta.get("EXIF_Status", ""))

    def test_extract_fake_score_synthetic_keywords(self):
        # Tests fake keywords
        preds = [{"label": "artificial", "score": 0.88}]
        self.assertEqual(extract_fake_score(preds), 88.0)

        preds = [{"label": "fake", "score": 0.754}]
        self.assertEqual(extract_fake_score(preds), 75.4)

        preds = [{"label": "deepfake", "score": 0.991}]
        self.assertEqual(extract_fake_score(preds), 99.1)

    def test_extract_fake_score_real_keywords(self):
        # Real keyword inverts score: (1 - 0.90) * 100 = 10.0%
        preds = [{"label": "real", "score": 0.90}]
        self.assertEqual(extract_fake_score(preds), 10.0)

        preds = [{"label": "authentic", "score": 0.80}]
        self.assertEqual(extract_fake_score(preds), 20.0)

    def test_extract_fake_score_empty_or_unknown(self):
        self.assertEqual(extract_fake_score([]), 0.0)
        preds = [{"label": "unknown_label", "score": 0.65}]
        self.assertEqual(extract_fake_score(preds), 65.0)

    def test_filename_ai_penalty(self):
        self.assertEqual(filename_ai_penalty("midjourney_v6_portrait.png"), 35.0)
        self.assertEqual(filename_ai_penalty("DALL-E-3_creation.jpg"), 35.0)
        self.assertEqual(filename_ai_penalty("gemini_generated_sample.webp"), 35.0)
        self.assertEqual(filename_ai_penalty("IMG_20241015_120405.jpg"), 0.0)
        self.assertEqual(filename_ai_penalty(""), 0.0)
        self.assertEqual(filename_ai_penalty(None), 0.0)

    def test_resolution_ai_penalty(self):
        self.assertEqual(resolution_ai_penalty(512, 512), 15.0)
        self.assertEqual(resolution_ai_penalty(1024, 1024), 15.0)
        self.assertEqual(resolution_ai_penalty(1024, 768), 15.0)
        self.assertEqual(resolution_ai_penalty(1920, 1080), 0.0)
        self.assertEqual(resolution_ai_penalty(640, 480), 0.0)

    def test_compute_image_forensics_normal(self):
        # Create a synthetic test image with varying pixels
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        forensics = compute_image_forensics(img)
        self.assertIn("ela_b64", forensics)
        self.assertIn("fft_b64", forensics)
        self.assertTrue(forensics["ela_b64"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(forensics["fft_b64"].startswith("data:image/jpeg;base64,"))
        self.assertIsInstance(forensics["ela_mean"], float)
        self.assertIsInstance(forensics["ela_std"], float)
        self.assertIsInstance(forensics["hf_ratio"], float)
        self.assertIsInstance(forensics["color_corr"], float)
        self.assertFalse(np.isnan(forensics["color_corr"]))
        self.assertFalse(np.isnan(forensics["hf_ratio"]))

    def test_compute_image_forensics_solid_color_no_nan(self):
        # Flat solid color image: checks that color_corr and hf_ratio NEVER produce NaN
        solid = Image.new("RGB", (100, 100), color=(200, 200, 200))
        forensics = compute_image_forensics(solid)
        self.assertFalse(np.isnan(forensics["color_corr"]))
        self.assertFalse(np.isnan(forensics["hf_ratio"]))
        self.assertEqual(forensics["color_corr"], 0.0)
        self.assertEqual(forensics["hf_ratio"], 0.0)

    def test_compute_academic_metrics(self):
        # True Positive case
        tp_res = compute_academic_metrics(y_true_binary=1, confidence=0.92)
        self.assertEqual(tp_res["Accuracy"], "92.0%")
        self.assertEqual(tp_res["Confusion_Matrix"]["TP"], 1)
        self.assertEqual(tp_res["Confusion_Matrix"]["FP"], 0)
        self.assertGreater(tp_res["Precision"], 0.0)
        self.assertGreater(tp_res["Recall"], 0.0)
        self.assertGreater(tp_res["F1_Score"], 0.0)
        self.assertGreater(tp_res["MCC"], 0.0)

        # True Negative case
        tn_res = compute_academic_metrics(y_true_binary=0, confidence=0.10)
        self.assertEqual(tn_res["Confusion_Matrix"]["TN"], 1)
        self.assertEqual(tn_res["Confusion_Matrix"]["TP"], 0)
        self.assertFalse(np.isnan(tn_res["MCC"]))

    def test_generate_tampering_mask_and_bbox(self):
        img = Image.new("RGB", (120, 120), color=(100, 150, 200))
        ann_fake, mask_fake = generate_tampering_mask_and_bbox(img, is_fake=True)
        self.assertTrue(ann_fake.startswith("data:image/jpeg;base64,"))
        self.assertTrue(mask_fake.startswith("data:image/jpeg;base64,"))

        ann_real, mask_real = generate_tampering_mask_and_bbox(img, is_fake=False)
        self.assertTrue(ann_real.startswith("data:image/jpeg;base64,"))
        self.assertTrue(mask_real.startswith("data:image/jpeg;base64,"))

    def test_record_telemetry(self):
        initial_total = live_analytics["total_analyzed"]
        breakdown = {"Metric A": 50.0, "Metric B": 75.0}
        record_telemetry("Text", is_fake=True, confidence_pct=88.5, sentiment="Negative", breakdown=breakdown)

        self.assertEqual(live_analytics["total_analyzed"], initial_total + 1)
        self.assertGreaterEqual(live_analytics["verdict_counts"]["Elevated signal"], 1)
        self.assertGreaterEqual(live_analytics["sentiment_counts"]["Negative"], 1)
        self.assertGreaterEqual(live_analytics["modality_counts"]["Text"], 1)
        self.assertLessEqual(len(live_analytics["recent_history"]), 15)
        self.assertEqual(live_analytics["latest_bar_breakdown"], breakdown)


if __name__ == "__main__":
    unittest.main()
