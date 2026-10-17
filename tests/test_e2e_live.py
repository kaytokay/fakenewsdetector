import unittest
import asyncio
from io import BytesIO
import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from starlette.testclient import TestClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, lifespan, ml_models, live_analytics


@unittest.skipUnless(os.environ.get("RUN_LIVE_MODELS"), "Set RUN_LIVE_MODELS=1 to run live heavy deep-learning model inferences on CPU")
class TestLiveEndToEnd(unittest.TestCase):
    """
    End-to-End integration tests using the actual live Hugging Face pipelines
    and EasyOCR models cached on the system.
    """

    @classmethod
    def setUpClass(cls):
        # Run lifespan startup to load all real neural models
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)
        cls.lifespan_ctx = lifespan(app)
        cls.loop.run_until_complete(cls.lifespan_ctx.__aenter__())
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.loop.run_until_complete(cls.lifespan_ctx.__aexit__(None, None, None))
        finally:
            cls.loop.close()

    def test_live_models_loaded(self):
        expected = ["text_classifier", "ai_image_detector", "diffusion_detector", "deepfake_detector", "ocr_reader"]
        for model_name in expected:
            self.assertIn(model_name, ml_models, f"Model {model_name} should be loaded in live test.")
            self.assertIsNotNone(ml_models[model_name])

    def test_live_text_analysis(self):
        # Test real inference through BERT model
        news_text = "The Ministry of Health announced a new nationwide vaccination schedule starting next month."
        response = self.client.post("/analyze-text", json={"text": news_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Text")
        self.assertIn(data["prediction"], ["Elevated misinformation signals", "No strong misinformation signals"])
        self.assertIn("elevated_signals", data)
        self.assertIsInstance(data["confidence_pct"], float)
        self.assertIn("metadata", data)
        self.assertIn("bar_chart_data", data)
        self.assertIn("BERT Fake Linguistic Score (%)", data["bar_chart_data"])

    def test_live_image_analysis_temp_e(self):
        # Test real image inference on the workspace file temp_e.jpg
        temp_e_path = os.path.join(os.path.dirname(__file__), "..", "temp_e.jpg")
        self.assertTrue(os.path.exists(temp_e_path), "temp_e.jpg should exist in workspace")

        with open(temp_e_path, "rb") as f:
            img_bytes = f.read()

        response = self.client.post(
            "/analyze-image",
            files={"file": ("temp_e.jpg", img_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Image")
        self.assertIn("prediction", data)
        self.assertIn("elevated_signals", data)
        self.assertIn("ela_heatmap_b64", data)
        self.assertIn("fft_spectrum_b64", data)
        self.assertIn("annotated_image_b64", data)
        self.assertIn("tampering_mask_b64", data)
        self.assertIn("metadata", data)
        self.assertIn("FFT_Spectral_Energy_Ratio", data["metadata"])
        self.assertIn("Color_Covariance", data["metadata"])

    def test_live_screenshot_analysis_with_easyocr(self):
        # Generate a synthetic screenshot containing clearly readable text
        img = Image.new("RGB", (400, 150), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "BREAKING NEWS ALERT", fill=(0, 0, 0))
        draw.text((20, 60), "Official government policy update announced.", fill=(0, 0, 0))
        draw.text((20, 90), "Economic measures take effect immediately.", fill=(0, 0, 0))

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/analyze-screenshot",
            files={"file": ("news_screenshot.png", buf.getvalue(), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Screenshot")
        self.assertIn("OCR Extracted", data["preview"])
        self.assertIn("elevated_signals", data)
        self.assertIn("annotated_image_b64", data)
        self.assertIn("tampering_mask_b64", data)

    def test_live_video_analysis(self):
        # Generate a real 10-frame video using OpenCV VideoWriter
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temp_video:
            temp_video_name = temp_video.name

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(temp_video_name, fourcc, 10.0, (120, 120))
        for i in range(10):
            # Alternating gradient frames
            frame = np.full((120, 120, 3), (i * 20) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()

        try:
            with open(temp_video_name, "rb") as f:
                video_bytes = f.read()

            response = self.client.post(
                "/analyze-video",
                files={"file": ("sample_clip.avi", video_bytes, "video/x-msvideo")}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["modality"], "Video")
            self.assertIn("prediction", data)
            self.assertIn("elevated_signals", data)
            self.assertIn("video_timeline", data)
            self.assertGreater(len(data["video_timeline"]), 0)
            self.assertIn("metadata", data)
            self.assertIn("Frames_Sampled", data["metadata"])
        finally:
            if os.path.exists(temp_video_name):
                try:
                    os.remove(temp_video_name)
                except Exception:
                    pass

    def test_live_telemetry_stream(self):
        # Ensure stream-stats returns valid SSE response without blocking
        with self.client.stream("GET", "/stream-stats") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            for line in response.iter_lines():
                if line and line.startswith("data:"):
                    self.assertIn("total_analyzed", line)
                    break


if __name__ == "__main__":
    unittest.main()
