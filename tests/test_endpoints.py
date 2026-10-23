import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image
from starlette.testclient import TestClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, ml_models, live_analytics, telemetry_connections, MAX_TELEMETRY_CONNECTIONS


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Save and reset models store before each test
        self.orig_models = ml_models.copy()
        ml_models.clear()

    def tearDown(self):
        # Restore original state
        ml_models.clear()
        ml_models.update(self.orig_models)

    # -----------------------------------------------------------------------
    # 1. Frontend Route (GET /)
    # -----------------------------------------------------------------------
    def test_serve_frontend(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("TruthShield", response.text)
        self.assertIn("Review news claims", response.text)

    # -----------------------------------------------------------------------
    # 2. Text Analysis Endpoint (POST /analyze-text)
    # -----------------------------------------------------------------------
    def test_analyze_text_model_not_loaded(self):
        response = self.client.post("/analyze-text", json={"text": "A valid test news article headline."})
        self.assertEqual(response.status_code, 500)
        self.assertIn("not loaded", response.json()["detail"])

    def test_analyze_text_too_short(self):
        ml_models["text_classifier"] = MagicMock()
        response = self.client.post("/analyze-text", json={"text": "Short"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 10 characters", response.json()["detail"])

    @patch("truthshield.routes.claims.verify_news_on_web", return_value={"corroborated": False, "match_count": 0, "sources": [], "query": ""})
    def test_empty_text_classifier_output_is_inconclusive(self, _mock_web):
        ml_models["text_classifier"] = MagicMock(return_value=[])
        response = self.client.post("/analyze-text", json={"text": "A sufficiently long claim for classifier testing."})
        self.assertEqual(response.status_code, 503)
        self.assertIn("inconclusive", response.json()["detail"])

    def test_telemetry_stream_rejects_connections_over_limit(self):
        acquired = [telemetry_connections.acquire(blocking=False) for _ in range(MAX_TELEMETRY_CONNECTIONS)]
        try:
            self.assertTrue(all(acquired))
            response = self.client.get("/stream-stats")
            self.assertEqual(response.status_code, 429)
        finally:
            for _ in acquired:
                telemetry_connections.release()

    @patch("truthshield.routes.claims.verify_news_on_web", return_value={"corroborated": False, "match_count": 0, "sources": [], "query": ""})
    def test_empty_text_classifier_output_is_inconclusive(self, _mock_web):
        ml_models["text_classifier"] = MagicMock(return_value=[])
        response = self.client.post("/analyze-text", json={"text": "A sufficiently long claim for classifier testing."})
        self.assertEqual(response.status_code, 503)
        self.assertIn("inconclusive", response.json()["detail"])

    @patch("truthshield.routes.claims.verify_news_on_web")
    def test_analyze_text_success_unverified(self, mock_web):
        mock_web.return_value = {
            "corroborated": False,
            "match_count": 0,
            "sources": [],
            "query": "aliens landed secretly in antarctica",
        }
        # Mock classifier returning fake probability (LABEL_0 is fake in Pulk17 model)
        mock_clf = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.88}])
        ml_models["text_classifier"] = mock_clf

        payload = {"text": "Aliens landed secretly in Antarctica according to anonymous leaked memo."}
        response = self.client.post("/analyze-text", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Text")
        self.assertEqual(data["prediction"], "Elevated misinformation signals")
        self.assertTrue(data["elevated_signals"])
        self.assertEqual(data["confidence_pct"], 88.0)
        self.assertIn("bar_chart_data", data)
        self.assertIn("live_stats", data)

    @patch("truthshield.routes.claims.verify_news_on_web")
    def test_analyze_text_success_corroborated(self, mock_web):
        mock_web.return_value = {
            "corroborated": True,
            "match_count": 5,
            "sources": [{"title": "Official Announcement", "link": "https://example.com/news", "source": "Reuters", "pubDate": "Oct 17"}],
            "query": "central bank interest rate decision",
        }
        # LABEL_1 is real in Pulk17 model
        mock_clf = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])
        ml_models["text_classifier"] = mock_clf

        payload = {"text": "Central bank announces quarterly interest rate decision following policy meeting."}
        response = self.client.post("/analyze-text", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["prediction"], "No strong misinformation signals")
        self.assertFalse(data["elevated_signals"])
        self.assertEqual(data["metadata"]["Related_Coverage"], "5 related result(s)")

    # -----------------------------------------------------------------------
    # 3. Screenshot Analysis Endpoint (POST /analyze-screenshot)
    # -----------------------------------------------------------------------
    def test_analyze_screenshot_models_not_loaded(self):
        response = self.client.post(
            "/analyze-screenshot",
            files={"file": ("test.png", b"fakebytes", "image/png")}
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("loading", response.json()["detail"])

    def test_analyze_screenshot_invalid_image(self):
        ml_models["ocr_reader"] = MagicMock()
        ml_models["text_classifier"] = MagicMock()
        ml_models["ai_image_detector"] = MagicMock()

        response = self.client.post(
            "/analyze-screenshot",
            files={"file": ("corrupt.png", b"not-a-real-image", "image/png")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid image", response.json()["detail"])

    def test_analyze_screenshot_ocr_too_short(self):
        mock_ocr = MagicMock()
        mock_ocr.readtext.return_value = ["Hi"]  # Only 2 chars
        ml_models["ocr_reader"] = mock_ocr
        ml_models["text_classifier"] = MagicMock()
        ml_models["ai_image_detector"] = MagicMock()

        # Valid PNG image bytes
        buf = BytesIO()
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/analyze-screenshot",
            files={"file": ("blank.png", buf.getvalue(), "image/png")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not extract enough text", response.json()["detail"])

    @patch("truthshield.routes.claims.verify_news_on_web")
    def test_analyze_screenshot_success(self, mock_web):
        mock_web.return_value = {"corroborated": False, "match_count": 0, "sources": [], "query": "breaking election update"}
        mock_ocr = MagicMock()
        mock_ocr.readtext.return_value = ["BREAKING:", "Official reports state major electoral update announced today."]
        mock_tclf = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])
        mock_iclf = MagicMock(return_value=[{"label": "real", "score": 0.95}])

        ml_models["ocr_reader"] = mock_ocr
        ml_models["text_classifier"] = mock_tclf
        ml_models["ai_image_detector"] = mock_iclf

        buf = BytesIO()
        Image.new("RGB", (200, 200), color=(220, 220, 220)).save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/analyze-screenshot",
            files={"file": ("news_screenshot.png", buf.getvalue(), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Screenshot")
        self.assertIn("prediction", data)
        self.assertIn("elevated_signals", data)
        self.assertIn("Whole_Screenshot_AI_Image_Signal", data["metadata"])
        self.assertNotIn("annotated_image_b64", data)

    # -----------------------------------------------------------------------
    # 4. Image Deepfake Analysis Endpoint (POST /analyze-image)
    # -----------------------------------------------------------------------
    def test_analyze_image_models_not_loaded(self):
        response = self.client.post(
            "/analyze-image",
            files={"file": ("test.jpg", b"anybytes", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("loading", response.json()["detail"])

    def test_analyze_image_invalid_format(self):
        ml_models["ai_image_detector"] = MagicMock()
        response = self.client.post(
            "/analyze-image",
            files={"file": ("bad.jpg", b"garbage-data", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid image", response.json()["detail"])

    def test_empty_vision_classifier_output_returns_explicit_unavailable_error(self):
        ml_models["ai_image_detector"] = MagicMock(return_value=[])
        buf = BytesIO()
        Image.new("RGB", (100, 100), color=(120, 120, 120)).save(buf, format="PNG")
        response = self.client.post(
            "/analyze-image",
            files={"file": ("empty-result.png", buf.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("no media signal", response.json()["detail"])

    def test_analyze_image_success_with_ai_filename(self):
        mock_ai = MagicMock(return_value=[{"label": "artificial", "score": 0.82}])
        mock_df = MagicMock(return_value=[{"label": "fake", "score": 0.77}])
        ml_models["ai_image_detector"] = mock_ai
        ml_models["deepfake_detector"] = mock_df

        buf = BytesIO()
        # 512x512 triggers power-of-2 resolution penalty
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        img.save(buf, format="JPEG")
        buf.seek(0)

        # filename with 'midjourney' triggers filename penalty
        response = self.client.post(
            "/analyze-image",
            files={"file": ("midjourney_test_render.jpg", buf.getvalue(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Image")
        self.assertEqual(data["prediction"], "Elevated whole-image AI-generation signal")
        self.assertTrue(data["elevated_signals"])
        self.assertGreaterEqual(data["confidence_pct"], 50.0)
        self.assertIn("ela_heatmap_b64", data)
        self.assertIn("fft_spectrum_b64", data)
        self.assertIn("bar_chart_data", data)

    def test_analyze_image_solid_flat_no_nan_crash(self):
        # Specifically tests the fixed NaN divide bug with flat solid white image
        ml_models["ai_image_detector"] = MagicMock(return_value=[{"label": "real", "score": 0.90}])
        buf = BytesIO()
        Image.new("RGB", (200, 200), color=(255, 255, 255)).save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/analyze-image",
            files={"file": ("solid_white.png", buf.getvalue(), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["Color_Covariance"], "0.0")

    # -----------------------------------------------------------------------
    # 5. Video Deepfake Analysis Endpoint (POST /analyze-video)
    # -----------------------------------------------------------------------
    def test_analyze_video_models_not_loaded(self):
        response = self.client.post(
            "/analyze-video",
            files={"file": ("clip.mp4", b"videobytes", "video/mp4")}
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("not loaded", response.json()["detail"])

    def test_analyze_video_invalid_file(self):
        ml_models["deepfake_detector"] = MagicMock()
        response = self.client.post(
            "/analyze-video",
            files={"file": ("corrupt.mp4", b"not-a-video-file", "video/mp4")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not open this video file", response.json()["detail"])

    def test_analyze_video_valid_synthetic_video(self):
        mock_df = MagicMock(return_value=[{"label": "deepfake", "score": 0.85}])
        ml_models["deepfake_detector"] = mock_df

        # Synthesize a temporary valid avi video with OpenCV
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temp_video:
            temp_video_name = temp_video.name

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(temp_video_name, fourcc, 10.0, (120, 120))
        for _ in range(15):
            frame = np.full((120, 120, 3), 128, dtype=np.uint8)
            out.write(frame)
        out.release()

        with open(temp_video_name, "rb") as f:
            video_bytes = f.read()
        os.remove(temp_video_name)

        response = self.client.post(
            "/analyze-video",
            files={"file": ("test_clip.avi", video_bytes, "video/x-msvideo")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["modality"], "Video")
        self.assertIn("Inconclusive", data["prediction"])
        self.assertFalse(data["elevated_signals"])
        self.assertEqual(data["assessment"]["status"], "INCONCLUSIVE")
        self.assertIn("video_timeline", data)
        self.assertGreater(len(data["video_timeline"]), 0)
        self.assertIn("metadata", data)
        self.assertIn("bar_chart_data", data)

    # -----------------------------------------------------------------------
    # 6. URL Scraper & Auto-Router (POST /analyze-url)
    # -----------------------------------------------------------------------
    def test_analyze_url_network_error(self):
        with patch("truthshield.routes.urls.fetch_public_url", side_effect=Exception("Connection refused")):
            response = self.client.post("/analyze-url", json={"url": "http://invalid-fake-domain.com"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("Failed to fetch URL", response.json()["detail"])

    def test_analyze_url_routes_to_image(self):
        buf = BytesIO()
        Image.new("RGB", (100, 100), color=(200, 100, 50)).save(buf, format="JPEG")
        fake_content = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.content = fake_content
        mock_resp.raise_for_status = MagicMock()

        ml_models["ai_image_detector"] = MagicMock(return_value=[{"label": "artificial", "score": 0.70}])

        with patch("truthshield.routes.urls.fetch_public_url", return_value=mock_resp):
            response = self.client.post("/analyze-url", json={"url": "https://example.com/photo.jpg"})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data["modality"], "URL (Image)")
            self.assertEqual(data["prediction"], "Elevated synthetic signals")
            self.assertTrue(data["elevated_signals"])

    def test_analyze_url_empty_scraped_article(self):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = "<html><body></body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("truthshield.routes.urls.fetch_public_url", return_value=mock_resp):
            response = self.client.post("/analyze-url", json={"url": "https://example.com/empty-article"})
            self.assertEqual(response.status_code, 400)
            self.assertIn("Could not extract sufficient article text", response.json()["detail"])

    @patch("truthshield.routes.claims.verify_news_on_web")
    def test_analyze_url_routes_to_text(self, mock_web):
        mock_web.return_value = {"corroborated": False, "match_count": 0, "sources": [], "query": "astronomers discover water on moon"}
        html_doc = """
        <html>
            <body>
                <h1>Moon Water Discovery</h1>
                <p>Astronomers have confirmed the presence of subsurface water deposits across lunar craters.</p>
                <p>The mission results were published following spectroscopic telemetry analysis.</p>
            </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html_doc
        mock_resp.raise_for_status = MagicMock()

        mock_clf = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.85}])
        ml_models["text_classifier"] = mock_clf

        with patch("truthshield.routes.urls.fetch_public_url", return_value=mock_resp):
            response = self.client.post("/analyze-url", json={"url": "https://example.com/space-article"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["modality"], "URL")
            self.assertEqual(data["prediction"], "No strong misinformation signals")
            self.assertFalse(data["elevated_signals"])


if __name__ == "__main__":
    unittest.main()
