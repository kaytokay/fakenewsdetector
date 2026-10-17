# TruthShield

**A multimodal tool for screening news claims, screenshots, images, and videos for misinformation and synthetic-media signals.**

TruthShield is a local FastAPI app with a browser dashboard. It combines machine-learning outputs with basic image forensics, metadata, OCR, and optional publisher-reference searches. Results are screening clues for review, not proof that content is true, false, authentic, or AI-generated.

## Features

- **News text:** misinformation classification, sentiment, and related publisher or fact-check references.
- **Screenshots:** OCR text extraction followed by claim screening and reference search.
- **Images:** whole-image AI-generation screening, separate face-forgery screening, EXIF/C2PA clues, pixel statistics, ELA, and FFT previews.
- **Videos:** frame-by-frame image and face screening, pixel statistics, and exploratory rPPG analysis.
- **URLs:** fetches a public article or image and routes it to the relevant analysis.
- **Dashboard:** responsive single-page interface, session telemetry, and live updates.

## Requirements

- Python 3.11 recommended (also used by the Docker image).
- Internet access on first startup to download model weights and browser assets.
- Several gigabytes of free disk space and sufficient RAM for the selected PyTorch, OCR, and vision models. Startup may take a while; models that fail to load are logged and their features will be unavailable.
- Docker, if using the container setup.

The app runs on CPU by default. Video analysis and model startup can be slow on modest hardware.

## Run locally on Windows

Open PowerShell in the project directory:

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m textblob.download_corpora
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The first server start downloads and initializes the models. Keep the terminal open while using the app; press `Ctrl+C` to stop it.

For the Windows launcher, run `start_app.bat` after creating the `env` environment and installing the requirements. It can start, stop, restart, check, and open the local app.

## Run with Docker

From the project directory:

```sh
docker build -t truthshield .
docker run --rm -p 8000:8000 truthshield
```

Then open [http://localhost:8000](http://localhost:8000). The image installs dependencies and downloads TextBlob corpora during build; model weights are loaded/downloaded at application startup.

## API

The dashboard and API are served from the same origin. FastAPI's interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the app is running.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web dashboard |
| `POST` | `/analyze-text` | Analyze a text claim |
| `POST` | `/analyze-screenshot` | OCR and analyze a screenshot |
| `POST` | `/analyze-image` | Analyze an image |
| `POST` | `/analyze-video` | Analyze a video |
| `POST` | `/analyze-url` | Fetch and analyze a public URL |
| `GET` | `/stream-stats` | Live session statistics (Server-Sent Events) |

Upload endpoints accept multipart form data with a `file` field. Text and URL endpoints accept JSON bodies (`{"text":"..."}` and `{"url":"https://..."}`).

## Models and interpreting results

Models are initialized when the FastAPI app starts. The current code uses Hugging Face pipelines for text classification, image classification, and face-forgery screening; an ONNX whole-image detector; EfficientNet for an additional face-crop signal; and EasyOCR for screenshots. Model downloads and availability are external dependencies.

Image scores from whole-image AI-generation and face-forgery models are shown separately because they measure different tasks. A model score is not a calibrated probability of truth or authenticity. EXIF, C2PA, pixel, and compression analyses provide context and can be missing or misleading. Face-specific checks can flag real portraits, and URL searches do not independently verify every claim.

## Privacy and limitations

- The app has no account system or database; session analytics are held in process memory and reset when the server restarts.
- Uploaded media is analyzed by the local server. Video processing uses a temporary file that the analysis code removes afterward.
- URL analysis makes outbound requests to the supplied public URL; text verification may query external publisher or fact-check sources.
- This project is intended for coursework and demonstrations, not high-stakes decisions or production moderation.
- Keep `.env` and other credentials out of Git. `.env` is ignored by this repository.

## Tests

Run the repository's Python tests with:

```sh
python -m unittest discover -s tests
```

Some end-to-end checks may need network access or loaded model dependencies.

## Project files

- `main.py` — FastAPI app, model setup, and analysis endpoints.
- `index.html` — browser interface.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container build and launch instructions.
- `start_app.bat` — Windows local app manager.
- `tests/` — helper, endpoint, and end-to-end tests.
