from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# 1. Initialize the App
app = FastAPI(title="Fake News Detector API")

# 2. Load the AI Model when the server starts
try:
    print("Loading AI Model... (This may take a minute on first run)")
    fake_news_classifier = pipeline(
        "text-classification", 
        model="Pulk17/Fake-News-Detection"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    fake_news_classifier = None

# 3. Define the expected input format from the user
class NewsInput(BaseModel):
    text: str

# 4. Create the endpoint to handle the analysis
@app.post("/analyze")
async def analyze_news(news: NewsInput):
    # Check if model loaded properly
    if not fake_news_classifier:
        raise HTTPException(status_code=500, detail="AI Model failed to load.")
    
    # Ensure the user actually sent enough text
    if len(news.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short to analyze.")

    try:
        # Run the AI model on the text
        result = fake_news_classifier(news.text, truncation=True, max_length=512)
        prediction = result[0]
        
        # 1. Map raw model labels to readable text
        label_mapping = {
            "LABEL_0": "Fake News",
            "LABEL_1": "Real News"
        }
        
        # 2. Convert to readable output
        readable_prediction = label_mapping.get(prediction['label'], prediction['label'])
        
        # Return the clean results
        return {
            "analyzed_text_preview": news.text[:100] + "...",
            "prediction": readable_prediction,
            "confidence_score": round(prediction['score'], 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))