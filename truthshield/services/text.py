"""Text classification and external reporting search helpers."""
from collections.abc import Mapping
import re, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from fastapi import HTTPException
from textblob import TextBlob
def text_fake_score(prediction: dict, classifier) -> float:
    """Map known model labels safely, using config metadata only when it is a mapping."""
    label = str(prediction.get("label", "")).strip().lower()
    config = getattr(getattr(classifier, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)
    mapped = ""
    if label.startswith("label_"):
        try:
            label_id = int(label.removeprefix("label_"))
        except ValueError:
            label_id = None
        if isinstance(id2label, Mapping) and label_id is not None:
            mapped = str(id2label.get(label_id, id2label.get(str(label_id), ""))).strip().lower()
        # This application's configured Pulk17 model uses LABEL_0=fake, LABEL_1=real.
        if mapped not in {"fake", "false", "misinformation", "misleading", "real", "true", "reliable", "genuine"}:
            mapped = {"label_0": "fake", "label_1": "real"}.get(label, "")
    resolved = mapped or label
    fake_names = {"fake", "false", "misinformation", "misleading", "label_0"}
    real_names = {"real", "true", "reliable", "genuine", "label_1"}
    if resolved in fake_names:
        return round(float(prediction["score"]) * 100, 2)
    if resolved in real_names:
        return round((1 - float(prediction["score"])) * 100, 2)
    raise HTTPException(status_code=422, detail=f"Text classifier returned an unknown label: {label or '(empty)'}")


def analyze_sentiment(text: str) -> tuple:
    """TextBlob sentiment analysis — matches IIT Patna FDT Paper 4."""
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity * 100, 2)
    if polarity > 0.1:
        return "Positive", polarity, subjectivity
    elif polarity < -0.1:
        return "Negative", polarity, subjectivity
    return "Neutral", polarity, subjectivity


def verify_news_on_web(text: str, strict_relevance: bool = False) -> dict:
    """
    Find related reporting using claim-term overlap in titles and RSS descriptions.
    Relevance is a discovery signal for human review, not proof that a claim is true.
    """
    # 1. Strip UI crumbs, clickbait prefixes, dates, and non-alphanumeric noise
    cleaned = re.sub(r'Home\s*>\s*[^>]+>', ' ', text)
    cleaned = re.sub(
        r'\b(VIDEO OF THE DAY|LIVE|BREAKING NEWS|BREAKING|EXCLUSIVE|TRENDING|TOP NEWS|MONEYCONTROL NEWS|RHP Briefing|HIGHLIGHTS|ALERT)\b\s*:?',
        ' ', cleaned, flags=re.I
    )
    cleaned = re.sub(
        r'\b(October|November|December|January|February|March|April|May|June|July|August|September)\s+\d{1,2},?\s+\d{2,4}\b',
        ' ', cleaned, flags=re.I
    )
    cleaned = re.sub(r'\b\d{1,2}[\.:]\d{2}\b', ' ', cleaned)
    cleaned = re.sub(r'[^\w\s.]', ' ', cleaned)

    # 2. Extract key nouns, named entities, verbs, numbers, and descriptive terms
    blob = TextBlob(cleaned)
    stop = {
        'today', 'yesterday', 'tomorrow', 'report', 'briefing', 'highlights', 'news', 'said', 'says',
        'update', 'first', 'second', 'routine', 'later', 'following', 'heading', 'company', 'breaking',
        'exclusive', 'trending', 'video', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what',
        'when', 'where', 'about', 'into', 'more', 'their', 'will', 'would', 'could', 'should', 'secret',
        'is', 'are', 'was', 'were', 'has', 'had', 'have', 'been', 'being', 'get', 'gets', 'got', 'make', 'makes'
    }

    important_tokens = []
    named_entity_tokens = []
    for w, pos in blob.tags:
        w_lower = w.lower()
        if (pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ') or pos in ('CD',)) and len(w) >= 3 and w_lower not in stop:
            if w_lower not in [x.lower() for x in important_tokens]:
                important_tokens.append(w)
            if strict_relevance and (pos == "NNP" or pos == "NNPS") and w_lower not in [x.lower() for x in named_entity_tokens]:
                named_entity_tokens.append(w)

    if len(important_tokens) < 2:
        important_tokens = [w for w in cleaned.split() if len(w) >= 3 and w.lower() not in stop][:4]

    if strict_relevance:
        # URL checks must be anchored by distinctive page details, not generic
        # topic words (for example, two unrelated earthquake stories).
        query_tokens = (named_entity_tokens[:4] + [
            token for token in important_tokens if token.lower() not in {item.lower() for item in named_entity_tokens[:4]}
        ])[:8]
    else:
        query_tokens = important_tokens[:5]
    query = ' '.join(query_tokens).strip()
    if not query:
        return {"corroborated": False, "evidence_status": "NO_RELEVANT_REPORTING", "trusted_relevant_source_count": 0, "trusted_relevant_result_count": 0, "match_count": 0, "sources": [], "query": ""}

    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            debunk_words = {
                'hoax', 'debunked', 'fake', 'false claim', 'fact check', 'conspiracy',
                'unfounded', 'myth', 'rumour', 'rumor', 'parody', 'satire', 'alleged',
                'allegedly', 'speculation', 'mystery', 'possible', 'ufo'
            }
            trusted_publishers = {
                "reuters", "associated press", "ap news", "bbc", "bbc news", "afp",
                "al jazeera", "the guardian", "new york times", "the new york times",
                "the washington post", "washington post", "cnn", "npr", "pbs",
                "the times of india", "times of india", "ndtv", "livemint",
                "the hindu", "indian express", "the indian express",
            }
            valid_sources = []
            for item in items[:8]:
                title = item.findtext("title", default="").strip()
                link = item.findtext("link", default="").strip()
                pub_date = item.findtext("pubDate", default="").strip()
                source_el = item.find("source")
                source_name = source_el.text if source_el is not None else ""
                if not source_name and "-" in title:
                    source_name = title.split("-")[-1].strip()

                description = BeautifulSoup(item.findtext("description", default=""), "html.parser").get_text(" ", strip=True)
                result_text = f"{title} {description}"
                result_tokens = {token.lower() for token in re.findall(r"\d+(?:\.\d+)?|[a-z]+", result_text.lower())}
                result_lower = result_text.lower()
                is_debunk = any(dw in result_lower for dw in debunk_words)
                normalized_source = re.sub(r"[^a-z0-9]+", " ", source_name.lower()).strip()
                trusted_source = normalized_source in trusted_publishers

                # Compare claim terms against the full result text, not the wording
                # or order of its headline. Allow simple singular/plural variation.
                matched_terms = [
                    q for q in query_tokens
                    if q.lower() in result_tokens
                    or (q.isalpha() and q.lower().endswith("s") and q.lower()[:-1] in result_tokens)
                    or (q.isalpha() and (q.lower() + "s") in result_tokens)
                ]
                matches = len(matched_terms)

                # Keep matching debunks and fact-checks as references too; dropping these
                # would hide precisely the sources a user needs to assess a disputed claim.
                overlap_ratio = matches / len(query_tokens) if query_tokens else 0.0
                if strict_relevance:
                    named_anchors = {token.lower() for token in named_entity_tokens[:4]}
                    has_named_anchor = not named_anchors or any(term.lower() in named_anchors for term in matched_terms)
                    is_relevant_match = matches >= min(3, len(query_tokens)) and overlap_ratio >= 0.6 and has_named_anchor
                else:
                    is_relevant_match = (matches >= 2 or (len(query_tokens) == 1 and matches == 1)) and (matches >= 3 or overlap_ratio >= 0.6)
                if is_relevant_match:
                    valid_sources.append({
                        "title": title,
                        "description": description[:500],
                        "link": link,
                        "source": source_name,
                        "pubDate": pub_date,
                        "is_fact_check": is_debunk,
                        "matched_terms": matched_terms,
                        "trusted_source": trusted_source,
                    })

            relevant_sources = {
                source["source"].lower()
                for source in valid_sources
                if source["trusted_source"]
            }
            trusted_source_count = len(relevant_sources)
            trusted_result_count = sum(1 for source in valid_sources if source["trusted_source"])
            is_relevant = len(valid_sources) >= 1
            return {
                "corroborated": is_relevant,
                "evidence_status": "RELEVANT_REPORTING_FOUND" if is_relevant else "NO_RELEVANT_REPORTING",
                "trusted_relevant_source_count": trusted_source_count,
                "trusted_relevant_result_count": trusted_result_count,
                "match_count": len(valid_sources),
                "sources": valid_sources,
                "query": query
            }
    except Exception as e:
        print(f"[WARN] Live news search notice: {e}")
        return {"corroborated": False, "evidence_status": "NO_RELEVANT_REPORTING", "trusted_relevant_source_count": 0, "trusted_relevant_result_count": 0, "match_count": 0, "sources": [], "query": query}

