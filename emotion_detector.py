
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotions_with_scores(text):
    results = emotion_classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotions = sorted_results[:3]
    return [(e['label'].lower(), round(e['score'] * 100, 2)) for e in top_emotions]
