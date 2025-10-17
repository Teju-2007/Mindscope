import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
import time
from transformers import pipeline

from voice_input import get_voice_text
from ml_predictor import predict_mental_state
from face_emotion import detect_face_emotion
from logger import log_emotion

# App setup
st.set_page_config(page_title="MindScope", page_icon="🧠", layout="centered")
st.title("🧠 MindScope: Your AI Wellness Companion")

# Voice feedback
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Suggestions
def get_suggestion(emotion):
    suggestions = {
        "sadness": "Try journaling your thoughts or listening to calming music.",
        "joy": "Celebrate your moment! Share it with someone you love.",
        "anger": "Take a deep breath. A short walk or quiet time can help.",
        "fear": "Ground yourself with a breathing exercise or talk to a trusted friend.",
        "surprise": "Reflect on what surprised you — is it something exciting or stressful?",
        "neutral": "Stay mindful. A short meditation can help maintain balance.",
        "disgust": "Step away from the trigger. Clean space and fresh air can reset your mood."
    }
    return suggestions.get(emotion.lower(), "Take a moment to reflect. You’re doing great.")

# Journaling prompt
def get_journaling_prompt(emotion):
    prompts = {
        "sadness": "What’s been weighing on your heart lately? Write freely.",
        "joy": "What brought you joy today? How can you invite more of it?",
        "anger": "What triggered your anger? What would help you release it?",
        "fear": "What are you afraid of right now? What might help you feel safe?",
        "surprise": "What unexpected thing happened today? How did it make you feel?",
        "neutral": "How are you feeling overall? What’s been on your mind?",
        "disgust": "What situation or thought felt off today? Why do you think it affected you?"
    }
    return prompts.get(emotion.lower(), "Write about anything that’s on your mind. Let it flow.")

# Emotion detection
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotions_with_scores(text):
    results = emotion_classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotions = sorted_results[:3]
    return [(e['label'].lower(), round(e['score'] * 100, 2)) for e in top_emotions]

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_message = st.chat_input("How are you feeling today?")

if user_message:
    st.session_state.chat_history.append(("user", user_message))
    top_emotions = detect_emotions_with_scores(user_message)
    primary_emotion = top_emotions[0][0]
    st.session_state["emotion"] = primary_emotion
    st.session_state["primary_emotion"] = primary_emotion
    log_emotion("Chat", primary_emotion)

    response_text = f"You seem to be feeling {primary_emotion}. Here's your emotional breakdown:"
    st.session_state.chat_history.append(("ai", response_text))

    for role, message in st.session_state.chat_history:
        st.chat_message(role).write(message)

    labels = [e[0].capitalize() for e in top_emotions]
    scores = [e[1] for e in top_emotions]
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color="lightcoral")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Top 3 Emotions")
    st.pyplot(fig)

    suggestion = get_suggestion(primary_emotion)
    st.chat_message("assistant").write(f"💡 Suggestion: {suggestion}")
    speak(f"You seem to be feeling {primary_emotion}. {suggestion}")

    st.markdown("### What would you like to do next?")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📝 Journal"):
            prompt = get_journaling_prompt(primary_emotion)
            st.subheader("📝 Journaling Prompt")
            st.write(f"**Prompt:** {prompt}")
            journal_entry = st.text_area("Write your thoughts here...")
            if st.button("Save Entry"):
                with open("journal_log.txt", "a") as f:
                    f.write(f"\n---\nEmotion: {primary_emotion}\nPrompt: {prompt}\nEntry: {journal_entry}\n")
                st.success("Your journal entry has been saved.")

    with col2:
        if st.button("🧘 Breathe"):
            st.subheader("🧘 Guided Breathing")
            for cycle in range(3):
                st.write("🫁 Inhale...")
                time.sleep(4)
                st.write("😌 Hold...")
                time.sleep(4)
                st.write("🌬️ Exhale...")
                time.sleep(4)
                st.write("---")
            st.success("You’ve completed 3 calming breaths.")

    with col3:
        if st.button("📞 Get Support"):
            st.subheader("📞 Mental Health Resources")
            st.write("""
            If you're feeling overwhelmed, you're not alone. Here are some support options:
            - [iCall India](https://icallhelpline.org): Free mental health support
            - [AASRA](http://www.aasra.info): 24/7 helpline: +91-9820466726
            - [YourDOST](https://yourdost.com): Online counseling platform
            """)

# Sidebar navigation
menu = st.sidebar.radio("Navigate", ["🏠 Home", "💬 Emotion Detection", "📊 Mood Tracker", "🧘 Wellness Tools", "📓 Journaling"])

# Home Page
if menu == "🏠 Home":
    st.subheader("Your AI-powered mental wellness companion")
    st.write("""
    MindScope helps you understand and support your emotional health using AI.

    💬 Detect emotions from text, voice, and face  
    📊 Track your mood over time  
    🧘 Practice calming exercises  
    📓 Reflect through guided journaling  
    🔊 Receive personalized suggestions and voice feedback

    Built with care for the Hackathon — by Hacktrack Squad 
    """)

# Emotion Detection
if menu == "💬 Emotion Detection":
    st.subheader("💬 Text Emotion Detection")
    user_input = st.text_area("Type how you're feeling today:")

    if st.button("Analyze"):
        if user_input:
            top_emotions = detect_emotions_with_scores(user_input)
            primary_emotion = top_emotions[0][0]
            st.session_state["emotion"] = primary_emotion
            st.session_state["primary_emotion"] = primary_emotion
            st.success(f"Detected Emotion: {primary_emotion}")
            log_emotion("Text", primary_emotion)
            suggestion = get_suggestion(primary_emotion)
            st.info(f"💡 Suggestion: {suggestion}")
            speak(f"You seem to be feeling {primary_emotion}. {suggestion}")

    st.subheader("🎤 Voice Emotion Detection")
    if st.button("Use Voice"):
        voice_text = get_voice_text()
        st.write(f"You said: {voice_text}")
        if voice_text:
            top_emotions = detect_emotions_with_scores(voice_text)
            primary_emotion = top_emotions[0][0]
            st.session_state["emotion"] = primary_emotion
            st.session_state["primary_emotion"] = primary_emotion
            st.success(f"Detected Emotion: {primary_emotion}")
            log_emotion("Voice", primary_emotion)
            suggestion = get_suggestion(primary_emotion)
            st.info(f"💡 Suggestion: {suggestion}")
            speak(f"You seem to be feeling {primary_emotion}. {suggestion}")

    st.subheader("🧑‍🦰 Facial Emotion Detection")
    if st.button("Use Facial Detection"):
        emotion = detect_face_emotion()
        st.session_state["emotion"] = emotion
        st.session_state["primary_emotion"] = emotion
        st.success(f"Facial Emotion Detected: {emotion}")
        log_emotion("Face", emotion)
        suggestion = get_suggestion(emotion)
        st.info(f"💡 Suggestion: {suggestion}")
        speak(f"You seem to be feeling {emotion}. {suggestion}")

# Journaling
if menu == "📓 Journaling":
    if "primary_emotion" in st.session_state:
        prompt = get_journaling_prompt(st.session_state["primary_emotion"])
        st.subheader("📝 Journaling Prompt")
        st.write(f"**Prompt:** {prompt}")
        journal_entry = st.text_area("Write your thoughts here...")
        if st.button("Save Entry"):
            with open("journal_log.txt", "a") as f:
                                f.write(f"\n---\nEmotion: {st.session_state['primary_emotion']}\nPrompt: {prompt}\nEntry: {journal_entry}\n")
            st.success("Your journal entry has been saved.")
    else:
        st.info("Please analyze an emotion first to receive a personalized journaling prompt.")

if menu == "🧘 Wellness Tools":
    st.subheader("🧘 Guided Breathing")
    if st.button("Start Breathing Exercise"):
        st.write("🌬️ Breathe with Me")
        for cycle in range(3):
            st.write("🫁 Inhale...")
            time.sleep(4)
            st.write("😌 Hold...")
            time.sleep(4)
            st.write("🌬️ Exhale...")
            time.sleep(4)
            st.write("---")
        st.success("You’ve completed 3 calming breaths.")

# 📊 Mood Tracker
if menu == "📊 Mood Tracker":
    st.subheader("📊 Emotion Frequency")
    try:
        df = pd.read_csv("mood_log.csv", names=["Timestamp", "Source", "Emotion"])
        emotion_counts = df["Emotion"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(emotion_counts.index, emotion_counts.values, color="skyblue")
        ax.set_title("Emotion Frequency")
        ax.set_ylabel("Count")
        ax.set_xlabel("Emotion")
        st.pyplot(fig)

        st.subheader("📅 Daily Mood Trends")
        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
        daily_trend = df.groupby(["Date", "Emotion"]).size().unstack(fill_value=0)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        daily_trend.plot(kind="line", ax=ax2, marker="o")
        ax2.set_title("Daily Emotion Trends")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Date")
        ax2.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig2)

    except FileNotFoundError:
        st.info("No mood data found yet. Start analyzing emotions to build your tracker.")
