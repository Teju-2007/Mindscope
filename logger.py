from datetime import datetime

def log_emotion(source, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("mood_log.csv", "a") as file:
        file.write(f"{timestamp},{source},{emotion}\n")
