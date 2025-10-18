from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_this_with_a_secret_in_production'
socketio = SocketIO(app, cors_allowed_origins="*")

session_data = {"emotions": [], "summary": {}}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit('update_data', session_data)

@socketio.on('video_frame')
def handle_video_frame(data_image):
    try:
        sbuf = base64.b64decode(data_image.split(',')[1])
        nparr = np.frombuffer(sbuf, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = None
        # Support for DeepFace returning either a dict or list
        if isinstance(result, list) and result:
            dominant_emotion = result[0]['dominant_emotion']
        elif isinstance(result, dict):
            dominant_emotion = result['dominant_emotion']
        if dominant_emotion:
            emotion_mapping = {
                'happy': 'Happy',
                'sad': 'Sad',
                'neutral': 'Focused',
                'angry': 'Frustrated',
                'fear': 'Surprised',
                'disgust': 'Bored',
                'surprise': 'Surprised'
            }
            mapped = emotion_mapping.get(dominant_emotion.lower(), 'Neutral')
            timestamp = len(session_data["emotions"])
            session_data["emotions"].append({"time": timestamp, "emotion": mapped})
            summary = session_data["summary"]
            summary[mapped] = summary.get(mapped, 0) + 1
            socketio.emit('update_data', session_data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
