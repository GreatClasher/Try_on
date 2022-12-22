from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
app=Flask(__name__)

cap = cv2.VideoCapture(0)
add = "https://100.72.67.241:8080/video"
cap.open(add)

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
def gen_frames():
    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=5,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.99,
                                model_name='Shoe') as objectron:

            while True:
                success, frames = cap.read()  # read the camera frame
                if not success:
                    break
                else:
                    # Resize frame of video 
                    # image = cv2.resize(frames, (0,0),fx =0.25,fy=0.25)
                    image = cv2.resize(frames, (500,200), interpolation= cv2.INTER_AREA)
                    image.flags.writeable = False   
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = objectron.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detected_objects:
                        for detected_object in results.detected_objects:
                            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                            mp_drawing.draw_axis(image, detected_object.rotation,detected_object.translation)

                    ret, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)