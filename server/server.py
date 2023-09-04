# Import flask and datetime module for showing date and time
from flask import Flask, Response
import cv2
import pickle


# Initializing flask app
app = Flask(__name__)


# Function used for face detection given frame
def face_detection(frame):
    # Path to cascade classifier
    face_cascade = cv2.CascadeClassifier(
        '../model/opncv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../model/opncv/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(
        '../model/opncv/haarcascade_smile.xml')
    # Convert the frame to grayscale for face and eye detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ******************************** Face detection ********************************
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Iterate over detected faces and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ******************************** Eye detection ********************************
        # Region of interest (ROI) for eyes within the detected face
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(50, 50))

        # Iterate over detected eyes and draw rectangles
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (255, 0, 0), 2)

    return frame


# Function For Webcam Streaming
def webcam():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if success:
            frame = face_detection(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break  # Release the camera when there's an issue with capturing a frame

    camera.release()  # Release the camera outside the loop

# Route for send


@app.route('/webcam')
def webcam_diplay():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Running app
if __name__ == '__main__':
    app.run(debug=True)
