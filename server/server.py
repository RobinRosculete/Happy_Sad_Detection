# Import flask and datetime module for showing date and time
from flask import Flask, Response
import cv2 as cv


# Initializing flask app
app = Flask(__name__)

# Funciton For webcam streaming


def webcam():
    camera = cv.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if success:
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            camera.release()
# Route for send


@app.route('/webcam')
def webcam_diplay():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Running app
if __name__ == '__main__':
    app.run(debug=True)
