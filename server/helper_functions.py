# This files consists of the helper functions for the server.py file

# Requierd Libraries
import cv2
import joblib
import numpy as np
import pywt
import json

# *************** Global Variables ***************
# Path to cascade classifier
face_cascade = cv2.CascadeClassifier(
    '../model/opncv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../model/opncv/haarcascade_eye.xml')


# Function For Webcam Streaming
def webcam():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if success:
            # Calling Face Detection to detect faces caught by webcam and draw a rectangle
            frame = face_detection(frame)
            results = classify_image(frame)
            print(results)
            results_str = "Results: " + str(results)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' +
                   b'Results: ' + results_str.encode() + b'\r\n')
        else:
            break  # Release the camera when there's an issue with capturing a frame
        # return results
    camera.release()  # Release the camera outside the loop


# Funciton used to clasify face reaction (Happy or Sad) given an image/frame


def classify_image(frame):
    image = get_cropped_image_if_eyes(frame)
    class_name_to_number, class_number_to_name, model = load_saved_model()
    result = []
    if image is not None and len(image) > 0:
        scalled_raw_img = cv2.resize(image, (32, 32))
        img_har = wavelet2d(image, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': class_number_to_name[model.predict(final)[0]],
            'class_probability': np.around(model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': class_name_to_number
        })

    return result

# Function to load save model and class_dictionary


def load_saved_model():
    with open("../model/class_dictionary.json", "r") as f:
        class_name_to_number = json.load(f)
        class_number_to_name = {v: k for k,
                                v in class_name_to_number.items()}

        with open("../model/saved_model.pkl", "rb") as f:
            model = joblib.load(f)
    return class_name_to_number, class_number_to_name, model


# Function used for face detection given frame


def face_detection(frame):
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


def wavelet2d(image, mode='haar', level=1):
    imageArray = image
    # converting to gray scale
    imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)

    # convertion values to float
    imageArray = np.float32(imageArray)
    imageArray /= 255

    # calculating coefficents
    coefficients = pywt.wavedec2(imageArray, mode, level=level)

    # process coefficients
    coefficients_H = list(coefficients)
    coefficients_H[0] *= 0

    # recunstructing image
    imageArray_H = pywt.waverec2(coefficients_H, mode)
    imageArray_H *= 255
    imageArray_H = np.uint8(imageArray_H)
    return imageArray_H

# function purpose to crop an image that contains both face and two eyes


def get_cropped_image_if_eyes(frame):
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (face_x, face_y, face_w, face_h) in faces:
        roi_color = frame[face_y:face_y+face_h, face_x:face_x+face_w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        if len(eyes) > 2:
            return roi_color
        else:
            return None
