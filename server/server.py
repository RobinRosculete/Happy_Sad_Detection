# Import flask and datetime module for showing date and time
from flask import Flask, Response
from helper_functions import webcam


# Initializing flask app
app = Flask(__name__)

# Route for send


@app.route('/webcam')
def webcam_diplay():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Running app
if __name__ == '__main__':
    app.run(debug=True)
