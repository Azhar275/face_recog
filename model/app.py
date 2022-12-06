from flask import Flask, render_template, Response, request, redirect, session, url_for
import cv2
from flask_bootstrap import Bootstrap
import streamlit as st
import streamlit.components.v1 as components

# bootstrap 4 collapse example

import videotester

app = Flask(__name__)
bootstrap = Bootstrap(app)
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        frame = videotester.generate_frames()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    components.html("<html><body><h1>Hello, World</h1></body></html>", width=200, height=200)
#     return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False)
