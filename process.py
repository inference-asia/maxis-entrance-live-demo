from importlib import import_module
import os
from flask import Flask, render_template, Response
from camera_opencv import Camera
import cv2
from video_util import WebcamVideoStream
from base_camera import BaseCamera

app = Flask(__name__)

@app.route('/')
def index():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
    
