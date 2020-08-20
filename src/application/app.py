# Activate eventlet
from eventlet import wsgi
import eventlet

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from engineio.payload import Payload

import base64
import serverOpenPose as serverOP

import time

import pprint as pp

# Maximum payload length for more requests
Payload.max_decode_packets = 500

# Create flask application
app = Flask(__name__)
# Activtate flask socket io
socketio = SocketIO(app)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/posenet", methods=['GET'])
def posenetApp():
    return render_template("homepage.html")

@app.route("/openpose", methods=['GET'])
def openPoseRender():
    return render_template("old.html")

@socketio.on('posenet_keypoints')
def posenet_keypoints(keypoints):
    #print(keypoints)
    emit('response_back', "Help")

@socketio.on('connect')
def test_connect():
    # serverOP.initOP()
    emit('resp','You are now connected to OpenPose Server!')

@socketio.on('translateForMe')
def pushImageToModel(data_image):
    # Send to processing OpenCV Side
    word = serverOP.translateWord(data_image)
    # Send back to web browser via websocket
    emit('output_word', word)

@socketio.on('imageSend')
def handleImageData(data_image):
    try: 
        # Send to processing OpenCV Side
        data = serverOP.processFrames(data_image).decode('utf-8')
        # emit the frame back
        b64_src = 'data:image/jpg;base64,'
        stringData  = b64_src + data     
        # Send back to web browser via websocket
        emit('response_back', stringData)
    except Exception as e:
        print("Could not emit image back to client, error:", e)

if __name__ == '__main__':
    #socketio.run(app)
    #wsgi.server(eventlet.listen(('', 5000)), app)
    wsgi.server(eventlet.wrap_ssl(eventlet.listen(('', 5000)),certfile='cert.pem',keyfile='key.pem'), app)
