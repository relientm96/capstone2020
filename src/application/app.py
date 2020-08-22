# Activate eventlet
from eventlet import wsgi
import eventlet

from flask_cors import CORS
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from engineio.payload import Payload

import base64
import serverOpenPose as serverOP

import numpy as np
import time

# Number of clients
clients = 0

# Maximum payload length for more requests
Payload.max_decode_packets = 500

# Create flask application
app = Flask(__name__)
CORS(app)
# Activtate flask socket io
socketio = SocketIO(app)

'''
Socket Io App Routes
'''
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/group1-shard1of1.bin", methods=['GET'])
def getGroupShardBin():
    return send_from_directory("modeljs","group1-shard1of1.bin")

@app.route("/modeljs/<path:path>", methods=['GET'])
def sendModelJSON(path):
    return send_from_directory("modeljs", path)

@app.route("/posenet", methods=['GET'])
def posenetApp():
    return render_template("homepage.html")

'''
App Socket Io Handlers
'''
@socketio.on('posenet_keypoints')
def posenet_keypoints(keypoints):
    pass

@socketio.on('connect')
def test_connect():
    global clients
    clients += 1
    emit('clientcount', str(clients), broadcast=True)

@socketio.on('disconnect')
def test_disconnect():
    global clients
    if clients > 0:
        clients -= 1
    emit('clientcount', str(clients), broadcast=True)
    
@socketio.on('translateForMe')
def getKeypointsFromOpenPose(data_image):
    # Send to processing OpenCV Side
    keypoints = serverOP.returnKeypointsFlattened(data_image)
    # Send back to web browser via websocket
    # Check if given keypoints valid?
    if np.array_equal(keypoints,np.zeros(1)):
        # Something went wrong, no one is in frame
        emit('no_one_here')
    else:
        emit('keypoints_recv', str(keypoints.tolist()))
    
@socketio.on('imageSend')
def handleImageData(data_image):
    try: 
        # Send to processing OpenCV Side
        data, keypoints = serverOP.processFrames(data_image)
        # Decode base64
        data = data.decode('utf-8')
        # emit the frame back
        b64_src = 'data:image/jpg;base64,'
        stringData  = b64_src + data     
        # Send back keypoints to browser for prediction
        emit('keypoints_recv', str(keypoints.tolist()))
        # Send back image to browser for display
        emit('response_back', stringData)
    except Exception as e:
        print("Could not emit image back to client, error:", e)

if __name__ == '__main__':
    #socketio.run(app)
    wsgi.server(eventlet.listen(('', 5000)), app)
    #wsgi.server(eventlet.wrap_ssl(eventlet.listen(('', 5000)),certfile='cert.pem',keyfile='key.pem'), app)
