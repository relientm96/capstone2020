from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import base64
import serverOpenPose as serverOP

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@socketio.on('connect')
def test_connect():
    # serverOP.initOP()
    emit('resp','You are now connected to OpenPose Server!')

@socketio.on('imageSend')
def handleImageData(data_image):
    # Send to processing OpenCV Side
    data = serverOP.processFrames(data_image).decode('utf-8')
    # emit the frame back
    b64_src = 'data:image/jpg;base64,'
    stringData  = b64_src + data     
    emit('response_back', stringData)

if __name__ == '__main__':
    socketio.run(app)