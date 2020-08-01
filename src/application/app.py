# Activate eventlet
from eventlet import wsgi
import eventlet

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from engineio.payload import Payload

import base64
import serverOpenPose as serverOP

import time

# Maximum payload length for more requests
Payload.max_decode_packets = 500

# Create flask application
app = Flask(__name__)
# Activtate flask socket io
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
    try: 
        # Get starting time
        start = time.process_time()

        # Send to processing OpenCV Side
        data = serverOP.processFrames(data_image).decode('utf-8')
        # emit the frame back
        b64_src = 'data:image/jpg;base64,'
        stringData  = b64_src + data     
        # Send back to web browser via websocket
        emit('response_back', stringData)

        # Check time after emitting back (our "fps" check)
        end = time.process_time()
        print("Time to process frame", end-start, "seconds")
        
    except Exception as e:
        print("Could not emit image back to client, error:", e)

if __name__ == '__main__':
    #socketio.run(app)
    wsgi.server(eventlet.listen(('', 5000)), app)
