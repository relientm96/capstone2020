import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import numpy as np
import sys

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

# Socket io
import socketio
sio = socketio.AsyncServer(async_mode='aiohttp')

# Custom Imports
from openpose import initializeOpenPose
from utils import removeConfidenceAndShapeAsNumpy

# Global References for OpenPose object
op = None
opWrapper = None
keypoints = None

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        if "openpose" in self.transform:
            img = frame.to_ndarray(format="bgr24")

            datum = op.Datum()
            datum.cvInputData = img
            opWrapper.emplaceAndPop([datum])
            image = datum.cvOutputData

            # Get keypoints using numpy slicing
            keypoints = removeConfidenceAndShapeAsNumpy(datum)
            # Send keypoints using socketion once frame processed
            await sio.emit('keypoints', str(keypoints.tolist()))
            
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        else:
            return frame

#--------------------------------------------------------------------------
# Defined App Routes

# index               : index home page rendering
# rolling_window_load : loads rolling window module (javascript)
# client_load         : loads webrtc client (javascript)
# offer               : loads handler for webrtc channel
# load_model_json     : loads model json (converted keras model in tf.js)
# load_model_shard    : loads shards of model.json
# main_js_load        : loads main js file
# on_shutdown         : clean up before app closes
#--------------------------------------------------------------------------
async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def main_js_load(request):
    content = open(os.path.join(ROOT, "main.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def rolling_window_load(request):
    content = open(os.path.join(ROOT, "rollingwindow.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def client_load(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def load_model_json(request):
    content = open(os.path.join(ROOT, "modeljs/model.json"), "r").read()
    return web.Response(content_type="application/json", text=content)

async def load_model_shard(request):
    content = open(os.path.join(ROOT, "modeljs/group1-shard1of1.bin"), "rb").read()
    return web.Response(content_type="application/octet-stream", body=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":

    # Initialize OpenPose
    op, opWrapper = initializeOpenPose()
    print("OpenPose Initialized!")

    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for HTTP server (default: 5000)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", client_load)
    app.router.add_get("/rollingwindow.js", rolling_window_load)
    app.router.add_get("/model.json", load_model_json)
    app.router.add_get("/group1-shard1of1.bin", load_model_shard)
    app.router.add_get("/main.js", main_js_load)
    app.router.add_post("/offer", offer)
    
    sio.attach(app)
    
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
