/**
 * WebRTC Demo using Python Aiortc
 */

var videoWidth = 800;
var videoHeight = 600;

/*
// get DOM elements
var iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');
*/

// peer connection
var pc = null;

// data channel
var dc = null;

// Check if firefox/chrome
// Ref: https://stackoverflow.com/questions/9847580/how-to-detect-safari-chrome-ie-firefox-and-opera-browser
// Firefox 1.0+
var isFirefox = typeof InstallTrigger !== 'undefined';
console.log("Is Firefox", isFirefox);

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    // Trying without ICE
    config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];

    pc = new RTCPeerConnection(config);

    /*
    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    //iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    //iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    //signalingLog.textContent = pc.signalingState;
    */

    // connect audio / video
    pc.addEventListener('track', function(evt) {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        console.log("STARTING ICE");
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        console.log("Setting Up Video Media Stream");
        var offer = pc.localDescription;
        var codec = 'VP8'; //VP8 or H264
        offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: 'openpose'
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        console.log("Returning JSON", response)
        return response.json();
    }).then(function(answer) {
        console.log("Receiving an answer, done!", answer)
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function start() {

    document.getElementById('StatusOfVideo').innerHTML = "Please Wait For WebRTC To Setup (might take awhile)";
    // Create peer connections
    pc = createPeerConnection();

    // Mobile Support
    const mobile =/Android/i.test(navigator.userAgent) || /iPhone|iPad|iPod/i.test(navigator.userAgent);

    // Web media api enable
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Browser API navigator.mediaDevices.getUserMedia not available");
        throw new Error(
            "Browser API navigator.mediaDevices.getUserMedia not available"
        );
    }

    // Set up video streaming
    if (isFirefox){
        // Firefox specific
        var constraints = {
            audio: false,
            video : {
                width : mobile? undefined: videoWidth,
                height: mobile? undefined: videoHeight
            }
        };
    }
    else {
        // Assume chrome, can apply maxFrameRate setting
        var constraints = {
            audio: false,
            video : {
                width : mobile? undefined: videoWidth,
                height: mobile? undefined: videoHeight,
                frameRate : {
                    max : 5
                }
            }
        }
    }

    if (constraints.video) {
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            stream.getTracks().forEach(function(track) {
                if (isFirefox){
                    const sender = pc.addTrack(track, stream);
                    /**
                     * Firefox weird errors when limiting frame rate
                     * Refer to threads:
                     * https://stackoverflow.com/questions/35516416/firefox-frame-rate-max-constraint
                     * https://stackoverflow.com/questions/57400849/webrtc-change-bandwidth-receiving-invalidmodificationerror-on-chrome
                     * https://stackoverflow.com/questions/63303684/use-high-resolution-local-video-but-limit-video-size-in-webrtc-connection/63331550#63331550
                     */
                    const params = sender.getParameters();
                    console.log(params);
                    if (!params.encodings) {
                        params.encodings = [{}];
                    }
                    params.encodings[0].maxFramerate = 5;
                    sender.setParameters(params);
                    console.log(params);
                }
                else {
                    console.log("Chrome Only")
                    pc.addTrack(track, stream);
                }
            });
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }

    /*
    // Setting up datachannel to get keypoints
    const parameters = {
        "ordered": true
    }
    dc = pc.createDataChannel('chat', parameters);
    dc.onclose = function() {
        console.log("Closing Data Channel");
    };
    dc.onopen = function() {
        var message = 'Hello From Client!';
        dc.send(message);
    };
    dc.onmessage = function(evt) {
        console.log(evt.data);
    };
    */
}

function stop() {

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);

}


function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

export {start, stop}

