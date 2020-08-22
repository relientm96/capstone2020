/*---------------------------------Pose Detection Program ---------------------------------*/

/**
    Inputs to PoseNet Model
  
    Image scale factor  — A number between 0.2 and 1. Defaults to 0.50. What to scale the image by before feeding it through the network. 
                        Set this number lower to scale down the image and increase the speed when feeding through the network at the cost of accuracy.
    
    Flip horizontal     — Defaults to false. If the poses should be flipped/mirrored horizontally. 
                        This should be set to true for videos where the video is by default flipped horizontally (i.e. a webcam), 
                        and you want the poses to be returned in the proper orientation.

    Output stride       — Must be 32, 16, or 8. Defaults to 16. Internally, this parameter affects the height and width of the layers in the neural network. 
                        At a high level, it affects the accuracy and speed of the pose estimation. 
                        The lower the value of the output stride the higher the accuracy but slower the speed, 
                        the higher the value the faster the speed but lower the accuracy.

    Maximum pose detections — An integer. Defaults to 5. The maximum number of poses to detect.

    Pose confidence score threshold — 0.0 to 1.0. Defaults to 0.5. At a high level, this controls the minimum confidence score of poses that are returned.
   
    Non-maximum suppression (NMS) radius — A number in pixels. At a high level, this controls the minimum distance between poses that are returned. 
                                        This value defaults to 20, which is probably fine for most cases. It should be increased/decreased as a way to filter out less accurate poses but only if tweaking the pose confidence score is not good enough.
*/

/**
 * CAMERA Functions
 * Referenced from posenet's github camera demo
 */

// Hardcoded Pixel Values
var videoWidth = 640;
var videoHeight = 480;

//Threshold to render poses
minConfidence = 0.2;
// Flag to enable/disable pose
displayPose = 1;
function enablePoints(){
    // User clicks this if he/she wants to see rendered pose points
    displayPose ^= 1;
}
// Socketio obj
var socket = io();
// Translated word placeholder
var word = "Hello"

// Mobile Support
const mobile = /Android/i.test(navigator.userAgent) || /iPhone|iPad|iPod/i.test(navigator.userAgent);
// Asynchronous function to set camera configurations
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }
    const video = document.getElementById('videoElement');
    video.width = videoWidth;
    video.height = videoHeight;
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

// Async function to load camera permissions and settings
// Referenced from posenet's camera.js demo code
async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

// Reads in socket io message to acknowledge connection is made
socket.on('resp', function(responseText){
    document.getElementById('camSetup').innerHTML = responseText;
});

//Reads translated result
socket.on('response_back', function(translated_word){
    word = translated_word
});

function drawLine(sourceX, sourceY, destX, destY){
    const canvas = document.getElementById('output')
    let ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(sourceX, sourceY);
    ctx.lineTo(destX, destY);
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
}

// Function to draw a line between two points
function drawAllSkeleton(array){
    const pairs = {
        0 : 1,
        1 : 3,
        2 : 0,
        4 : 2,
        5 : 7,
        6 : 8,
        7 : 9,
        8 : 10,
        12: 6,
        11: 5
    }
    for (i = 0; i < array.length; i++) {
        // Only redraw them if score higher than threshold score
        // threshold score here is set by slider by user
        x = array[i].position['x'];
        y = array[i].position['y'];
        if (pairs[i]) {
            drawLine(x, y, array[pairs[i]].position['x'], array[pairs[i]].position['y'])
        }
    }
    // Complete the skeleton
    drawLine(array[11].position['x'], array[11].position['y'], array[12].position['x'], array[12].position['y'])
    drawLine(array[5].position['x'], array[5].position['y'], array[6].position['x'], array[6].position['y'])
}

// Detects poses in real time with a WebCam Stream
// Referenced from posenet's camera.js demo code
function detectPoseInRealTime(video, net) {
    // Canvas details
    var canvas = document.getElementById('output')
    let ctx = canvas.getContext('2d');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    async function getPose() {
        const pose = net.estimatePoses(video, {
                flipHorizontal: false,
                maxDetections: 1,
                scoreThreshold: minConfidence,
                nmsRadius: 20
            }).then(function(results) {
                // Redraw Key Points for each new rendered frame
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
                if (displayPose){
                    for (i = 0; i < results[0].keypoints.length; i++) {
                        x = results[0].keypoints[i].position['x'];
                        y = results[0].keypoints[i].position['y'];
                        ctx.fillStyle = "#FF0000";
                        ctx.fillRect(x, y, 10, 10);
                    }
                    drawAllSkeleton(results[0].keypoints)
                }
            })
            // Looping frame rendering continuosly
        window.requestAnimationFrame(getPose);
    }
    getPose();
}

//  Function to run entire thing
async function run() {
    const net = await posenet.load();
    let video;
    try {
        video = await loadVideo();  
        document.getElementById('camSetup').textContent = "";
    } catch (e) {
        document.getElementById('camSetup').textContent =
            'this browser does not support video capture,' +
            'or this device does not have a camera';
        throw e;
    }
    detectPoseInRealTime(video, net);
}

// Web media api enable
navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

// Start program
run();


