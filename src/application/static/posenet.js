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
minConfidence = 0.4;

// Callback to check which mode to do
handEnabled = false;
poseEnabled = true;

// State to render
var render_state = 0;

window.onload = function(){
    document.getElementById("buttonCtrl").innerHTML = "Enable Hand"
    document.getElementById("buttonCtrl").className = "waves-effect waves-light btn red";
}

function handleButton(){

    render_state += 1;

    if (render_state > 2){
        render_state = 0;
    }
    
    if (render_state == 0){
        document.getElementById("buttonCtrl").innerHTML = "Enable Hand"
        document.getElementById("buttonCtrl").className = "waves-effect waves-light btn red";
        document.getElementById("camSetup").innerHTML = "Now Doing Pose Estimation!";
    }
    else if (render_state == 1) {
        document.getElementById("buttonCtrl").innerHTML = "Enable Both"
        document.getElementById("buttonCtrl").className = "waves-effect waves-light btn purple";
        document.getElementById("camSetup").innerHTML = "Now Doing Hand Estimation! (Keep Hand In frame to process video)";
    }
    else {
        document.getElementById("buttonCtrl").innerHTML = "Enable Pose"
        document.getElementById("buttonCtrl").className = "waves-effect waves-light btn orange";
        document.getElementById("camSetup").innerHTML = "Now Doing Both Pose and Hand Estimation! (Keep Hand In frame to process video)";
    }
}

const state = {
    backend: 'webgl'
};

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
    drawLine(array[0].position['x'], array[0].position['y'], array[2].position['x'], array[2].position['y'])
    drawLine(array[11].position['x'], array[11].position['y'], array[12].position['x'], array[12].position['y'])
    drawLine(array[5].position['x'], array[5].position['y'], array[6].position['x'], array[6].position['y'])
}

// Detects poses in real time with a WebCam Stream
// Referenced from posenet's camera.js demo code
function detectPoseInRealTime(video, net, model) {
    // Canvas details
    var canvas = document.getElementById('output')
    var ctx = canvas.getContext('2d');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    async function getPose() {
                
        if (render_state == 0){
            const posepredictions = await net.estimatePoses(video, {
                flipHorizontal: false,
                maxDetections: 1,
                scoreThreshold: minConfidence,
                nmsRadius: 20
            });
            if (posepredictions.length > 0){
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
                const results = posepredictions[0].keypoints;
                for (i = 0; i < results.length; i++) {
                    x = results[i].position['x'];
                    y = results[i].position['y'];
                    ctx.fillStyle = "#FF0000";
                    ctx.fillRect(x, y, 10, 10);
                    ctx.fill();
                }
                drawAllSkeleton(results);
            }
        }
        
        if (render_state == 1){
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const handpredictions = await model.estimateHands(video);
            if (handpredictions.length > 0)  {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
                const keypoints = handpredictions[0].landmarks;
                for (i = 0; i < keypoints.length; i++) {
                    const [x, y] = keypoints[i];
                    ctx.fillStyle = "#00FFFF";
                    ctx.fillRect(x, y, 10, 10);
                    ctx.fill();
                }
            }    
        }

        if (render_state == 2){
            const handpredictions = await model.estimateHands(video);
            const posepredictions = await net.estimatePoses(video, {
                flipHorizontal: false,
                maxDetections: 1,
                scoreThreshold: minConfidence,
                nmsRadius: 20
            });
            if ((handpredictions.length > 0) && (posepredictions.length > 0)) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();
                const keypoints = handpredictions[0].landmarks;
                for (i = 0; i < keypoints.length; i++) {
                    const [x, y] = keypoints[i];
                    ctx.fillStyle = "#00FFFF";
                    ctx.fillRect(x, y, 10, 10);
                    ctx.fill();
                }
                const results = posepredictions[0].keypoints;
                for (i = 0; i < results.length; i++) {
                    x = results[i].position['x'];
                    y = results[i].position['y'];
                    ctx.fillStyle = "#FF0000";
                    ctx.fillRect(x, y, 10, 10);
                    ctx.fill();
                }
                drawAllSkeleton(results);
            }    
        }


        /*
       const pose = net.estimatePoses(video, {
                    flipHorizontal: false,
                    maxDetections: 1,
                    scoreThreshold: minConfidence,
                    nmsRadius: 20
                }).then(function(results) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.save();
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    ctx.restore();
                    // Redraw Key Points for each new rendered frame
                    for (i = 0; i < results[0].keypoints.length; i++) {
                        x = results[0].keypoints[i].position['x'];
                        y = results[0].keypoints[i].position['y'];
                        ctx.fillStyle = "#FF0000";
                        ctx.fillRect(x, y, 10, 10);
                        ctx.fill();
                    }
                    drawAllSkeleton(results[0].keypoints)
                })
        */
        // Looping frame rendering continuosly
        window.requestAnimationFrame(getPose);
    }
    getPose();
}

//  Function to run entire thing
async function run() {
    await tf.setBackend(state.backend);
    const net   = await posenet.load();
    const model = await handpose.load();
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
    detectPoseInRealTime(video, net, model);
}

// Web media api enable
navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

// Start program
run();


