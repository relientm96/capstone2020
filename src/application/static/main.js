// Hardcoded Pixel Values
var videoWidth = 620;
var videoHeight = 480;
// Canvas details
var canvas = document.getElementById("canvasOutput");
canvas.width = videoWidth;
canvas.height = videoHeight;
// Image render details
const imageOutput = document.getElementById("imageOut");
// Socketio obj
const socket = io();
// Translated word placeholder
var word = "Hello & Welcome!";
// Flag to indicate if we want to enable skeleton rendering
var isSkeleton = 0;
document.getElementById("buttonControl").innerHTML = "Enable Skeleton";
/**
 * Frame control variables
 * Video is rendered on canvas every 1000/fps milliseconds
 * The "frame" is transmitted to the server every frameThresholdCount times of fps
 */
// Control FPS (how often to transmit frames to openpose server)
const fps = 15;
// Control how often you are sending images to openpose server
const frameThresholdCount = 3;
// Counter for frames
var frameCount = 0;
// Count number of clients connected
var clientNumber = 0;
// Flag to indicate if total number of clients connected is not too many
var isOneClient = 1;
// Max number of clients
const maxClients = 2;
// Model
var model;
// Signs that define output
dictOfSigns = {
  0: "ambulance",
  1: "help",
  2: "hospital",
  3: "pain",
};

// Define Rolling Window
class RollingWindow {
    constructor(windowWidth, numbJoints, points){
        this.windowWidth = windowWidth;
        this.numbJoints  = numbJoints;
        this.points      = points;
    }
    printPoints(){
        console.log(this.points);
    }
    getPoints(){
        return this.points;
    }
    addPoint(incomingKp){
        if (incomingKp.length != this.numbJoints){
            console.log("Error! Need to have same length as number of joints: " + this.numbJoints)
            return false;
        }
        // Shift register
        // Remove the oldest from the first index
        // Add the most recent entry, enters from the last index        
        this.points.shift()
        this.points.push(incomingKp)
        return true;
    }
    shape(){
        // Returns the rows, columns of the rolling window
        return [ this.points.length, this.points[0].length ];
    }
}
// Rolling window variables
const numbJoints  = 98; 
const windowWidth = 75;
var rw;

// Mobile Support
const mobile =
  /Android/i.test(navigator.userAgent) ||
  /iPhone|iPad|iPod/i.test(navigator.userAgent);
// Asynchronous function to set camera configurations
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }
  const video = document.getElementById("videoElement");
  video.width = videoWidth;
  video.height = videoHeight;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
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

// Button will control if we want to enable skeleton rendering or not
function enableSkeleton() {
  isSkeleton ^= 1;
  btnElem = document.getElementById("buttonControl");
  btnElem.className = "";
  if (isSkeleton) {
    canvas.style.display = "none";
    imageOutput.style.display = "";
    btnElem.className = "waves-effect waves-light btn red";
    btnElem.innerHTML = "Disable Skeleton";
  } else {
    canvas.style.display = "";
    imageOutput.style.display = "none";
    btnElem.className = "waves-effect waves-light btn blue";
    btnElem.innerHTML = "Enable Skeleton";
  }
}

function zeros(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; ++i) {
        array.push(dimensions.length == 1 ? 0 : zeros(dimensions.slice(1)));
    }
    return array;
}

// Async function to load camera permissions and settings
async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

// Detects poses in real time with a WebCam Stream
function detectPoseInRealTime(video) {
  // Canvas used
  //const canvas = document.getElementById('canvasOutput');
  const ctx = canvas.getContext("2d");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  async function getPose() {
    if (isOneClient) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      if (isSkeleton == 0) {
        ctx.font = "30px Arial";
        ctx.fillStyle = "white";
        ctx.fillText(word, 20, 50);
      }
      if (frameCount > frameThresholdCount) {
        // Transmit image to backend for processing
        let format = "image/jpeg";
        var data = canvas.toDataURL(format, 0.35); // lossy compression at 35%
        data = data.replace("data:" + format + ";base64,", "");
        if (isSkeleton) {
          socket.emit("imageSend", data);
          if (imageOutput.src === "") {
            document.getElementById("status").innerHTML =
              "Loading Images from Server, please wait...";
          } else {
            document.getElementById("status").innerHTML =
              "Streaming Images from OpenPose Server";
          }
        } else {
          document.getElementById("status").innerHTML = "Translating Sign!";
          socket.emit("translateForMe", data);
        }
        frameCount = 0;
      }
      frameCount += 1;
      setTimeout(() => {
        window.requestAnimationFrame(getPose);
      }, 1000 / fps);
    } else {
      if (clientNumber == 1) {
        isOneClient = 1;
      }
    }
  }
  getPose();
}

// Socket io Callback functions
socket.on("connect", function () {
  console.log("Connected...!", socket.connected);
});
socket.on("clientcount", function (responseText) {
  document.getElementById("clientStatus").innerHTML =
    "Current number of signers: " + responseText;
  clientNumber = parseInt(responseText);
  console.log(clientNumber);
  if (clientNumber > maxClients) {
    document.getElementById("status").innerHTML =
      "We only limit our app usage to one person at a time! Sorry!";
    isOneClient = 0;
  } else {
    document.getElementById("status").innerHTML = "Please Refresh!";
    isOneClient = 1;
  }
});
socket.on("output_word", function (translatedWord) {
  word = translatedWord;
});
socket.on("response_back", function (image) {
  // This function is called whenever server finishes processing the frame
  imageOutput.src = image;
});
socket.on("keypoints_recv", function (keypoints) {
  // Receives openpose keypoints after rendering from backend
  var kp_arr = JSON.parse(keypoints)
  // Append point to rolling window
  rw.addPoint(kp_arr);
});

//  Function to run entire thing
async function run() {
  let video;
  rw = new RollingWindow(windowWidth, numbJoints, zeros([windowWidth, numbJoints]))
  try {
    video = await loadVideo();
    // model = await tf.loadLayersModel(window.location.href + 'getGestureModel', strict=false)
    model = await tf.loadLayersModel(
      window.location.href + "modeljs/model.json"
    );
  } catch (e) {
    document.getElementById("status").textContent =
      "this browser does not support video capture," +
      "or this device does not have a camera";
    throw e;
  }
  detectPoseInRealTime(video);
}
// Web media api enable
navigator.getUserMedia =
  navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia;
// Start program
run();
