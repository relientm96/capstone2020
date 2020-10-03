/**
 * Main File 
 */

import {start, stop} from './client.js'
import {RollingWindow} from './rollingwindow.js'

// Rolling window variables
const numbJoints  = 98; 
const windowWidth = 75;
// Rolling Window
var rw;
// Model reference object
var model;
// Enumerated Signs that define output
const dictOfSigns = {
    0: "ambulance",
    1: "help",
    2: "hospital",
    3: "pain",
};
// Socketio obj
const socket = io({transports: ['websocket']});
// Probability and sign variable
var sign = document.getElementById("currentsign");
var prob = document.getElementById("probability");

socket.on("keypoints", function(keypoints){
    // Receives openpose keypoints after rendering from backend
    var kp_arr = JSON.parse(keypoints)
    // Append point to rolling window
    if (rw.addPoint(kp_arr)){
        // Do prediction once we have new updated rolling window from this keypoint
        var keypoint_shaped = tf.tensor3d(rw.getPoints().flat(), [1, windowWidth, numbJoints])
        var pred_result = model.predict(keypoint_shaped);
        sign.innerHTML = dictOfSigns[pred_result.argMax(1).dataSync()[0]];
        prob.innerHTML = Math.max.apply(Math, pred_result.dataSync()).toFixed(2).toString();
    }
});

async function main(){
    // Start WebRTC Client Connection
    start();
    try {
        // Load TF Model from Server
        model = await tf.loadLayersModel(window.location.href + "model.json");
    } catch (e) {
        throw e;
    }
    // Initialise Rolling Window
    rw = new RollingWindow(windowWidth, numbJoints)
    // Initialize sign and prob
    sign.innerHTML = "No Video Yet";
    const initProb = 0;
    prob.innerHTML = initProb.toString();
    /*
    async function doPredictions() {
        setTimeout(() => {
            window.requestAnimationFrame(doPredictions);
        }, 1000 / freq); 
    }
    doPredictions();
    */
};

window.onunload = () => {
    stop();
}

// Start running here
main();
