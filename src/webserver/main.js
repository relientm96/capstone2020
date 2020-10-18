/**
 * Main File 
 */

import {start, stop} from './client.js'
import {RollingWindow, zeros} from './rollingwindow.js'

// Rolling window variables
const numbJoints  = 98; 
const windowWidth = 35;
// Rolling Window
var rw;
// Model reference object
var model;
// Enumerated Signs that define output
const dictOfSigns = {
	0:"ambulance",
	1:"help", 
	2:"pain", 
	3:"hospital", 
	4:"thumbs"
}
// Socketio obj
const socket = io();
// Probability and sign variable
var sign = document.getElementById("currentsign");
var prob = document.getElementById("probability");

/**
 * Retrieve the array key corresponding to the largest element in the array.
 *
 * @param {Array.<number>} array Input array
 * @return {number} Index of array element with largest value
 */
function argMax(array) {
    // Referenced from https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
    return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
  }

socket.on("keypoints", function(keypoints){

    if (rw.isInit == false){
        document.getElementById("loadingWidget").style = "display:none";
    }

    // Receives openpose keypoints after rendering from backend
    var kp_arr = JSON.parse(keypoints)
    // Append point to rolling window
    if (rw.addPoint(kp_arr)){
        // Do prediction once we have new updated rolling window from this keypoint
        var keypoint_shaped = tf.tensor3d(rw.getPoints().flat(), [1, windowWidth, numbJoints])
        try {
            
            model.predict(keypoint_shaped).data().then(
                result => {
                    // Resulting prob:
                    var probVal = Math.max.apply(Math, result);
                    sign.innerHTML = dictOfSigns[argMax(result)];
                    prob.innerHTML = probVal.toFixed(2).toString();
                    var range   = probVal - Math.min.apply(Math, result);
                    // Check if we need to do a reset for the next round if prob difference too high
                    if (range < 0.9) {
                        // Initiate a reset to the window by copying over frames for whole window
                        rw.isInit = true;
                    }
                    else {
                        rw.isInit = false;
                    }
                }
            )
            
           /*
           // Synchronous prediction
           var predictions = model.predict(keypoint_shaped).dataSync();
           var probVal = Math.max.apply(Math, predictions);
           sign.innerHTML = dictOfSigns[argMax(predictions)];
           prob.innerHTML = probVal.toFixed(2).toString();
           // Check if we need to do a reset for the next round if prob difference too high
           var range   = probVal - Math.min.apply(Math, predictions);
           if (range < 0.8) {
               // Initiate a reset to the window by copying over frames for whole window
               rw.isInit = true;
            }
            else {
                rw.isInit = false;
            }
            */
        }
        catch (e){
            throw e;
        }
    }
});

/*
socket.on("alert_elbows", (value) => {
    console.log(value)
    sign.innerHTML = "Please Keep Elbows in frame!"
});
*/

async function main(){
    // Start WebRTC Client Connection
    start();
    try {
        // Load TF Model from Server
        model = await tf.loadLayersModel(window.location.href + "model.json");
        model.summary();
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
