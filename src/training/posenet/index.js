/***************** Only change things here!! *********************/
/*
Make sure that video labels are as such:

/videos -+
         |
         + -- ambulance/ --+ 
                           |
                           + - ambulance_1.mp4
                           + - ambulance_2.mp4
                           ...
         + -- help/ --+
                      |
                      + - help_1.mp4
                      + - help_2.mp4
                      ...
*/

// Path of your directory of all videos, structured as above
// Needs to be a relative path from this directory
const videoInputDir = '../videos/';

// Desired number of frames to stop
/*
  Due to labelling it if desiredFrames = 75, it will go from:
  help_1_1
  help_1_2
  help_1_3
  ...
  help_1_76 <----- Note it is +1 from desiredFrames
 */
const desiredFrames = 75;
// NOTE: My code does not handle < 75 frames so might need to handle that offline in python
// But my code will stop reading in frames after 75 frames

// Sign Strings to enumerated map
const dictOfSigns = {
    'ambulance':'0',
    'help'     :'1',
    'hospital' :'2',
    'pain'     :'3'
}
// Turn on if want logs
const LOG = true;

//============ Dont change variables from here ============//
/************ Implementation Starts Here ******************/

const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const fs = require('fs');
const path = require('path');
const ffmpeg = require('ffmpeg');
// Model reference
var net;
// 11 parts so 22 x,y's
const numbJoints = 22;

const poseNetPartsArray = [
    // In total we have 11 parts, so 22 keypoints for x,y's
    "nose_x",
    "nose_y",
    "leftEye_x",
    "leftEye_y",
    "rightEye_x",
    "rightEye_y",
    "leftEar_x",
    "leftEar_y",
    "rightEar_x",
    "rightEar_y",
    "leftShoulder_x",
    "leftShoulder_y",
    "rightShoulder_x",
    "rightShoulder_y",
    "leftElbow_x",
    "leftElbow_y",
    "rightElbow_x",
    "rightElbow_y",
    "leftWrist_x",
    "leftWrist_y",
    "rightWrist_x",
    "rightWrist_y"
]

function processResult(keypoints){
    var processedArray = [];
    // Want to include points only up to 10'th one (rightWrist)
    // 22 in total (11 x,y's)
    for (let i = 0; i <= 10; i++){
        processedArray.push(keypoints[i]['position']['x']);
        processedArray.push(keypoints[i]['position']['y']);
    }
    return processedArray;
}

function writeKeypointsToFile(array, sign, file){
    file = file.split('.')[0];
    fs.appendFileSync('x.txt', file + ',' + array.join(',') + '\n', function(err){
        if (err) {
            console.log("Error Appending to x.txt :", err);
        }
    });
    fs.appendFileSync('y.txt', file + ',' + dictOfSigns[sign] + '\n', function(err){
        if (err) {
            console.log("Error Appending to y.txt :", err);
        }
    });
}

async function processImageOutputKeypoints(image_path){
    try { 
        let data = fs.readFileSync(image_path);
        let imageTensor = tf.node.decodeImage(data);
        let pose = await net.estimateSinglePose(imageTensor, {
            flipHorizontal: true
        });
        return processResult(pose['keypoints']);
    } catch(err){
        console.log(err);
    }
}

async function processVideoFrames(vidDirectoryPath, sign){
    const outputfiles = fs.readdirSync(vidDirectoryPath);
    outputfiles.sort(
        function(a,b){
            // Sort by label number of frames
            label1 = path.basename(path.join(vidDirectoryPath, a)).split('.')[0].split('_')[2];
            label2 = path.basename(path.join(vidDirectoryPath, b)).split('.')[0].split('_')[2];
            if (label1 - label2 < 0){
                return -1;
            }
            else if (label1 - label2 == 0){
                return 0;
            }
            else {
                return 1;
            }
        }
    )
    var i;
    for (i = 0; i < outputfiles.length; i++){
        if ( i > desiredFrames ){
            // Exceeding frames
            break;
        } 
        else { 
            async function quickProcess(file){
                let imagepath = path.join(vidDirectoryPath, file);
                let result = await processImageOutputKeypoints(imagepath, sign, file);
                // Write result array to textfile
                writeKeypointsToFile(result, sign, file);
            }
            if(LOG){
                console.log("Processing", outputfiles[i]);
            }
            file = outputfiles[i];
            quickProcess(file);
        }
    }
    if(LOG){
        // Delete after processing all frames through posenet
        console.log("Finished this video directory", vidDirectoryPath);
        console.log("\n==========================\n")
    }
    fs.rmdirSync(vidDirectoryPath, { recursive: true });
}

async function extractFramesUsingFFmpeg(vidpath, fileSavePath, sign){
    try {
        new ffmpeg(vidpath, function (err, video) {
            if (!err) {
                label = fileSavePath;
                video.fnExtractFrameToJPG(fileSavePath, {
                    frame_rate: 30,
                    file_name : fileSavePath
                }, function(error, files){
                    // Now route each processed directory of images to process posenet on them
                    if(!error){
                        if(LOG){
                            console.log(`Extracted ${files.length} frames from ${fileSavePath}`);
                        }
                        // Process each frame from vid dir
                        processVideoFrames(fileSavePath, sign);
                    }
                    else {
                        console.log(error);
                    }
                
                });
            } else {
                console.log('Error: ' + err);
            }
        });
    } catch (e) {
        console.log(e.code);
        console.log(e.msg);
    }
}

async function main(){
    /**
     * Initialize x,y txt by removing and recreating them to recreate dataset
     */
    // Delete current txt files first to re-create dataset
    if ( (fs.existsSync('x.txt')) && (fs.existsSync('y.txt')) ){
        fs.unlink('x.txt', (err) => {
            if (err) throw err;
            console.log('Re-creating x.txt, old x.txt deleted');
        });
        // Delete current txt files first to re-create dataset
        fs.unlink('y.txt', (err) => {
            if (err) throw err;
            console.log('Re-creating y.txt, old y.txt deleted');
        });
    }
    // Append headers to csv files
    fs.appendFile('x.txt',  "filename," + poseNetPartsArray.join(',') + '\n', function (err) {
        if (err) {
            console.log("Error Appending to x.txt :", err);
        }   
    })
    fs.appendFile('y.txt', 'filename, signEnum' + '\n', function (err) {
        if (err) {
            console.log("Error Appending to y.txt :", err);
        }
    })

    // Load posenet model
    net = await posenet.load();

    const outputdir  = './output/';
    // Delete current output folder if it exists (recursively)
    if (fs.existsSync(outputdir)) {
        // Delete current directory to start over
        fs.rmdirSync(outputdir, { recursive: true });
    }
    // Re-create output directory of images folder
    fs.mkdirSync(outputdir);

    const videodatabase = fs.readdirSync(videoInputDir)
    for (const signDir of videodatabase){
        let input_dir = path.join(videoInputDir, signDir)
        // Sign assumed to be label of last element in input_dir path
        const sign = path.basename(input_dir);
        // Start reading input directory of videos and process them individually
        const dir = fs.readdirSync(input_dir)
        for (const file of dir) {
            let fileWithNoExtension = file.split('.')[0];
            let outputdirpath       = path.join(outputdir, fileWithNoExtension);
            let inputfilepath       = path.join(input_dir, file);
            // Make the subdirectories of each video as label
            fs.mkdirSync(outputdirpath);
            //console.log(fileWithNoExtension, outputdirpath)
            // Extract each frame as a jpg and process keypoints on it using posenet
            extractFramesUsingFFmpeg(inputfilepath, outputdirpath, sign)
        }
    }
}

// Run the program
main();

