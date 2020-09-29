// Helper function to create array of zeros
function zeros(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; ++i) {
        array.push(dimensions.length == 1 ? 0 : zeros(dimensions.slice(1)));
    }
    return array;
}

// Define Rolling Window
class RollingWindow {
    constructor(windowWidth, numbJoints){
        this.windowWidth    = windowWidth;
        this.numbJoints     = numbJoints;
        this.points         = zeros([windowWidth, numbJoints]); 
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

export {RollingWindow}

