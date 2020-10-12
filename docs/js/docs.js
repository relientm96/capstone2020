function fileToMarked(file, element){
    $.ajax({
        url: file,
        crossDomain: true,
        beforeSend: function(request) {
            request.setRequestHeader("Access-Control-Allow-Origin", "*");
            request.setRequestHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
            request.setRequestHeader("Access-Control-Allow-Headers", "X-PINGOTHER, Content-Type");
        },
        success: function(response){
            element.innerHTML = marked(response);
            $(`#${element.id} ul`).addClass("browser-default");
        }
    })
}

mdFilePaths = {
    "overview"              : "markdown/overview.md",
    "features"              : "markdown/features.md",
    "datasetprocessing"     : "markdown/datasetprocessing.md",
    "modeldevelopment"      : "markdown/modeldevelopment.md",
    "applicationdeployment" : "markdown/applicationdeployment.md",
    "acknowledgements"      : "markdown/acknowledgements.md",
    "introduction"          : "markdown/introduction.md",
    "humanpose"             : "markdown/humanpose.md"
}

$(document).ready(function(){
    $('.sidenav').sidenav();
    $('.scrollspy').scrollSpy();
      
    marked.setOptions({
        highlight: function(code) {
          return hljs.highlightAuto(code).value;
        },
    })
    
    for (var key in mdFilePaths){
        // check if the property/key is defined in the object itself, not in parent
        if (mdFilePaths.hasOwnProperty(key)) {           
            elem = document.getElementById(key)
            fileToMarked(mdFilePaths[key], elem);
        }
    }  
    console.log("Rendered finish, now load materialbox");
    $('.materialboxed').materialbox();
    console.log("Loaded Material Box");
});