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
            console.log(typeof(response));
            element.innerhtml = response;
        }
    })
}

$(document).ready(function(){
    $('.sidenav').sidenav();
    $('.scrollspy').scrollSpy();
    $('.materialboxed').materialbox();

    overview_elem = document.getElementById('overview')
    fileToMarked("markdown/overview.md", overview_elem);    
});
