var socket = io();

socket.on('connect', function(){
    console.log("Connected...!", socket.connected)
});

const video = document.querySelector("#videoElement");

video.width = 500;
video.height = 375; ;

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function (err0r) {
        console.log(err0r)
        console.log("Something went wrong!");
    });
}

function capture(video, scaleFactor) {
    if(scaleFactor == null){
        scaleFactor = 1;
    }
    var w = video.videoWidth * scaleFactor;
    var h = video.videoHeight * scaleFactor;
    var canvas = document.createElement('canvas');
        canvas.width  = w;
        canvas.height = h;
    var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
    return canvas;
}

const FPS = 30;

setInterval(() => {

    var type = "image/png"
    var video_element = document.getElementById("videoElement")
    var frame = capture(video_element, 1)
    var data = frame.toDataURL(type);
    data = data.replace('data:' + type + ';base64,', ''); //split off junk at the beginning

    socket.emit('image', data);
}, 10000/FPS);


socket.on('response_back', function(image){
    const image_id = document.getElementById('image');
    image_id.src = image;
});
