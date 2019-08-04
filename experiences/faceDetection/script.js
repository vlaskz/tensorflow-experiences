/*

USING FACE-API.JS, A TENSORFLOW FACE RECOGNITION ADD-ON OR API OR WRAPPER.
NOTHING MORE TONIGHT.
IF ANYONE SEE THIS, JUST GO REST A LITTLE... NOBODY IS MADE OF IRON, EVEN TONY STARK

*/


const video = document.getElementById('video')
const constraints = { audio: false, video: { width: 720, height: 560 } }

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo)

function startVideo() {
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (mediaStream) {
            video.srcObject = mediaStream
            video.onloadedmetadata = function (e) {
                video.play()
            }
        })
        .catch(function (err) { console.log(err.name + ": " + err.message); })
}

video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    document.body.append(canvas)
    const displaySize = {width: video.width, height:video.height}
    faceapi.matchDimensions(canvas, displaySize)
    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video,
            new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()

            const resizedDetections = faceapi.resizeResults(detections, displaySize)
            canvas.getContext('2d').clearRect(0,0,canvas.width, canvas.height)
            faceapi.draw.drawDetections(canvas, resizedDetections)
            faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
    }, 100)
})