const video = document.getElementById('webcam');
const predictionSpan = document.getElementById('prediction');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

const canvas = document.createElement('canvas');
canvas.width = 300;
canvas.height = 300;

function captureAndSend() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 300, 300);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        predictionSpan.textContent = `${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
      })
      .catch(err => {
        console.error('Error:', err);
      });
  }, 'image/jpeg');
}

// Call prediction every second
setInterval(captureAndSend, 1000);
