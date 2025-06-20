from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io, base64
from io import StringIO
from PIL import Image
import cv2
import numpy as np
import imutils

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/', methods=['POST', 'GET'])
def test_webcam_online():
    return render_template('face_webcam.html')

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', debug=True)
