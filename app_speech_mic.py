from flask import Flask
from flask import request
from flask import render_template
import os

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def speech_mic():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('assets/audio/audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render_template('speech_mic.html', request="POST")
    else:
        return render_template("speech_mic.html")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002, debug=True)
