import openai
from flask import Flask, render_template, request, jsonify
import tempfile, os
from dotenv import load_dotenv
from utils import audio_record,audio_transcribe


app = Flask(__name__)

@app.route("/", methods=["POST","GET","HEAD"])
def index():
    if request.methods=="POST":
        raw_audio = audio_record()
        transcription= audio_transcribe(raw_audio)
        
    else:
        return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)




