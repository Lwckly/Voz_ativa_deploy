import openai
from flask import Flask, render_template, request, jsonify
import tempfile, os

app = Flask(__name__)
openai.api_key = os.getenv("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")
#have to create .env to ignore api
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio.save(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcription = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    os.remove(tmp_path)
    return jsonify({"text": transcription.text})
