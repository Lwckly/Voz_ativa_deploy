import openai
from flask import Flask, render_template, request, jsonify
import tempfile, os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("api")
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

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

