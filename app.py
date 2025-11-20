# app.py
import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from utils import process_wav_bytes
import traceback

# load .env if present
load_dotenv()

app = Flask(__name__)





# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_and_transcribe():
    try:
        print("---- /upload called ----")

        if not request.files:
            print("No files in request.files")
            return jsonify({"transcription": None, "error": "no files"}), 400

        # pick the first incoming file (front-end uses field name 'file')
        field = list(request.files.keys())[0]
        f = request.files[field]
        raw = f.read()

        print(f"Received file field='{field}' filename='{f.filename}' bytes={len(raw)}")

        # Call your processing/transcription function you defined earlier
        try:
            transcription = process_wav_bytes(raw)   # <--- your function
            print("Transcription (first 200 chars):", (transcription or "")[:200])
        except Exception as e:
            # If the processing step throws, print full traceback to console for debugging
            tb = traceback.format_exc()
            print("Error during process_wav_bytes():\n", tb)
            return jsonify({"transcription": None, "error": "processing_failed", "traceback": tb}), 500

        # Return the transcription under the expected key
        return jsonify({"transcription": transcription}), 200

    except Exception as e:
        tb = traceback.format_exc()
        current_app.logger.error("Upload endpoint error:\n%s", tb)
        return jsonify({"transcription": None, "error": str(e), "traceback": tb}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
