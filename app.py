from flask import Flask, render_template, request, send_file
import os
import glob
import json

from collections import Counter
from utils.chunk_audio import split_audio
from utils.predict_emotion import predict_emotion


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CHUNK_FOLDER = "chunks"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {"wav"}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create every folder the app writes to. This is what was missing
# before — the app crashed on a fresh clone because "chunks" and
# "results" were never created, only "uploads" was.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHUNK_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "audio" not in request.files:
        return "No file selected."

    file = request.files["audio"]

    if file.filename == "" or not allowed_file(file.filename):
        return "No file selected."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Clear out chunks left over from a PREVIOUS upload before making
    # new ones. Before this fix, os.listdir(chunk_folder) below would
    # read every leftover chunk from every earlier upload too, so
    # results from different files could get mixed into one report.
    for old_chunk in glob.glob(os.path.join(CHUNK_FOLDER, "*")):
        os.remove(old_chunk)

    try:
        num_chunks = split_audio(filepath)
    except Exception as e:
        print(type(e), e)
        raise

    chunk_files = sorted(os.listdir(CHUNK_FOLDER))
    results = []
    emotion_list = []
    confidence_list = []

    for chunk in chunk_files:
        chunk_path = os.path.join(CHUNK_FOLDER, chunk)
        emotion, confidence = predict_emotion(chunk_path)

        emotion_list.append(emotion)
        confidence_list.append(confidence)

        print(f"Chunk: {chunk}")
        print(f"Emotion: {emotion}")
        print(f"Confidence: {confidence:.2f}")
        print("-----------------------")

        results.append({
            "chunk": chunk,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    # --- Single source of truth for "overall emotion" ---
    # Before: this was computed TWICE, once by majority vote (Counter)
    # for the JSON report, then overwritten by "highest single-chunk
    # confidence" right before rendering the page — so the downloaded
    # JSON and the page could disagree. We keep majority vote as the
    # one definition, used everywhere.
    emotion_counter = Counter(emotion_list)
    overall_emotion = emotion_counter.most_common(1)[0][0]

    average_confidence = round(
        sum(confidence_list) / len(confidence_list), 2
    ) if confidence_list else 0.0

    print(f"Average Confidence: {average_confidence}")
    print("\nEmotion Count:")
    print(emotion_counter)
    print(f"\nOverall Emotion: {overall_emotion}")

    analysis_report = {
        "filename": file.filename,
        "overall_emotion": overall_emotion,
        "total_chunks": num_chunks,
        "average_confidence": average_confidence,
        "emotion_distribution": dict(emotion_counter),
        "chunks": results
    }

    result_file = os.path.join(RESULTS_FOLDER, "emotion_analysis.json")
    with open(result_file, "w") as json_file:
        json.dump(analysis_report, json_file, indent=4)

    return render_template(
        "result.html",
        filename=file.filename,
        total_chunks=num_chunks,
        overall_emotion=overall_emotion,
        average_confidence=average_confidence,   # <-- was missing before
        results=results
    )


@app.route("/download")
def download():
    return send_file(
        os.path.join(RESULTS_FOLDER, "emotion_analysis.json"),
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)