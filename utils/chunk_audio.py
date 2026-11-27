import os
import shutil

from pydub import AudioSegment

CHUNK_FOLDER = "chunks"
CHUNK_LENGTH_MS = 3000   # 3-second chunks, about the length of the training clips
MIN_LAST_CHUNK_MS = 1000  # a tiny leftover piece at the end is too short to judge

# ---------------------------------------------------------------
# ffmpeg setup (no hardcoded C:\ path any more)
# ---------------------------------------------------------------
# Standard 16-bit .wav files are read by pydub without ffmpeg at all.
# ffmpeg is only a fallback for unusual files. To use it:
#   1. install ffmpeg and add its "bin" folder to your system PATH, OR
#   2. set FFMPEG_DIR to that folder before running the app, e.g. in
#      PowerShell:   $env:FFMPEG_DIR = "C:\path\to\ffmpeg\bin"
ffmpeg_dir = os.environ.get("FFMPEG_DIR")
if ffmpeg_dir:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

_ffmpeg = shutil.which("ffmpeg")
if _ffmpeg:
    AudioSegment.converter = _ffmpeg


def split_audio(filepath):
    """Split an uploaded .wav into 3-second chunks in the chunks/ folder.

    Returns the number of chunks saved.
    """
    # Create the folder FIRST. Before, os.listdir() ran on a folder that
    # might not exist yet, which crashed the app on a fresh clone.
    os.makedirs(CHUNK_FOLDER, exist_ok=True)

    # 1. Delete old chunks from any previous upload
    for name in os.listdir(CHUNK_FOLDER):
        os.remove(os.path.join(CHUNK_FOLDER, name))

    # 2. Load the uploaded audio
    print("Uploaded File:", filepath)
    print("File Exists:", os.path.exists(filepath))
    audio = AudioSegment.from_file(filepath, format="wav")

    if len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    # 3. Split into chunks
    chunk_count = 0
    for start in range(0, len(audio), CHUNK_LENGTH_MS):
        chunk = audio[start:start + CHUNK_LENGTH_MS]

        # Skip a very short leftover at the end (but keep it if it is the
        # only chunk, so short files still get a result)
        if len(chunk) < MIN_LAST_CHUNK_MS and chunk_count > 0:
            continue

        # Zero-padded names (chunk_0001) so sorting keeps the right order.
        # Before, "chunk_10" sorted before "chunk_2".
        filename = f"chunk_{chunk_count + 1:04d}.wav"
        chunk.export(os.path.join(CHUNK_FOLDER, filename), format="wav")
        chunk_count += 1

    return chunk_count
