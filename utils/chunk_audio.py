import os

from pydub import AudioSegment

import pydub.utils

FFMPEG_PATH = r"C:\ffmpeg-8.1.2-essentials_build\ffmpeg-8.1.2-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")

pydub.utils.get_prober_name = lambda: os.path.join(FFMPEG_PATH, "ffprobe.exe")

def split_audio(filepath) :

    chunk_folder = "chunks"
    
    files = os.listdir(chunk_folder)

    # chunks/chunk_0001.wav/uplaods

    # 1. Delete old chunks

    for i in range(len(files)):
     file_path = os.path.join(chunk_folder, files[i])
     os.remove(file_path)

    # 2. Load uploaded audio
    print("Uploaded File:", filepath)
    print("File Exists:", os.path.exists(filepath))
    
    audio = AudioSegment.from_file(filepath, format="wav")



    # 3. Split into chunks
    chunk_length = 3000
    chunk_count = 0
    for i in range(0, len(audio), chunk_length):
       
       chunk = audio[i:i + chunk_length]
       chunk_number = i // chunk_length
       chunk_count += 1
       
       filename = f"chunk_{chunk_number}.wav"
       
       file_path = os.path.join(chunk_folder, filename)
       
       chunk.export(file_path, format="wav")
    return chunk_count

   

    

    
    os.makedirs("chunks", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    