from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg-8.1.2-essentials_build\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-8.1.2-essentials_build\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe"

print("Converter:", AudioSegment.converter)
print("FFprobe:", AudioSegment.ffprobe)

audio = AudioSegment.from_wav("uploads/03-01-01-01-01-02-02.wav")

print("Loaded successfully!")