# import os
# import speech_recognition as sr
# import ffmpeg
#
# command2mp3 = "ffmpeg -i uploads/uploadedvideo.mp4  uploads/uploadedvideo.mp3"
# command2wav = "ffmpeg -i uploads/uploadedvideo.mp3  uploads/uploadedvideo.wav"
#
# os.system(command2mp3)
# os.system(command2wav)
#
# r = sr.Recognizer()
#
# with sr.AudioFile('uploads/uploadedvideo.wav') as source:
#     audio = r.record(source, duration=300)
#
# f= open("uploads/uploadedtext.txt","w+")
# f.write(r.recognize_google(audio))
# f.close()


import speech_recognition as sr
import moviepy.editor as mp
clip = mp.VideoFileClip("uploads/uploadedvideo.mp4")
clip.audio.write_audiofile("uploads/converted.wav")
r = sr.Recognizer()
audio = sr.AudioFile("uploads/converted.wav")
with audio as source:
    audio_file = r.record(source)
result = r.recognize_google(audio_file)
with open('uploads/uploadedtext.txt',mode ='w') as file:
#     file.write("Recognized Speech:")
#     file.write("\n")
    file.write(result)
