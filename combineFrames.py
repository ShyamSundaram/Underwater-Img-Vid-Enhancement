import cv2
import os
import moviepy
import moviepy.editor
from moviepy.editor import *

image_folder = 'frames'
video_name = 'video.avi'

#get frame rate
def getFps():
    file1 = open("fps.txt","r")
    fps = int(file1.read())
    return fps

#attach audio file with the combined video into 
def attachAudio(image_folder,video_name='video.avi'): 
    videoclip = VideoFileClip(video_name)
    audioclip = AudioFileClip("audio.mp3")

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile("combined_video.mp4")

def combineFrames(image_folder,video_name='video.avi'):

    fps=getFps()

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for i in range(len(images)):
        images[i]=int(images[i][:-4])

    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0]))+'.png')

    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, str(image))+'.png'))

    cv2.destroyAllWindows()
    video.release()

# combineFrames("frames")
# attachAudio("frames")