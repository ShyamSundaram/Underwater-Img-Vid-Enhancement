import cv2
import os

image_folder = 'frames'
video_name = 'video.avi'

def combineFrames(image_folder,video_name='video.avi'):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for i in range(len(images)):
        images[i]=int(images[i][:-4])

    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0]))+'.png')

    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 29, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, str(image))+'.png'))

    cv2.destroyAllWindows()
    video.release()