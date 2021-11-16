import cv2
import os

# image_folder = 'frames'
# video_name = 'video.avi'
#get frame rate

def getFps():
    file1 = open("fps.txt","r")
    fps = file1.read()
    return fps
    
def combineFrames(image_folder,video_name='video.avi'):

    fps=getFps()

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for i in range(len(images)):
        images[i]=int(images[i][:-4])
    
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0]))+'.png')

    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, int(0), int(fps), (int(width),int(height)))
    c=0
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, str(image))+'.png'))
        yield c

    cv2.destroyAllWindows()
    video.release()