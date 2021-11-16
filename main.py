import enhance
import getFrames
import combineFrames

if __name__=="__main__":
    vid=input('Video name: ')

    print('[1/3]Extracting frames...')
    getFrames.getFps(vid)
    getFrames.getFrames(vid)
    print('Frames extracted...\n')
    print('[2/3]Enhancing frames...')
    enhance.EnahanceInParallel("frames")
    print('Frames enhanced.\n')
    print('[3/3]Combining frames...')
    combineFrames.combineFrames("results")
    print('Done. Video asaved as video.avi')