import enhance
import getFrames
import combineFrames
from tqdm import tqdm
import datetime

if __name__=="__main__":
    vid=input('Video name: ')
    starttime = datetime.datetime.now()
    print('[1/3]Extracting frames...')
    getFrames.getFps(vid)
    for i in tqdm(getFrames.getFrames(vid)):
        pass
    print('Frames extracted...\n')

    print('[2/3]Enhancing frames...')
    enhance.EnahanceInParallel("frames",5)
    print('Frames enhanced.\n')

    print('[3/3]Combining frames...')
    combineFrames.combineFrames("results")
    # for i in tqdm(combineFrames.combineFrames("results")):
    #     pass
    endtime = datetime.datetime.now()
    print('Done. Video saved as video.avi. Total time taken: '+str(endtime-starttime))