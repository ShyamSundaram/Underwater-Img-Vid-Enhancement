### Requirements
    pip install -r requirements.txt

### Train the model
    $ python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
### Test the model
    $ python test.py --checkpoint CHECKPOINTS_PATH
For convenience, you can run the following command to quickly see the results using the trained model reported in our paper.

    $ python test.py --checkpoint ./checkpoints/model_best_2842.pth.tar
