from djitellopy import Tello
from AdaBins import models, model_io
from PIL import Image
import cv2
import argparse as ap
import os
# from models import UnetAdaptiveBins
# import model_io
# from PIL import Image

def get_adabins_model():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    # ap.add_argument("-m", "--model", type=str, default="mmod_human_face_detector.dat", help="path to dlib's CNN face detector model")
    # ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")
    # args = vars(ap.parse_args())
    
    os.chdir("C:\\Users\\Krubics JH\\DJI Drone Scan\\Positional Scan Images")
    rgb = Image.open("WIN_20211024_16_24_30_Pro.jpg")


    MIN_DEPTH = 1e-3
    MAX_DEPTH_NYU = 10
    MAX_DEPTH_KITTI = 80

    N_BINS = 256
    model = models.UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
    os.chdir("C:\\Users\\Krubics JH\\DJI Drone Scan\\AdaBins")
    pretrained_path = "./pretrained/AdaBins_kitti.pt"
    model, _, _ = model_io.load_checkpoint(pretrained_path, model)

    bin_edges, predicted_depth = model(rgb)


    # from infer import InferenceHelper

    # infer_helper = InferenceHelper(dataset='nyu')

    # # predict depth of a batched rgb tensor
    # example_rgb_batch = ...
    # bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

    # # predict depth of a single pillow image
    # img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
    # bin_centers, predicted_depth = infer_helper.predict_pil(img)

    # # predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
    # infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")

    # Variance Threshold of Image
    # import cv2
    #
    # def winVar(img, wlen):
    #     wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
    #     borderType=cv2.BORDER_REFLECT) for x in (img, img * img))
    #     return wsqrmean - wmean * wmean

    return predicted_depth


get_adabins_model()


