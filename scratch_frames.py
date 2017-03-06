from detection import *
import numpy as np
import pickle
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
from scipy.ndimage.measurements import label
from math import ceil

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions - think (24, 24) may be good
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
heatmaps_threshold = 12

heatmaps = []
heatmap_sum = np.zeros((720,1280)).astype(np.float)

def process_image(image):
    global heatmaps, heatmap_sum
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    ystart = y_start_stop[0]
    ystop = y_start_stop[1]
    scale = 1.5

    out_img, bbox_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                   spatial_size,
                                   hist_bins, type='JPG')

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Add the heatmap to the heatmap list and the rolling sum
    heatmaps.append(heat)
    heatmap_sum = heatmap_sum + heat
    if len(heatmaps) > heatmaps_threshold:
        old_heatmap = heatmaps.pop(0)
        heatmap_sum -= old_heatmap

    # Visualize the heatmap when displaying
    heatmap = np.clip(heatmap_sum, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)
    return draw_img

if __name__ == '__main__':

    with open('svc_pickle_try.p', 'rb') as f:
        dist_pickle = pickle.load(f)
    #with open('svc_pickle.p', 'rb') as f:
    #    dist_pickle = pickle.load(f)
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    for index in range(670,1100):
        vidcap = cv2.VideoCapture('project_video.mp4')
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, image = vidcap.read()
        new_image = process_image(image)
        cv2.imshow('img', new_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
