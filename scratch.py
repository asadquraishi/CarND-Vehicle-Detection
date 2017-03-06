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
orient = 9  # HOG orientations - since literature suggests not much gain after 9
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions - think (24, 24) may be good
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
#y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
y_start_stop = [400, 500, 656]  # Min and max in y to search in slide_window()

def process_image(image):
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    ystart = y_start_stop[0]
    ystop1 = y_start_stop[1]
    ystop2 = y_start_stop[2]
    scale1 = 0.8
    scale2 = 1.5

    out_img1, bbox_list1 = find_cars(image, ystart, ystop1, scale1, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                   spatial_size,
                                   hist_bins, type='JPG')
    out_img2, bbox_list2 = find_cars(image, ystop1, ystop2, scale2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                   spatial_size,
                                   hist_bins, type='JPG')

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list1 + bbox_list2)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)
    return draw_img

if __name__ == '__main__':

    # Uncomment the following to retrain the classifier
    '''svc, X_scaler = train_classifier(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    data = {
        'svc': svc,
        'scaler': X_scaler,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'spatial_size': spatial_size,
        'hist_bins': hist_bins
    }
    with open('svc_pickle_try.p', 'wb') as f:
        pickle.dump(data, f)'''

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

    image_list = [
        ('2 cars close forward', 'test_images/test1.jpg'),
        ('1 car mid-far', 'test_images/test3.jpg'),
        ('no cars', 'test_images/test2.jpg'),
        ('1 car off screen, 2nd on', 'test_images/test5.jpg'),
        ('2 cars close more forward', 'test_images/test6.jpg'),
        ('2 cars very close', 'test_images/test7.jpg'),
        ('1 car close, other forward', 'test_images/test8.jpg')
    ]

    fig, axarray = plt.subplots(ceil(len(image_list) / 2), 2)

    index = 0
    for y in range(len(axarray)):
        for x in range(2):
            print(index)
            title = image_list[index][0]
            file = image_list[index][1]
            image = mpimg.imread(file)
            new_image = process_image(image)
            axarray[y][x].imshow(new_image)
            axarray[y][x].set_title(title, fontsize=20)
            index += 1
            if index == len(image_list):
                break

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()