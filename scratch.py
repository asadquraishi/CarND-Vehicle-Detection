import glob
from detection import *
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label


train_test_splits = 0.8
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell - think
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions - think (24, 24) may be good
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [480, None]  # Min and max in y to search in slide_window()

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# deal with time series of data problem by manually splitting the train/test data
def train_test(train_test_split = 0.8, path=None):
    images = glob.glob(path)
    images_train = images[0:int(len(images)*train_test_split)]
    images_test = images[len(images_train):]
    return images_train, images_test

def train_classifier_old():
    car1 = glob.glob('vehicles/GTI*/*.png')
    car2 = glob.glob('vehicles/KITTI*/*.png')
    noncar1 = glob.glob('non-vehicles/GTI/*.png')
    noncar2 = glob.glob('non-vehicles/Extras/*.png')
    cars = car1 + car2
    notcars =  noncar1 + noncar2

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    # Perform feature engineering
    print("Performing feature engineering...")
    print("Extracting car features.")
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    print("Extracting non-car features.")
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    return svc, X_scaler

def train_classifier():
    # deal with time series of data problem by manually splitting the train/test data
    cars1_train, cars1_test = train_test(train_test_splits, path='vehicles/GTI*/*.png')
    noncars1_train, noncars1_test = train_test(train_test_splits, path='non-vehicles/GTI/*.png')
    cars2_train, cars2_test = train_test(train_test_splits, path='vehicles/KITTI*/*.png')
    noncars2_train, noncars2_test = train_test(train_test_splits, path='non-vehicles/Extras/*.png')

    cars_train = cars1_train + cars2_train
    cars_test = cars1_test + cars2_test
    noncars_train = noncars1_train + noncars2_train
    noncars_test = noncars1_test + noncars2_test

    #print("cars_train {}, cars_test {}, noncar_train {}, noncar_test {}".format(len(cars_train), len(cars_test), len(noncars_train), len(noncars_test)))

    # Perform feature engineering
    print("Performing feature engineering...")
    print("Extracting car feature train set.")
    car_train_features = extract_features(cars_train, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    print("Extracting non-car feature train set.")
    notcar_train_features = extract_features(noncars_train, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    print("Extracting car feature test set.")
    car_test_features = extract_features(cars_test, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    print("Extracting non-car feature test set.")
    notcar_test_features = extract_features(noncars_test, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X_train = np.vstack((car_train_features, notcar_train_features)).astype(np.float64)
    X_test = np.vstack((car_test_features, notcar_test_features)).astype(np.float64)

    # Fit a per-column scaler
    X_train_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    scaled_X_train = X_train_scaler.transform(X_train)
    # Fit a per-column scaler
    X_test_scaler = StandardScaler().fit(X_test)
    # Apply the scaler to X
    scaled_X_test = X_test_scaler.transform(X_test)
    # shuffle the data
    rand_state = np.random.randint(0, 100)
    X_train = shuffle(scaled_X_train, random_state = rand_state)
    X_test = shuffle(scaled_X_test, random_state = rand_state)

    # Define the labels vector
    y_train = np.hstack((np.ones(len(car_train_features)), np.zeros(len(notcar_train_features))))
    y_test = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    return svc,  # note that I would need to return the train scalar

if __name__ == '__main__':

    # Uncomment the following to retrain the classifier
    svc, X_scaler = train_classifier_old()
    data = {
        'svc': svc,
        'scaler': X_scaler,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'spatial_size': spatial_size,
        'hist_bins': hist_bins
    }
    with open('svc_pickle.p', 'wb') as f:
        pickle.dump(data, f)

    with open('svc_pickle.p', 'rb') as f:
        dist_pickle = pickle.load(f)
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    image = mpimg.imread('test_images/test1.jpg')
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    '''windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()'''

    ystart = 400
    ystop = 656
    scale =1.5

    out_img, bbox_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)

    #window_img = draw_boxes(draw_image, bbox_list, color=(0, 0, 255), thick=6)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    #plt.imshow(window_img)
    plt.show()