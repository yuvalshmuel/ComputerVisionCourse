import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import cv2


def convert_to_numpy(scipy_points):
    """
    :param scipy_points: 2xN array, row 1 are x values of the points and
           row 2 are the corresponding y values of the points
    :return: 3xN vector of the points, row 1 are x coords and row 2 are y coords
             row 3 is all 1's
    """
    num_points = len(scipy_points[0])
    points = np.empty((0, num_points), np.double)
    points = np.append(points, np.array([scipy_points[0]]), axis=0)
    points = np.append(points, np.array([scipy_points[1]]), axis=0)
    points = np.append(points, np.array([[1]*num_points]), axis=0)
    return points


def find_image_size(points):
    """
    :param points: 2xn numpy matrix representing pixel coords
    :return: the size of the image required to contain all the given pixel coords
    """
    min_x = np.min(points[0,:])
    min_y = np.min(points[1,:])
    max_x = np.max(points[0,:])
    max_y = np.max(points[1,:])
    return min_x, min_y, max_x, max_y


def ptont_images_with_points(mp_src, mp_dst):
    """ """
    # import the images
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')
    # gather points
    src_points = convert_to_numpy(mp_src)
    dst_points = convert_to_numpy(mp_dst)
    # image plot
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_src)
    axarr[1].imshow(img_dst)
    for each in src_points.T:
        axarr[0].scatter([each[0]], [each[1]], s=15)
    axarr[1].imshow(img_dst)
    for each in dst_points.T:
        axarr[1].scatter([each[0]], [each[1]], s=15)
    plt.show()

####################################
# Part A
####################################

def compute_homography_naive(mp_src, mp_dst):
    """
    :param mp_src: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the src image.

    :param mp_dst: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the dst image.

    :return: H - Projective transformation matrix from src to dst
    """

    ########### test  with CV2 ############
    # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    # import the images
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')
    # use cv2
    h_real , status_real = cv2.findHomography(mp_src.T,mp_dst.T)
    checkPoint = np.dot(h_real, np.append(mp_src.T[0], 1))
    checkPoint /= checkPoint[2] # to affine

    # plot and calculate the src image to the dst coordinats
    result = cv2.warpPerspective(img_src, h_real,(img_dst.shape[0],img_dst.shape[1]))
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow( img_src)
    axarr[1].imshow( img_dst)
    axarr[2].imshow(result) # plot src image
    #######################

    num_points = len(mp_src[0])
    matches_matrix = np.empty((0, 9), np.double)

    for i in range(num_points):
        x_src = mp_src[0][i]
        y_src = mp_src[1][i]
        x_dst = mp_dst[0][i]
        y_dst = mp_dst[1][i]
        row1 = np.array([[x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst]], dtype=np.double)
        row2 = np.array([[0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src, -y_dst]], dtype=np.double)
        matches_matrix = np.append(matches_matrix, row1, axis=0)
        matches_matrix = np.append(matches_matrix, row2, axis=0)
    # compute A'A - 9x9 matrix
    matches_matrix_tilde = np.matmul(np.transpose(matches_matrix), matches_matrix)
    # compute SVD to get the first eigen vector - s is the eigenvalues
    u, s, vh = np.linalg.svd(matches_matrix_tilde, full_matrices=True)
    print(u)
    print(s)
    homography_matrix = u[:, -1].reshape(3, 3)
    return homography_matrix


####################################
# Part B
####################################

def test_homography(H, mp_src, mp_dst, max_err):
    """
    :param H: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the src image
    :param mp_src:A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the dst image
    :param mp_dst:
    :param max_err:A scalar that represents the maximum distance (in pixels) between the
    mapped src point to its corresponding dst point, in order to be
    considered as valid inlier
    :return:
    fit_percent – The probability (between 0 and 1) validly mapped src points (inliers).
    dist_mse - Mean square error of the distances between validly mapped src points,
    to their corresponding dst points (only for inliers).
    """
    num_points = len(mp_src[0])
    matches_matrix = np.empty((0, 9), np.double)

    for i in range(num_points):
        x_src = mp_src[0][i]
        y_src = mp_src[1][i]
        x_dst = mp_dst[0][i]
        y_dst = mp_dst[1][i]


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    """
    :param mp_src: A variable containing 2 rows and N columns, where the i column
represents coordinates of match point i in the src image.
    :param mp_dst: A variable containing 2 rows and N columns, where the i column
represents coordinates of match point i in the dst image.
    :param inliers_percent: The expected probability (between 0 and 1) of correct match points
from the entire list of match points.
    :param max_err: A scalar that represents the maximum distance (in pixels) between
the mapped src point to its corresponding dst point, in order to be
considered as valid inlier.
    :return:
    H – Projective transformation matrix from src to dst.
    """


####################################
# Part C
####################################

def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    """
    :param img_src: Source image expected to undergo projective transformation.
    :param img_dst: Destination image to which the source image is being mapped to.
    :param mp_src: A variable containing 2 rows and N columns, where the i column
represents coordinates of match point i in the src image.
    :param mp_dst: A variable containing 2 rows and N columns, where the i column
represents coordinates of match point i in the dst image.
    :param inliers_percent: The expected probability (between 0 and 1) of correct match points from
the entire list of match points.
    :param max_err:A scalar that represents the maximum distance (in pixels) between the
mapped src point to its corresponding dst point, in order to be
considered as valid inlier.
    :return:
    img_pan – Panorama image built from two input images.
    """