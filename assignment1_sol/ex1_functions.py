import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import cv2
import math


def find_image_corners(img):
    width = img.shape[1]
    height = img.shape[0]
    corners = [[0, 0, height, height], [0, width, width, 0]]
    return corners


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
    num_points = len(mp_src[0])
    # print("num_points [{}]".format(num_points))
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
    homography_matrix = u[:, -1].reshape(3, 3)
    homography_matrix = np.divide(homography_matrix, homography_matrix[2, 2])
    return homography_matrix


# TODO: delete this verification method
def compute_homography_naive_verify(mp_src, mp_dst):
    """
    :param mp_src: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the src image.

    :param mp_dst: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the dst image.

    :return: H - Projective transformation matrix from src to dst
    """
    # https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    # import the images
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')
    # use cv2
    h_real , status_real = cv2.findHomography(mp_src.T, mp_dst.T)
    checkPoint = np.dot(h_real, np.append(mp_src.T[0], 1))
    checkPoint /= checkPoint[2] # to affine

    # plot and calculate the src image to the dst coordinats
    result = cv2.warpPerspective(img_src, h_real, (img_dst.shape[0],img_dst.shape[1]))
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow( img_src)
    axarr[1].imshow( img_dst)
    axarr[2].imshow(result) # plot src image
    plt.show()
    return h_real


####################################
# Part B
####################################

def test_homography(H, mp_src, mp_dst, max_err):
    """
    :param H: homography to test
    :param mp_src: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the src image
    :param mp_dst: A variable containing 2 rows and N columns, where the i column
    represents coordinates of match point i in the dst image
    :param max_err: A scalar that represents the maximum distance (in pixels) between the
    mapped src point to its corresponding dst point, in order to be
    considered as valid inlier
    :return:
    fit_percent – The probability (between 0 and 1) validly mapped src points (inliers).
    dist_mse - Mean square error of the distances between validly mapped src points,
    to their corresponding dst points (only for inliers).
    """
    src_points = convert_to_numpy(mp_src)
    src_points_mapped = np.matmul(H, src_points)  # 3xN points after being mapped with the homography
    src_points_mapped = np.divide(src_points_mapped, src_points_mapped[2, :])  # divide by the last row

    # assuming H is from the src to the dst (forward mapping) then we do not map the dst points
    dst_points = convert_to_numpy(mp_dst)

    dist = np.sqrt(np.sum((src_points_mapped - dst_points) ** 2, axis=0))
    num_inliers = np.count_nonzero(dist < max_err)
    num_points = len(mp_src[0])
    fit_percent = num_inliers / num_points
    # dist_mse = (1 / num_inliers) * np.sum((np.array((dist < max_err), dtype=int) * dist) ** 2, axis=0)
    dist_mse = (1 / num_inliers) * np.sum((np.array([1] * num_points, dtype=int) * dist) ** 2, axis=0)
    print("inliers [{}] total [{}] mse [{}]".format(num_inliers, num_points, dist_mse))
    return fit_percent, dist_mse


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
    p = 0.99  # we require 95% success probability from RANSAC algorithm
    n = 4  # we need 4 matching pairs of points to solve an homography
    prob_any_outlier = 1 - math.pow(inliers_percent, n)
    num_iterations = math.ceil(math.log(1 - p) / math.log(prob_any_outlier))
    print("num_iterations [{}]".format(num_iterations))

    num_points = len(mp_src[0])

    mse_best = None
    H_best = None
    for i in range(num_iterations):
        # randomly pick 4 matching points
        picked_indices = np.random.choice(num_points, n, replace=False)
        mp_src_pick = mp_src[:, picked_indices]
        mp_dst_pick = mp_dst[:, picked_indices]
        H = compute_homography_naive(mp_src_pick, mp_dst_pick)
        fit_percent, mse = test_homography(H, mp_src, mp_dst, max_err)
        if H_best is None or mse < mse_best:
            mse_best = mse
            H_best = H

    print("best mse [{}]".format(mse_best))
    return H_best


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