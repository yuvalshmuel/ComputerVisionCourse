
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