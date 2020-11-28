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

def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent = 100, max_err = 1):
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
    # https://github.com/tsherlock/panorama/blob/master/pano_stitcher.py
    # https://github.com/karanvivekbhargava/PanoramaStiching/blob/master/panorama.py
    # https: // github.com / tsherlock / panorama / blob / master / pano_stitcher.py
    ## todo: Change H Calculation here
    H_inden , junck=  cv2.findHomography( mp_src.T , mp_src.T)
    H, status_real = cv2.findHomography( mp_dst.T , mp_src.T)
    img_dst
    img_src # Left and atay the same image

    warped_src, A_src = warp_image(img_src,H_inden)
    warped_dst, A_dst = warp_image(img_dst  , H)
    result = create_mosaic([warped_src, warped_dst], [A_src, A_dst])
    plt.imshow(result)

    # warped_image = cv2.warpPerspective(imageA, H,
    #                                    (2* imageB.shape[1], imageB.shape[0]))
    # warped_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # find_image_size(mapped_points[0:1,:])
    # image_corners = convert_to_numpy(find_image_corners(img_src))
    # image_corners_mapped = np.matmul(H, image_corners)  # 3x4 image corners after being mapped with the homography
    # image_corners_mapped = np.divide(image_corners_mapped, image_corners_mapped[2, :])  # divide by the last row
    # min_x = np.min(image_corners_mapped[0, :])
    # min_y = np.min(image_corners_mapped[1, :])
    # min_x = int(min_x)
    # min_y = int(min_y)
    # img_src_padded = cv2.copyMakeBorder(img_src, -min_y if min_y < 0 else 0, 0, -min_x if min_x < 0 else 0, 0,
    #                                     cv2.BORDER_CONSTANT)
    # img_src_mapped = cv2.warpPerspective(img_src_padded, H, (img_src.shape[1], img_src.shape[0]))
    # #f, axarr = plt.subplots(1, 1)
    # #plt.axis('off')
    # #axarr.imshow(img_src_mapped)
    # #plt.show()
    # #####
    #
    # #result = cv2.warpPerspective(img_src, H,(img_src.shape[1] + img_dst.shape[1],max(img_src.shape[0],img_dst.shape[0])))
    # #result = cv2.warpPerspective(img_src, H,(2*img_dst.shape[1], max(img_src.shape[0], img_dst.shape[0])))
    # result = cv2.warpPerspective(img_src_mapped, H, (2 * img_dst.shape[1], max(img_src.shape[0], img_dst.shape[0])))
    # #result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #
    # # cheeat stiching
    # result[0:img_dst.shape[0],493: 493 + img_dst.shape[1]] = img_dst
    # #t_img_dst = cv2.copyMakeBorder(img_dst,abs(img_dst.shape[0] -  result.shape[0]), 0,0, 0,cv2.BORDER_CONSTANT)
    # #result[0:, img_dst.shape[1]:] = img_dst


def warp_image(image, homography):
    """Warps 'image' by 'homography'
    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.
    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, z = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    # Make new matrix that removes offset and multiply by homography
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    # height and width of new image frame
    height = int(round(ymax - ymin))
    width = int(round(xmax - xmin))
    size = (width, height)
    # Do the warp
    warped = cv2.warpPerspective(src=image, M=homography, dsize=size)

    return warped, (int(xmin), int(ymin))


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.
    Arguments:
    images: a list of 4-channel images to combine in the mosaic.
    origins: a list of the locations upper-left corner of each image in
    a common frame, e.g. the frame of a central image.
    Returns: a new 4-channel mosaic combining all of the input images. pixels
    in the mosaic not covered by any input image should have their
    alpha channel set to zero.
    """
    # find central image
    for i in range(0, len(origins)):
        if origins[i] == (0, 0):
            central_index = i
            break

    central_image = images[central_index]
    central_origin = origins[central_index]

    # zip origins and images together
    zipped = zip(origins, images)

    # sort by distance from origin (highest to lowest)
    func = lambda x: math.sqrt(x[0][0] ** 2 + x[0][1] ** 2)
    dist_sorted = sorted(zipped, key=func, reverse=True)
    # sort by x value
    x_sorted = sorted(zipped, key=lambda x: x[0][0])
    # sort by y value
    y_sorted = sorted(zipped, key=lambda x: x[0][1])

    # determine the coordinates in the new frame of the central image
    #if x_sorted[0][0][0] > 0:
    cent_x = 0  # leftmost image is central image
    #else:
      #  cent_x = abs(x_sorted[0][0][0])

    #if y_sorted[0][0][1] > 0:
    #    cent_y = 0  # topmost image is central image
    #else:
    cent_y = 0# abs(y_sorted[0][0][1])

    # make a new list of the starting points in new frame of each image
    spots = []
    for origin in origins:
        spots.append((origin[0]+cent_x, origin[1] + cent_y))

    zipped = zip(spots, images)

    # get height and width of new frame
    total_height = 0
    total_width = 0

    for spot, image in zipped:
        total_width = max(total_width, spot[0]+image.shape[1])
        total_height = max(total_height, spot[1]+image.shape[0])

    # print "height ", total_height
    # print "width ", total_width

    # new frame of panorama
    stitch = np.zeros((total_height, total_width, 4), np.uint8)

    # stitch images into frame by order of distance
    for image in dist_sorted:
        # offset_y = image[0][1] + cent_y
        # offset_x = image[0][0] + cent_x
        # for i in range(0, image[1].shape[0]):
        #     for j in range(0, image[1].shape[1]):
        #         # print i, j
        #         if image[1][i][j][3] != 0 :
        #             stitch[i+offset_y][j+offset_x][:4] = image[1][i][j]

        offset_y = image[0][1] + cent_y
        offset_x = image[0][0] + cent_x
        end_y = offset_y + image[1].shape[0]
        end_x = offset_x + image[1].shape[1]
        stitch[offset_y:end_y, offset_x:end_x, :4] = image[1]

    return stitch