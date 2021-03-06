import scipy.io
import time
import cv2

from ex1_functions import *
def tic():
    return time.time()
def toc(t):
    return float(tic()) - float(t)

##########################################################
# Don't forget to fill in your IDs!!!
ID1 = 1 # TODO: add ID
# students' IDs:
ID2 = 2 # TODO: add ID
##########################################################


# Parameters
max_err = 25
inliers_percent = 0.8

# Read the data:
img_src = mpimg.imread('src.jpg')
img_dst = mpimg.imread('dst.jpg')
matches = scipy.io.loadmat('matches') #matching points and some outliers
# matches = scipy.io.loadmat('matches_perfect') #loading perfect matches
match_p_dst = matches['match_p_dst'].astype(float)
match_p_src = matches['match_p_src'].astype(float)

ptont_images_with_points('src.jpg', 'dst.jpg', match_p_src, match_p_dst)

# Compute naive homography
tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
# H_naive = compute_homography_naive_verify(match_p_src, match_p_dst)  # TODO: remove this line
print('Naive Homography {:5.4f} sec'.format(toc(tt)))
print(H_naive)

image_corners = convert_to_numpy(find_image_corners(img_src))
image_corners_mapped = np.matmul(H_naive, image_corners)  # 3x4 image corners after being mapped with the homography
image_corners_mapped = np.divide(image_corners_mapped, image_corners_mapped[2,:])  # divide by the last row
min_x = np.min(image_corners_mapped[0,:])
min_y = np.min(image_corners_mapped[1,:])
max_x = np.min(image_corners_mapped[0,:])
max_y = np.min(image_corners_mapped[1,:])
min_x = int(min_x)
min_y = int(min_y)
max_x = int(max_x)
max_y = int(max_y)
border_top = -min_y if min_y < 0 else 0
border_bottom = max(0, max_x - img_dst.shape[1])
border_left = -min_x if min_x < 0 else 0
border_right = max(0, max_y - img_dst.shape[0])
img_src_padded = cv2.copyMakeBorder(img_src, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT)
img_src_mapped = cv2.warpPerspective(img_src_padded, H_naive, (img_src.shape[1], img_src.shape[0]))
f, axarr = plt.subplots(1, 1)
plt.axis('off')
axarr.imshow(img_src_mapped)
plt.show()

# Test naive homography
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive, match_p_src, match_p_dst, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# Compute RANSAC homography
tt = tic()
H_ransac = compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
print(H_ransac)

# Test RANSAC homography
tt = tic()
fit_percent, dist_mse = test_homography(H_ransac, match_p_src, match_p_dst, max_err)
print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

image_corners = convert_to_numpy(find_image_corners(img_src))
image_corners_mapped = np.matmul(H_ransac, image_corners)  # 3x4 image corners after being mapped with the homography
image_corners_mapped = np.divide(image_corners_mapped, image_corners_mapped[2,:])  # divide by the last row
# find_image_size(mapped_points[0:1,:])
min_x = np.min(image_corners_mapped[0,:])
min_y = np.min(image_corners_mapped[1,:])
min_x = int(min_x)
min_y = int(min_y)
img_src_padded = cv2.copyMakeBorder(img_src, -min_y if min_y < 0 else 0, 0, -min_x if min_x < 0 else 0, 0, cv2.BORDER_CONSTANT)
img_src_mapped = cv2.warpPerspective(img_src_padded, H_ransac, (img_src.shape[1], img_src.shape[0]))
f, axarr = plt.subplots(1, 1)
plt.axis('off')
axarr.imshow(img_src_mapped)
plt.show()


# Build panorama
tt = tic()
img_pan = panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err)
# img_pan = panorama(img_dst, img_src, match_p_dst, match_p_src, inliers_percent, max_err)  # TODO: remove this
print('Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Great Panorama')
plt.show()


## Student Files
#first run "create_matching_points.py" with your own images to create a mat file with the matching coordinates.
max_err = 25 # <<<<< YOU MAY CHANGE THIS
inliers_percent = 0.8 # <<<<< YOU MAY CHANGE THIS

img_src_test = mpimg.imread('src_test.jpg')
img_dst_test = mpimg.imread('dst_test.jpg')

matches_test = scipy.io.loadmat('matches_test')

match_p_dst = matches_test['match_p_dst']
match_p_src = matches_test['match_p_src']

ptont_images_with_points('src_test.jpg', 'dst_test.jpg', match_p_src, match_p_dst)

# Build student panorama

tt = tic()
img_pan = panorama(img_src_test, img_dst_test, match_p_src, match_p_dst, inliers_percent, max_err)
print('Student Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Awesome Panorama')
plt.show()


