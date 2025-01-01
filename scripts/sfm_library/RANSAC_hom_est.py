import numpy as np
import cv2
import random
import argparse
from matplotlib import pyplot as plt
import geometry_utils as gu
import plot_utils as pu

# Example usage:
if __name__ == "__main__":

    # Load images and feature matches (for illustration, using SIFT here)
    img1 = cv2.imread('./ext/image1.png')
    img2 = cv2.imread('./ext/image2.png')

    matches1 = np.array([...])  # Nx2 array of points in image 1
    matches2 = np.array([...])  # Nx2 array of points in image 2

    parser = argparse.ArgumentParser(
        description='Image matching using RANSAC and guided matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--useSuperGlue', action='store_true',
                        help='Use SuperGlue for feature matching')
    args = parser.parse_args()

    if args.useSuperGlue:
        path = './ext/image1_image2_matches.npz'
        npz = np.load(path) # Load dictionary with super point
        # Create a boolean mask with True for keypoints with a good match, and False for the rest
        mask = npz['matches'] > -1
        
        # Using the boolean mask, select the indexes of matched keypoints from image 2
        idxs = npz['matches'][mask]
        # Using the boolean mask, select the keypoints from image 1 with a good match
        matches1 = npz['keypoints0'][mask]
        matches2 = npz['keypoints1'][idxs]
        descriptors1 = npz['descriptors0']
        descriptors2 = npz['descriptors1']
    else:
        # Detect features using SIFT and match (for demo purposes)
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply NNDR (nearest-neighbor distance ratio) for filtering matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Get matched keypoints
        matches1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
        matches2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    # Estimate homography using RANSAC
    H, inliers = gu.ransac_homography(matches1, matches2)

    # Draw matches and inliers
    pu.draw_matches(img1, img2, matches1, matches2, inliers)
