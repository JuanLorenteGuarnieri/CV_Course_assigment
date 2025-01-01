import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import scipy.linalg as scAlg
from scipy.linalg import expm, logm
from scipy.optimize import least_squares

# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def inhom2hom (x, multiplier=1):
    """
    Convert inhomogeneous coordinates to homogeneous coordinates, using a
    multiplier for the homogeneous coordinate.
    Parameters:
        x: inhomogeneous coordinates.
        multiplier: multiplier for the homogeneous coordinate.
    Returns:
        Homogeneous coordinates.
    """
    return np.vstack((x, np.ones((1, x.shape[1]))*multiplier))

def hom2inhom (x):
    """
    Convert homogeneous coordinates to inhomogeneous coordinates.
    Parameters:
        x: homogeneous coordinates.
    Returns:
        Inhomogeneous coordinates.
    """
    sz = x.shape[0]
    return x[:sz-1] / x[sz-1]

def compute_projection_matrix(K, Rt):
    """
    Compute the projection matrix P = K [R|t].
    Parameters:
        K: intrinsic camera matrix.
        Rt: extrinsic camera matrix.
    Returns:
        Projection matrix P.
    """
    return np.dot(K, Rt)

def project_point(K, R, t, X):
    """Projects a 3D point X in reference frame onto 2D image using intrinsic matrix K, rotation R, and translation t."""
    X_cam = R @ X + t
    x_proj = K @ X_cam
    x_proj /= x_proj[2]  # Convert to homogeneous coordinates
    return x_proj[:2]  # Return only the 2D coordinates

def project_points_hom(P, X, lbda = 1):
    """
    Project points in homogeneous coordinates.
    Parameters:
        P: projection matrix.
        X: 3D points in homogeneous coordinates.
        lbda: scale factor.
    Returns:
        Projected points in homogeneous coordinates.
    """
    return np.dot(P, X) * lbda

def project_points_inhom(P, X, lbda = 1, multiplier=1):
    """
    Project points in inhomogeneous coordinates.
    Parameters:
        P: projection matrix.
        X: 3D points in inhomogeneous coordinates.
        lbda: scale factor.
    Returns:
        Projected points in inhomogeneous coordinates.
    """
    sol = project_points_hom(P, inhom2hom(X, multiplier = 1), lbda)
    return hom2inhom(sol)

def project_points_v0(P, X, lbda = 1):
    """
    Projects 3D points onto a 2D plane using a given projection matrix.

    Args:
        P (numpy.ndarray): A 3x4 projection matrix.
        X (numpy.ndarray): A 3xN array of 3D points.
        lbda (float, optional): A scaling factor for the projection. Default is 1.

    Returns:
        numpy.ndarray: A 2xN array of projected 2D points.
    """
    X_h = np.vstack((X, np.ones((1, X.shape[1]))))  # Convert to homogeneous coordinates
    x_h = (P @ X_h) * lbda  # Apply projection matrix
    x = x_h[:2] / x_h[2]  # Convert to inhomogeneous (x, y) by dividing by the third coordinate
    return x

def project_points_SVD(X):
    """
    Projects points onto a line using Singular Value Decomposition (SVD).

    This function takes a set of points and projects them onto a line by performing
    Singular Value Decomposition (SVD) on the transpose of the input matrix. It then
    sets the smallest singular value to zero to enforce the points lying on a line.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (n_points, n_dimensions) representing the 
                       coordinates of the points to be projected.

    Returns:
    numpy.ndarray: A 2D array of the same shape as the input, representing the 
                   coordinates of the points projected onto the line.
    """
    u, s, vh = np.linalg.svd(X.T)
    s[2] = 0  # If all the points are lying on the line s[2] = 0, therefore we impose it
    xProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    # xProjectedOnTheLine /= xProjectedOnTheLine[2, :]
    return xProjectedOnTheLine


# Function to compute the line equation from two points in homogeneous coordinates
def compute_line(point1, point2):
    """
    Compute the line equation from two points in homogeneous coordinates.
    Parameters:
        point1: first point in homogeneous coordinates.
        point2: second point in homogeneous coordinates.
    Returns:
        Line equation coefficients.
    """
    return np.cross(point1, point2)

# Function to compute intersection of two lines
def compute_intersection(line1, line2):
    """
    Compute the intersection of two lines.
    Parameters:
        line1: first line equation coefficients.
        line2: second line equation coefficients.
    Returns:
        Intersection point.
    """
    p = np.cross(line1, line2)
    return p / p[2]  # Convert from homogeneous to inhomogeneous

def point_line_distance(F, p1, p2):
    """
    Compute the distance from point to epipolar line.
    Args:
        F: fundamental matrix.
        p1, p2: points from images (Nx2 array).
    Returns:
        dist: distance from points to epipolar lines.
    """
    p1_homogeneous = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2_homogeneous = np.hstack((p2, np.ones((p2.shape[0], 1))))

    # Epipolar lines in the second image for points in the first image
    lines2 = (F @ p1_homogeneous.T).T

    # Epipolar lines in the first image for points in the second image
    lines1 = (F.T @ p2_homogeneous.T).T

    # Compute distances from points to their corresponding epipolar lines
    dist1 = np.abs(np.sum(lines1 * p1_homogeneous, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    dist2 = np.abs(np.sum(lines2 * p2_homogeneous, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)

    return dist1 + dist2

def point_to_plane_distance(plane, point):
    """
    Compute the distance of a point to a plane.
    Parameters:
        plane: plane equation coefficients.
        point: point coordinates.
    Returns:
        Distance from the point to the plane.
    """
    a, b, c, d = plane
    x_p, y_p, z_p = point
    # Calculate the distance using the plane equation
    distance = abs(a * x_p + b * y_p + c * z_p + d) / np.sqrt(a**2 + b**2 + c**2)
    return distance

def obtain_proyection_matrices(K, R, t):
    """
    Compute the projection matrices P1 and P2 from the intrinsic matrix, rotation and translation.
    Params:
        K: Camera intrinsic matrix (3x3).
        R: Rotation matrix (3x3).
        t: Translation vector (3x1).
    """
    # First projection matrix P1 assumes camera 1 is at the origin.
    # Second projection matrix P2 uses the correct rotation and translation.
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    return P1, P2


def compute_homography(K1, K2, R, t, plane_normal, d):
    """
    Compute the homography from the floor plane.
    Params:
        K1, K2: Intrinsic camera matrices of camera 1 and 2.
        R: Rotation matrix from camera 1 to camera 2.
        t: Translation vector from camera 1 to camera 2.
        plane_normal: Normal vector of the plane in the first camera frame.
        d: Distance from the origin to the plane.
    """
    t_nT = np.outer(t, plane_normal)
    H = K2 @ (R + t_nT / d) @ np.linalg.inv(K1)
    return H


def visualize_point_transfer(H, img1, img2, pts1):
    """
    Visualize point transfer using the estimated homography.
    Params:
        H: Homography matrix.
        img1, img2: The two images.
        pts1: Points in the first image to transfer.
    Returns:
        pts2: Transferred points in the second image.
    """
    # Convert points to homogeneous coordinates
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)

    # Apply homography to transfer points
    pts2_h = (H @ pts1_h.T).T

    # Convert back to 2D
    pts2 = pts2_h[:, :2] / pts2_h[:, 2].reshape(-1, 1)

    # Plot the points on both images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(pts1[:, 0], pts1[:, 1], color='r')
    plt.title("Points in Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(pts2[:, 0], pts2[:, 1], color='b')
    plt.title("Transferred Points in Image 2")

    plt.show()

    return pts2


def compute_homography(pts_src, pts_dst):
    """
    findHomography implementation.
    """
    A = []

    # Build the matrix A as seen in the slides.
    for i in range(pts_src.shape[0]):
        x, y = pts_src[i, 0], pts_src[i, 1]
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    # Get the eigen values.
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    # Normalize the homography matrix.
    H = H / H[2, 2]

    return H


def normalize_points(pts):
    """
    Normalize points so that the centroid is at the origin and the average distance to the origin is sqrt(2).
    Args:
        pts: Nx2 array of points.
    Returns:
        pts_normalized: Nx2 array of normalized points.
        T: 3x3 transformation matrix that normalizes the points.
    """
    # Get the centroid of the points.
    centroid = np.mean(pts, axis=0)

    # Substract the centroid from the points so that the centroid is at the origin.
    pts_shifted = pts - centroid

    # Compute the average distance to the origin.
    avg_dist = np.mean(np.linalg.norm(pts_shifted, axis=1))

    # Scale the points so that the average distance is sqrt(2).
    scale = np.sqrt(2) / avg_dist
    pts_normalized = pts_shifted * scale

    # Build and return the transformation matrix.
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    return pts_normalized, T


def compute_fundamental_matrix(x1, x2):
    """
    Estimate the fundamental matrix using the 8-point algorithm.
    Params:
        x1, x2: Corresponding points in the two images.
    Returns:
        F: The estimated fundamental matrix.
    """
    # Normalie the points.
    x1_normalized, T1 = normalize_points(x1)
    x2_normalized, T2 = normalize_points(x2)

    # Build the matrix A.
    N = x1_normalized.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1x = x1_normalized[i, 0]
        y1x = x1_normalized[i, 1]
        x2x = x2_normalized[i, 0]
        y2x = x2_normalized[i, 1]
        A[i] = [x1x*x2x, x1x*y2x, x1x, y1x*x2x, y1x*y2x, y1x, x2x, y2x, 1]

    # Solve Af = 0 using SVD.
    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint.
    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0  # Third singular value is 0.
    F_normalized = U @ np.diag(S) @ Vt

    F = T2.T @ F_normalized @ T1

    # Normalize the fundamental matrix so that the last element is 1.
    return F / F[2, 2]


def decompose_essential_matrix(E):
    """
    Decompose the essential matrix E into two possible rotation matrices (R1, R2) and a translation vector t.
    Params:
        E: Essential matrix (3x3).
    Returns:
        R1, R2: Two possible rotation matrices (3x3).
        t: Translation vector (3x1).
    """
    # Essential matrix SVD.
    U, S, Vt = np.linalg.svd(E)

    # Check if E has two equal singular values and one of them close to zero.
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # Aux matrix W.
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Possible rotations.
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Translation is the third element of U.
    t = U[:, 2]

    # Assure that the rotation matrices are valid (determinant = 1).
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return R1, R2, t


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate points in 3D from two sets of corresponding points in two images.
    Params:
        P1 (np.ndarray): First camera projection matrix (3x4).
        P2 (np.ndarray): Second camera projection matrix (3x4).
        pts1 (np.ndarray): 2D points in the first camera.
        pts2 (np.ndarray): 2D points in the second camera.
    Returns:
        np.ndarray: Triangulated 3D points.
    """
    n_points = pts1.shape[0]
    pts_3d_hom = np.zeros((n_points, 4))

    for i in range(n_points):
        A = np.array([
            (pts1[i, 0] * P1[2, :] - P1[0, :]),
            (pts1[i, 1] * P1[2, :] - P1[1, :]),
            (pts2[i, 0] * P2[2, :] - P2[0, :]),
            (pts2[i, 1] * P2[2, :] - P2[1, :])
        ])

        _, _, Vt = np.linalg.svd(A)
        pts_3d_hom[i] = Vt[-1]  # Last singular vector is the solution.

    # Convert to non-homogeneous coordinates.
    pts_3d = pts_3d_hom[:, :3] / pts_3d_hom[:, 3][:, np.newaxis]

    return pts_3d


def triangulate_points2(P1, P2, pts1, pts2):
    """
    Triangulate points in 3D from two sets of corresponding points in two images.
    Params:
        P1 (np.ndarray): First camera projection matrix (3x4).
        P2 (np.ndarray): Second camera projection matrix (3x4).
        pts1 (np.ndarray): 2D points in the first camera.
        pts2 (np.ndarray): 2D points in the second camera.
    """
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)[:, :2]
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)[:, :2]

    pts_4d_h = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    pts_3d = pts_4d_h[:3] / pts_4d_h[3]  # Convert from homogeneous to 3D coordinates.
    return pts_3d.T


def is_valid_solution(R, t, K, pts1, pts2):
    """
    Check if a solution (R, t) generates valid 3D points (in front of both cameras).
    Params:
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        K (np.ndarray): Intrinsic camera matrix.
        pts1 (np.ndarray): Points in the first image.
        pts2 (np.ndarray): Points in the second image.
    """

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points.
    pts_3d = triangulate_points(P1, P2, pts1, pts2)

    # Check if the points are in front of both cameras (Z coordinate positive).
    pts_cam1 = pts_3d[:, 2]
    pts_cam2 = (R @ pts_3d.T + t.reshape(-1, 1))[2, :]

    # Return true if all points are in front of both cameras.
    return np.all(pts_cam1 > 0) and np.all(pts_cam2 > 0)


def triangulate_points_from_cameras(R, t, K, pts1, pts2):
    """
    Triangulate points 3D, given two sets of points projected in 2D in two cameras.
    Params:
        R (np.ndarray): Rotation matrix between the cameras.
        t (np.ndarray): Translation vector between the cameras.
        K (np.ndarray): Intrinsic camera matrix.
        pts1 (np.ndarray): Points in the first camera.
        pts2 (np.ndarray): Points in the second camera.
    Returns:
        np.ndarray: Triangulated 3D points.
    """

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    return triangulate_points(P1, P2, pts1, pts2)

##################### RAMSAC #####################

def compute_epipolar_lines(F, points):
    """
    Compute the epipolar lines corresponding to points in the other image.
    Args:
        F: Fundamental matrix.
        points: Nx2 array of points in one image.
    Returns:
        Epipolar lines corresponding to points in the other image.
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    lines = (F @ points_h.T).T  # Epipolar lines
    return lines

def guided_matching(matches1, matches2, F, threshold=3.0):
    """
    Perform guided matching using the epipolar geometry
    Args:
        matches1: Nx2 array of points from image 1.
        matches2: Nx2 array of points from image 2.
        F: Fundamental matrix.
        threshold: Distance threshold for epipolar constraint.
    Returns:
        Refined matches after guided matching.
    """
    lines1 = compute_epipolar_lines(F.T, matches2)  # Epipolar lines in image 1 for points in image 2
    lines2 = compute_epipolar_lines(F, matches1)    # Epipolar lines in image 2 for points in image 1

    refined_matches = []

    # For each point in image 1, find the closest point in image 2 along the epipolar line
    for i, (p1, p2) in enumerate(zip(matches1, matches2)):
        line1 = lines1[i]
        line2 = lines2[i]

        # Distance from p1 to epipolar line in image 1
        dist1 = np.abs(line1[0] * p1[0] + line1[1] * p1[1] + line1[2]) / np.sqrt(line1[0]**2 + line1[1]**2)

        # Distance from p2 to epipolar line in image 2
        dist2 = np.abs(line2[0] * p2[0] + line2[1] * p2[1] + line2[2]) / np.sqrt(line2[0]**2 + line2[1]**2)

        # If both distances are below the threshold, keep the match
        if dist1 < threshold and dist2 < threshold:
            refined_matches.append((p1, p2))

    return np.array(refined_matches)

def ransac_fundamental_matrix(matches1, matches2, num_iterations=10000, threshold=3):
    """
    Perform RANSAC to estimate the fundamental matrix.
    Args:
        matches1, matches2: Corresponding points in the two images.
        num_iterations: Number of RANSAC iterations.
        threshold: Distance threshold for inliers.
    Returns:
        F: Fundamental matrix.
        inliers: Inliers corresponding to the fundamental matrix.
    """
    num_points = matches1.shape[0]
    best_inliers = []
    best_F = None

    if num_points < 8:
        print("Not enough points to estimate the fundamental matrix (need at least 8).")
        return None, None

    for _ in range(num_iterations):
        # Randomly sample 8 points for the 8-point algorithm
        sample_indices = random.sample(range(num_points), 8)
        sample_p1 = matches1[sample_indices]
        sample_p2 = matches2[sample_indices]

        # Estimate fundamental matrix from the sample
        try:
            F = compute_fundamental_matrix(sample_p1, sample_p2)
        except np.linalg.LinAlgError as e:
            print(f"Fundamental matrix computation failed with error: {e}")
            continue

        # Compute transfer error for all points
        errors = point_line_distance(F, matches1, matches2)

        # Identify inliers
        inliers = np.where(errors < threshold)[0]

        # Keep the fundamental matrix with the most inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F

    # Check if we have enough inliers to refine the fundamental matrix
    if best_F is None or len(best_inliers) < 8:
        print("RANSAC failed to find a valid fundamental matrix.")
        print("best_F is ", best_F)
        return None, None

    # Refine fundamental matrix using all inliers
    inlier_p1 = matches1[best_inliers]
    inlier_p2 = matches2[best_inliers]
    best_F = compute_fundamental_matrix(inlier_p1, inlier_p2)

    return best_F, best_inliers

def transfer_error(H, p1, p2):
    """
    Compute the transfer error between points transformed by homography.
    Args:
        H: homography matrix
        p1, p2: points from images (Nx2 arrays).
    Returns:
        Transfer error for each point.
    """
    p1_homogeneous = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2_projected = (H @ p1_homogeneous.T).T
    p2_projected /= p2_projected[:, 2][:, np.newaxis]
    error = np.linalg.norm(p2_projected[:, :2] - p2, axis=1)
    
    return error

def ransac_homography(matches1, matches2, num_iterations=1000, threshold=5):
    """
    Perform RANSAC to estimate a homography matrix.
    Args:
        matches1, matches2: Matched points between two images (Nx2 arrays).
        num_iterations: Number of RANSAC iterations.
        threshold: Transfer error threshold to classify inliers.
    Returns:
        best_H: Homography matrix with the most inliers.
        best_inliers: List of indices of inliers.
    """
    num_points = matches1.shape[0]
    best_inliers = []
    best_H = None

    for _ in range(num_iterations):
        # Randomly sample 4 points
        sample_indices = random.sample(range(num_points), 4)
        sample_p1 = matches1[sample_indices]
        sample_p2 = matches2[sample_indices]

        # Estimate homography from the sample
        H = compute_homography(sample_p1, sample_p2)

        # Compute transfer error for all points
        errors = transfer_error(H, matches1, matches2)

        # Identify inliers
        inliers = np.where(errors < threshold)[0]

        # Keep the homography with the most inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    # Refine homography using all inliers
    if best_H is not None and len(best_inliers) > 4:
        inlier_p1 = matches1[best_inliers]
        inlier_p2 = matches2[best_inliers]
        best_H = compute_homography(inlier_p1, inlier_p2)

    return best_H, best_inliers

##################### SIFT MATHCING #####################
def indexMatrixToMatchesList(matchesList):
    """
    Converts a list of match indices and distances to a list of DMatch objects.

    This function takes a list of matches, where each match is represented by a list of three elements:
    the index of the descriptor in the first set, the index of the descriptor in the second set, and the distance between the descriptors.
    It converts this list into a list of OpenCV DMatch objects, which can be used for further processing or visualization.

        matchesList: nMatches x 3 --> [[indexDesc1, indexDesc2, descriptorDistance], ...]
            A list of matches, where each match is represented by a list containing the index of the descriptor in the first set,
            the index of the descriptor in the second set, and the distance between the descriptors.
        dMatchesList: list of n DMatch objects
            A list of OpenCV DMatch objects created from the input matches list.
    """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
    Converts a list of DMatch objects to an index matrix.

    Parameters:
    dMatchesList (list): A list of n DMatch objects, where each DMatch object contains the following attributes:
        - queryIdx (int): Index of the descriptor in the query set.
        - trainIdx (int): Index of the descriptor in the train set.
        - distance (float): Distance between the descriptors.

    Returns:
    list: A list of lists where each sublist contains three elements:
        - indexDesc1 (int): Index of the descriptor in the query set.
        - indexDesc2 (int): Index of the descriptor in the train set.
        - descriptorDistance (float): Distance between the descriptors.
    """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbors Matching algorithm checking the Distance Ratio.
    A match is accepted only if the distance to the nearest neighbor is less than
    distRatio times the distance to the second nearest neighbor.
    -input:
        desc1: descriptors from image 1 (nDesc1 x 128)
        desc2: descriptors from image 2 (nDesc2 x 128)
        distRatio: distance ratio threshold (0.0 < distRatio < 1.0)
        minDist: minimum distance threshold to accept a match
    -output:
        matches: list of accepted matches with [[indexDesc1, indexDesc2, distance], ...]
    """
    matches = []
    nDesc1 = desc1.shape[0]

    for kDesc1 in range(nDesc1):
        # Compute L2 distance (Euclidean distance) between desc1[kDesc1] and all descriptors in desc2
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))

        # Sort the distances and get the two nearest neighbors
        indexSort = np.argsort(dist)
        d1 = dist[indexSort[0]]  # Distance to nearest neighbor
        d2 = dist[indexSort[1]]  # Distance to second nearest neighbor

        # Apply NNDR: check if d1 is less than distRatio * d2
        if d1 < distRatio * d2 and d1 < minDist:
            # If the match passes the distance ratio test and is below the minimum distance threshold, accept it
            matches.append([kDesc1, indexSort[0], d1])

    return matches

##################### OTHERS #####################

def resLineFitting(Op,xData):
    """
    Residual function for least squares method
    -input:
      Op: vector containing the model parameters to optimize (in this case the line). This description should be minimal
      xData: the measurements of our model whose residual we want to calculate
    -output:
      res: vector of residuals to compute the loss
    """
    theta = Op[0]
    d = Op[1]
    l_model = np.vstack((np.cos(theta),np.sin(theta), -d))
    res = (l_model.T @ xData).squeeze() # Since the norm is unitary the distance is easier
    res = res.flatten()
    return res

##################### BUNDLE ADJUSTMENT (TO DO) #####################
# Cross-product matrix for rotation vector
def crossMatrix(x):
    """
    Generate a cross-product (skew-symmetric) matrix from a vector.
    Args:
        x: A 3-element vector.
    Returns:
        A 3x3 skew-symmetric matrix.
    """
    M = np.array([[0, -x[2], x[1]], 
                  [x[2], 0, -x[0]], 
                  [-x[1], x[0], 0]])
    return M

# Inverse cross-product for retrieving rotation vector from a skew-symmetric matrix
def crossMatrixInv(M):
    """
    Retrieve the rotation vector from a cross-product (skew-symmetric) matrix.
    Args:
        M: A 3x3 skew-symmetric matrix.
    Returns:
        A 3-element vector.
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Calculate the reprojection residuals for bundle adjustment using two views.
    
    Parameters:
        Op (array): Optimization parameters including T_21 (rotation and translation between views) 
                    and X1 (3D points in reference frame 1).
        x1Data (array): (3 x nPoints) 2D points in image 1 (homogeneous coordinates).
        x2Data (array): (3 x nPoints) 2D points in image 2 (homogeneous coordinates).
        K_c (array): (3 x 3) intrinsic calibration matrix.
        nPoints (int): Number of 3D points.
        
    Returns:
        res (array): Residuals, which are the errors between the observed 2D matched points 
                     and the projected 3D points.
    """
    # Extract rotation (theta) and translation (t) from optimization parameters
    theta = Op[:3]                # Rotation vector (3 parameters)
    t_21 = Op[3:6]                # Translation vector (3 parameters)
    X1 = Op[6:].reshape((3, nPoints))  # 3D points (each with 3 coordinates)
    
    # Compute rotation matrix from rotation vector theta using exponential map
    R_21 = expm(crossMatrix(theta))  # Compute R_21 from theta

    # Residuals array
    residuals = []

    # Compute residuals for each point
    for i in range(nPoints):
        # Project point in ref 1 to ref 2
        x1_proj = project_point(K_c, R_21, t_21, X1[:, i])
        
        # Calculate residuals for x and y coordinates
        residuals.extend((x1_proj - x2Data[:2, i]).tolist())
        # print("Residual nº ", i, " completed")
    return np.array(residuals)

def bundle_adjustment(x1Data, x2Data, K_c, T_init, X_in):
    """
    Perform bundle adjustment using least-squares optimization.
    x1Data: Observed 2D points in image 1
    x2Data: Observed 2D points in image 2
    K_c: Intrinsic calibration matrix
    T_init: Initial transformation parameters (theta, t)
    X_in: Initial 3D points
    """
    
    # Define the fraction of points to use in the sample
    sample_fraction = 0.3  # For example, 30% of the points
    nPoints_sample = int(sample_fraction * X_in.shape[1])

    # Select a random sample of point indices
    sample_indices = np.random.choice(X_in.shape[1], nPoints_sample, replace=False)

    # Extract the sampled points using the selected indices
    X_init_sample = X_in[:, sample_indices]
    x1Data_sample = x1Data[:, sample_indices]
    x2Data_sample = x2Data[:, sample_indices]
    
    # Initial 3D points (X_init_sample) already available
    X_init = X_init_sample[:3, :]

    initial_params = np.hstack([T_init[:3], T_init[3:], X_init.T.flatten()])
    # initial_params = np.hstack([[ 0.011, 2.6345, 1.4543], [-1.4445, -2.4526, 18.1895], X_init.T.flatten()])

    # Run bundle adjustment optimization
    result = least_squares(resBundleProjection, initial_params, args=(x1Data_sample, x2Data_sample, K_c, nPoints_sample), method='trf') #method='lm'

    # Retrieve optimized parameters
    Op_opt = result.x
    theta_opt = Op_opt[:3]
    t_opt = Op_opt[3:6]
    X_opt = Op_opt[6:].reshape((nPoints_sample, 3))

    # Return optimized rotation, translation, and 3D points
    return expm(crossMatrix(theta_opt)), t_opt, X_opt

##################### KANNALA #####################

# Function to unproject a 2D point (u) back into 3D space using the Kannala-Brandt model
def unproject_kannala_brandt(u, K, D):
    """
    Unprojection usando el modelo Kannala-Brandt.

    Parámetros:
    - u: Punto en la imagen (np.array([u_x, u_y])).
    - K: Matriz intrínseca de la cámara (3x3).
    - D: Coeficientes de distorsión [k1, k2, k3, k4].

    Retorna:
    - v: Vector en la esfera unitaria (np.array([x, y, z])).
    """
    # Step 1: Convert the image point to camera coordinates
    u_hom = np.array([u[0], u[1], 1.0])  # Homogeneous coordinates
    x_c = np.linalg.inv(K) @ u_hom       # Coordinates in the camera system

    # Step 2: Compute r and phi
    r = np.sqrt((x_c[0]**2 + x_c[1]**2) / x_c[2]**2)
    phi = np.arctan2(x_c[1], x_c[0])

    # Step 3: Solve the polynomial to find theta using the coefficients from D
    # 9th-degree polynomial: d(theta) = r
    # Expressed as: k4 * theta^9 + k3 * theta^7 + k2 * theta^5 + k1 * theta^3 + theta - r = 0
    coeffs = [D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -r]  # Polynomial coefficients
    roots = np.roots(coeffs)                             # Solve the polynomial

    # Filter real solutions
    theta_solutions = roots[np.isreal(roots)].real
    # Select the only positive solution (assuming it is unique and valid)
    theta = theta_solutions[theta_solutions >= 0][0]

    # Step 4: Compute the vector on the unit sphere using theta and phi
    v = np.array([
        np.sin(theta) * np.cos(phi),  # x
        np.sin(theta) * np.sin(phi),  # y
        np.cos(theta)                 # z
    ])

    return v


# Kannala-Brandt projection model (already defined in the previous code)
def project_kannala_brandt(X, K, D):
    # Extract 3D coordinates
    x, y, z = X

    # Normalize the 3D point by the z-coordinate
    r = np.sqrt(x**2 + y**2)

    # Apply the Kannala-Brandt model distortion
    theta = np.arctan(r / z)
    theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)

    # Project the point into the image plane
    x_proj = K[0, 0] * theta_d * (x / r)
    y_proj = K[1, 1] * theta_d * (y / r)
    u = np.array([x_proj + K[0, 2], y_proj + K[1, 2], 1])  # 2D projection with homogeneous coordinates
    return u


def triangulate(x1, x2, K1, K2, T_wc1, T_wc2):
    """
    Triangulate points using two sets of 2D points and camera matrices.

    Arguments:
    x1 -- Points in camera 1's image (shape 3xN, homogeneous coordinates)
    x2 -- Points in camera 2's image (shape 3xN, homogeneous coordinates)
    K1, K2 -- Intrinsic matrices of camera 1 and 2
    T_wc1, T_wc2 -- Extrinsic transformations (world to camera)

    Returns:
    X -- Triangulated 3D points (shape 4xN, homogeneous coordinates)
    """
    # Convert extrinsics to projection matrices
    P1 = K1 @ T_wc1[:3, :]  # Projection matrix for camera 1
    P2 = K2 @ T_wc2[:3, :]  # Projection matrix for camera 2

    # Number of points
    n_points = x1.shape[1]
    X = np.zeros((4, n_points))

    for i in range(n_points):
        # Formulate the linear system A * X = 0
        A = np.vstack([
            x1[0, i] * P1[2, :] - P1[0, :],
            x1[1, i] * P1[2, :] - P1[1, :],
            x2[0, i] * P2[2, :] - P2[0, :],
            x2[1, i] * P2[2, :] - P2[1, :]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1]  # Homogeneous 3D point (last row of V)

    # Normalize homogeneous coordinates
    X /= X[3, :]
    return X

def resBundleProjectionFishEye(Op, x1Data, x2Data, x3Data, x4Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, nPoints):
    """
    Calculate reprojection residuals for bundle adjustment with four fish-eye cameras.

    Parameters:
        Op (array): Optimization parameters including T_wAwB (6 params) and 3D points (X).
        x1Data (array): (3 x nPoints) Observed 2D points in image 1.
        x2Data (array): (3 x nPoints) Observed 2D points in image 2.
        K_1, K_2 (array): Intrinsic calibration matrices for the two cameras.
        D1_k, D2_k (array): Distortion coefficients for the two cameras.
        T_wc1, T_wc2 (array): Extrinsic transformations for the stereo cameras.
        nPoints (int): Number of 3D points.

    Returns:
        res (array): Residuals between observed and projected points.
    """

    # Extract rotation (theta) and translation (t) from optimization parameters
    theta = Op[:3]
    t_21 = Op[3:6]
    X1 = Op[6:].reshape((3, nPoints))  # 3D points

    # Compute rotation matrix from theta
    R_21 = expm(crossMatrix(theta))  # Rotation matrix

    residuals = []

    for i in range(nPoints):
        # Project 3D points in camera 1
        x1_proj = project_kannala_brandt(X1[:, i], K_1, D1_k)

        # Transform 3D points to camera 2 reference frame
        X2 = T_wc2[:,:3] @ X1[:, i] + T_wc2[:, 3]
        X3 = R_21 @ X1[:, i] + t_21
        X4 = T_wc2[:,:3] @ X3 + T_wc2[:, 3]

        # Project 3D points in camera 2
        x2_proj = project_kannala_brandt(X2[:, i], K_1, D1_k)
        # Project 3D points in camera 3
        x3_proj = project_kannala_brandt(X3[:, i], K_2, D2_k)


        # Project 3D points in camera 4
        x4_proj = project_kannala_brandt(X4[:, i], K_2, D2_k)

        # Compute residuals for all cameras
        residuals.extend((x1_proj[:2] - x1Data[i, :2]).tolist())
        residuals.extend((x2_proj[:2] - x2Data[i, :2]).tolist())
        residuals.extend((x3_proj[:2] - x3Data[i, :2]).tolist())
        residuals.extend((x4_proj[:2] - x4Data[i, :2]).tolist())

    return np.array(residuals)

    # Extract T_wAwB (rotation and translation) and 3D points (X)
    theta = Op[:3]  # Rotation vector
    t_wAwB = Op[3:6]  # Translation vector
    X = Op[6:].reshape((3, nPoints))  # 3D points

    # Compute T_wAwB from theta and translation
    R_wAwB = expm(crossMatrix(theta))
    T_wAwB = np.eye(4)
    T_wAwB[:3, :3] = R_wAwB
    T_wAwB[:3, 3] = t_wAwB

    # Compute residuals
    residuals = []
    for i in range(nPoints):
        # Transform point to camera 1
        print(X[:, i])
        # X_c1 = T_wc1 @ np.vstack((X[:, i], 1))
        # x1_proj = project_kannala_brandt(X_c1[:3], K_1, D1_k)
        x1_proj = project_kannala_brandt(X[:, i], K_1, D1_k)

        # Transform point to camera 2
        # X_c2 = T_wc2 @ np.vstack((T_wAwB @ np.hstack((X[:, i], 1))[:3], 1))
        # x2_proj = project_kannala_brandt(X_c2[:3], K_2, D2_k)
        x2_proj = project_kannala_brandt(X[:, i], K_2, D2_k)

        # Compute residuals
        res_x1 = x1_proj - x1Data[:2, i]
        res_x2 = x2_proj - x2Data[:2, i]

        residuals.extend(res_x1.tolist() + res_x2.tolist())

    return np.array(residuals)


def bundle_adjustment_fish_eye(x1Data, x2Data, x3Data, x4Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, T_init, X_in):
    """
    Perform bundle adjustment for fish-eye stereo setup.
    """
    # Flatten initial parameters (T_wAwB and 3D points)
    initial_params = np.hstack([T_init[:3], T_init[3:], X_in.flatten()])

    # Run least-squares optimization
    result = least_squares(
        resBundleProjectionFishEye, initial_params,
        args=(x1Data, x2Data, x3Data, x4Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, X_in.shape[0]),
        method='trf'
    )

    # Retrieve optimized parameters
    Op_opt = result.x
    theta_opt = Op_opt[:3]
    t_opt = Op_opt[3:6]
    X_opt = Op_opt[6:].reshape((-1, 3))

    # Compute optimized T_wAwB
    R_opt = expm(crossMatrix(theta_opt))
    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt

    return T_opt, X_opt

##################### INTERPOLATION FUNCTIONS #####################

def numerical_gradient(img_int: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    :param img:image to interpolate
    :param point: [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: Ix_y = [[Ix_0,Iy_0],[Ix_1,Iy_1], ... [Ix_n,Iy_n]]
    """

    a = np.zeros((point.shape[0], 2), dtype= float)
    filter = np.array([-1, 0, 1], dtype=float)
    point_int = point.astype(int)
    img = img_int.astype(float)

    for i in range(0,point.shape[0]):
        py = img[point_int[i,0]-1:point_int[i,0]+2,point_int[i,1]].astype(float)
        px = img[point_int[i,0],point_int[i,1]-1:point_int[i,1]+2].astype(float)
        a[i, 0] = 1/2*np.dot(filter,px)
        a[i, 1] = 1/2*np.dot(filter,py)

    return a

def int_bilineal(img: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    Vq = scipy.ndimage.map_coordinates(img.astype(np.float), [point[:, 0].ravel(), point[:, 1].ravel()], order=1, mode='nearest').reshape((point.shape[0],))

    :param img:image to interpolate
    :param point: point subpixel
    point = [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: [gray0,gray1, .... grayn]
    """
    A = np.zeros((point.shape[0], 2, 2), dtype= float)
    point_lu = point.astype(int)
    point_ru = np.copy(point_lu)
    point_ru[:,1] = point_ru[:,1] + 1
    point_ld = np.copy(point_lu)
    point_ld[:, 0] = point_ld[:, 0] + 1
    point_rd = np.copy(point_lu)
    point_rd[:, 0] = point_rd[:, 0] + 1
    point_rd[:, 1] = point_rd[:, 1] + 1

    A[:, 0, 0] = img[point_lu[:,0],point_lu[:,1]]
    A[:, 0, 1] = img[point_ru[:,0],point_ru[:,1]]
    A[:, 1, 0] = img[point_ld[:,0],point_ld[:,1]]
    A[:, 1, 1] = img[point_rd[:,0],point_rd[:,1]]
    l_u = np.zeros((point.shape[0],1,2),dtype= float)
    l_u[:, 0, 0] = -((point[:,0]-point_lu[:,0])-1)
    l_u[:, 0, 1] = point[:,0]-point_lu[:,0]

    r_u = np.zeros((point.shape[0],2,1),dtype= float)
    r_u[:, 0, 0] = -((point[:,1]-point_lu[:,1])-1)
    r_u[:, 1, 0] = point[:, 1]-point_lu[:,1]
    grays = l_u @ A @ r_u

    return grays.reshape((point.shape[0],))

##################### LUCAS KANADE #####################
def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch

    # Mean of the patch
    i0_mean = np.mean(i0)
    i0_std = np.std(i0)  # Standard deviation of the patch

    # Initialize the result array
    result = np.zeros(search_area.shape, dtype=np.float64)

    # Margins to avoid boundary issues
    margin_y = i0.shape[0] // 2
    margin_x = i0.shape[1] // 2

    # Iterate over the search area (excluding margins)
    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            # Extract the corresponding region in the search area
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]

            # Compute mean and std of the region
            i1_mean = np.mean(i1)
            i1_std = np.std(i1)

            # Avoid division by zero
            if i1_std == 0 or i0_std == 0:
                result[i, j] = 0  # Invalid or texture-less region
            else:
                # Compute the NCC value
                ncc = np.sum((i0 - i0_mean) * (i1 - i1_mean)) / (i0_std * i1_std * i0.size)
                result[i, j] = ncc
    return result

def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):
    """
    Estimate the optical flow using normalized cross-correlation (NCC) for a single point.
    Parameters:
        img1_gray: np.array - Grayscale image at time t.
        img2_gray: np.array - Grayscale image at time t+1.
        i_img: int - Row index of the point.
        j_img: int - Column index of the point.
        patch_half_size: int - Half size of the patch to extract around the point (default is 5).
        searching_area_size: int - Size of the searching area around the point (default is 100).

    Returns:
        i_flow: int - Estimated row displacement.
        j_flow: int - Estimated column displacement.
    """
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    result = normalized_cross_correlation(patch, search_area)

    iMax, jMax = np.where(result == np.amax(result))

    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow

def lucas_kanade_sparse_optical_flow(img1, img2, points, initial_u, window_size=11, epsilon=1e-6, max_iter=50):
    """
    Refines optical flow using Lucas-Kanade approach starting from the initial motion vectors (u, v),
    utilizing bilinear interpolation and numerical gradient functions.
    
    Parameters:
        img1: np.array - Grayscale image at time t.
        img2: np.array - Grayscale image at time t+1.
        points: np.array[:, 2] - List of sparse points to refine
        initial_u: np.array[2] - Initial motion vector.
        window_size: int - Horizontal and vertical size of the centered window for
                           refinement, which must be odd (default is 11).
        epsilon: float - Convergence threshold for motion updates (default is 1e-6).
        max_iter: int - Maximum number of iterations (default is 50).

    Returns:
        refined_u: np.array[2] - Refined motion vector.
    """
    half_window = window_size // 2

    # Compute image gradients using numerical_gradient
    Ix, Iy = np.gradient(img1)

    refined_u = np.zeros_like(initial_u)

    for idx, (x, y) in enumerate(points):
        u = initial_u[idx].T.copy()
        # Extract the patch around the current point
        x_start, x_end = int(x - half_window), int(x + half_window + 1)
        y_start, y_end = int(y - half_window), int(y + half_window + 1)

        Ix_patch = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_patch = Iy[y_start:y_end, x_start:x_end].flatten()
        I0_patch = img1[y_start:y_end, x_start:x_end].flatten()

        # Compute the A matrix and vector b
        A = np.array([[np.sum(Ix_patch ** 2), np.sum(Ix_patch * Iy_patch)],
                      [np.sum(Ix_patch * Iy_patch), np.sum(Iy_patch ** 2)]])

        # Check if A is invertible
        if np.linalg.det(A) < 1e-6:
            print(f"Skipping point {x, y} due to non-invertible A matrix.")
            refined_u[idx] = u
            continue # A is not invertible, stop refinement

        A_inv = np.linalg.inv(A)

        for _ in range(max_iter):
            # Compute warped points using the current motion (u)
            x_coords, y_coords = np.meshgrid(
                np.arange(x_start, x_end) + u[0],
                np.arange(y_start, y_end) + u[1]
            )

            # Interpolate the warped patch using bilinear interpolation
            warped_points = np.vstack((y_coords.ravel(), x_coords.ravel())).T
            warped_patch = int_bilineal(img2, warped_points)

            # Compute error between img1 and warped img2
            error_patch = warped_patch - I0_patch

            # Compute the b vector
            b = -np.array([np.sum(Iy_patch * error_patch), np.sum(Ix_patch * error_patch)])

            # Solve for delta motion
            delta_u = A_inv @ b

            # Update motion
            u += delta_u

            # Check for convergence
            if np.linalg.norm([delta_u]) < epsilon:
                break

        # Update refined motion vector
        refined_u[idx] = u

    return refined_u

##################### OTHERS #####################

def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix

    adapted from https://github.com/liruoteng/OpticalFlowToolkit/

    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d