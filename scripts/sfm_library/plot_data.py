import numpy as np
import cv2
import geometry_utils as gu
import plot_utils as pu
import argparse
import os
import matplotlib.pyplot as plt


def estimate_pose_and_intrinsics(points1, points2, K1, distCoeffs, X_w, old_width, old_height):
  # Estimate the pose of the second camera
  print(f"points1 shape: {points1.shape}")
  print(f"points2 shape: {points2.shape}")
  print(f"X_w shape: {X_w.shape}")
  points2 = points2.astype(np.float32)
  X_w = X_w.astype(np.float32)

  if points2.shape[0] < 4 or X_w.T.shape[0] < 4:
    raise ValueError("Se requieren al menos 4 puntos para solvePnP.")
  retval, rvec, tvec = cv2.solvePnP(X_w.T, points2, K1, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
  R, _ = cv2.Rodrigues(rvec)
  T_wc = np.eye(4)
  T_wc[:3, :3] = R
  T_wc[:3, 3] = tvec.flatten()

  # Estimate the camera matrix K2
  K2 = None
  # K2, _ = cv2.calibrateCamera([points1], [points2], (old_width, old_height), None, None)

  return T_wc, K2

def plot(X_w, T_list):
  fig3D = plt.figure(1)
  ax = plt.axes(projection='3d', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  pu.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
  for i, T in enumerate(T_list):
    pu.drawRefSystem(ax, -T, '-', f'C{i+1}')

  ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')

  # Set equal scaling for all axes
  max_range = np.array([X_w[0, :].max()-X_w[0, :].min(), X_w[1, :].max()-X_w[1, :].min(), X_w[2, :].max()-X_w[2, :].min()]).max() / 2.0
  mid_x = (X_w[0, :].max()+X_w[0, :].min()) * 0.5
  mid_y = (X_w[1, :].max()+X_w[1, :].min()) * 0.5
  mid_z = (X_w[2, :].max()+X_w[2, :].min()) * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  print('Close the figure to continue. Left button for orbit, right button for zoom.')
  plt.show()
  
if __name__ == "__main__":
  X_w = np.loadtxt("../../data/processed/buildings/pilar/X_w_opt.txt")
  T_list = []
  for i in range(19):
    T = np.loadtxt(f"../../data/processed/buildings/pilar/T_opt_{i}.txt")
    T_list.append(T)
  plot(X_w, T_list)
  K = np.loadtxt("../../data/calibration/K_z2.txt")
  distCoeffs = np.loadtxt("../../data/calibration/radial_distorsion_z1.txt")
  
  # matches = [[] for _ in range(2)]
  # path = "../../data/processed/buildings/pilar/1_0_matches.npz"
  # npz = np.load(path)
  # mask = npz['matches'] > -1
  # idxs = npz['matches'][mask]
  # matches1 = npz['keypoints0'][mask]
  # matches2 = npz['keypoints1'][idxs]
  
  # img1 = cv2.cvtColor(cv2.imread('../../data/raw/buildings/pilar/1.jpg'), cv2.COLOR_BGR2RGB)
  # img2 = cv2.cvtColor(cv2.imread('../../data/raw/buildings/pilar/0.jpg'), cv2.COLOR_BGR2RGB)
  
  img1 = cv2.imread('../../data/raw/buildings/pilar/15.jpg')
  #escala la imagen 1 a la misma escala que la imagen 0
  img2 = cv2.imread('../../data/raw/buildings/pilar/0.jpg')
  # img1 = cv2.undistort(img1, K, distCoeffs)
  
  img1_eq = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
  img2_eq = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

  img1_eq = cv2.resize(img1_eq, (img2.shape[1], img2.shape[0]), interpolation = cv2.INTER_CUBIC)
  img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation = cv2.INTER_CUBIC)

  if False:
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_eq, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_eq, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply NNDR (nearest-neighbor distance ratio) for filtering matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.83 * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    matches1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    matches2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])
  else:
    path = f"../../data/processed/buildings/pilar/15_0_matches.npz"
    npz = np.load(path)
    mask = npz['matches'] > -1
    idxs = npz['matches'][mask]
    matches1 = npz['keypoints0'][mask]
    matches2 = npz['keypoints1'][idxs]

  print(f"Number of matches: {len(matches2)}")

  P1 = gu.compute_projection_matrix(K, T_list[0][:3, :])
  P2 = gu.compute_projection_matrix(K, T_list[1][:3, :])

#carga en x los puntos 2D cada cada camara alojada en processed/.../x_p_opt_i.txt
  x = []
  for i in range(19):
    x.append(np.loadtxt(f"../../data/processed/buildings/pilar/x_p_opt_{i}.txt"))

  print(f"matches1 shape: {matches1.shape}")
  print(f"x[0] shape: {x[0].shape}")
  print(f"matches1[0]: {matches1[0]}")
  print(f"x[0][:2, :].T[0]: {x[0][:2, :].T[0]}")
  
  
  H, inliers = gu.ransac_homography(matches1, matches2)
  
  # pu.draw_matches(img1, img2, matches1, matches2, inliers)

  # mask = np.array([np.any(np.all(matches1[i] == x[0].T[:, :2], axis=1)) for i in range(matches1.shape[0])])
  # matches1 = matches1[mask]
  # matches2 = matches2[mask]
  # mask2= np.array([np.any(np.all(matches1 == x[0][:2, :].T[i], axis=1)) for i in range(x[0][:2, :].T.shape[0])])
  # x[0] = x[0].T[mask2].T
  # x[1] = x[1].T[mask2].T
  
  # mask_existing = np.array([np.any(np.all(matches1[i] == matches[0], axis=1)) for i in range(matches1.shape[0])])
  # matches1 = matches1[mask_existing]
  # matches2 = matches2[mask_existing]
  # mask_existing = np.array([np.any(np.all(matches1 == matches[0][i], axis=1)) for i in range(matches[0].shape[0])])
  # for j in range(i-1):
  #   matches[j] = matches[j][mask_existing]

  threshold = 2  # Define a threshold for similarity

  def is_similar(p1, p2, threshold):
      return np.linalg.norm(p1 - p2) < threshold

  mask = np.array([np.any([is_similar(matches1[i], x[14].T[j, :2], threshold) for j in range(x[14].T.shape[0])]) for i in range(matches1.shape[0])])
  matches1 = matches1[mask]
  matches2 = matches2[mask]

  mask2 = np.array([np.any([is_similar(matches1[j], x[14].T[i, :2], threshold) for j in range(matches1.shape[0])]) for i in range(x[14].T.shape[0])])
  max_size = x[0].shape[1]
  x[0] = x[0].T[mask2].T
  x[14] = x[14].T[mask2].T

  X_w_ = gu.triangulate_points(P1, P2, x[0].T,  x[14].T)
  T_wc, K2 = estimate_pose_and_intrinsics(matches1[:max_size], matches2[:max_size], K, distCoeffs, X_w_.T, img2.shape[0], img2.shape[1])
  T_list_old_photo = []
  T_list_old_photo.append(T_list[0])
  T_list_old_photo.append(T_list[1])
  T_list_old_photo.append(T_list[2])
  T_list_old_photo.append(T_wc)
  T_list.append(T_wc)

  plot(X_w, T_list)
