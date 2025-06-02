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

  K2, T_wc = gu.compute_intrinsic_matrix(X_w,points2)
  return T_wc, K2
  
  if points2.shape[0] < 4 or X_w.T.shape[0] < 4:
    raise ValueError("Se requieren al menos 4 puntos para solvePnP.")
  retval, rvec, tvec = cv2.solvePnP(X_w.T, points2, K1, None, flags=cv2.SOLVEPNP_EPNP)
  R, _ = cv2.Rodrigues(rvec)

  T_wc = np.zeros((3, 4))
  T_wc[:3, :3] = R
  T_wc[:3, 3] = tvec.flatten()

  # Estimate the intrinsic matrix K2 for the old photo
  scale_x = old_width / old_width
  scale_y = old_height / old_height
  K2 = K1.copy()
  K2[0, 0] *= scale_x
  K2[1, 1] *= scale_y
  K2[0, 2] *= scale_x
  K2[1, 2] *= scale_y

  return T_wc, K2

def estimate_pose_and_intrinsics2(points1, points2, K1, distCoeffs, X_w, old_width, old_height):
  P = gu.compute_projection_matrix_dlt(X_w, points2)
  R, t = gu.compute_camera_pose(P)
  T_wc = np.eye(4)
  T_wc[:3, :3] = R
  T_wc[:3, 3] = t.flatten()
  return T_wc, K1

def plot(X_w, T_list, X_w_2 = None, T_list_2 = None):
  fig3D = plt.figure(1)
  ax = plt.axes(projection='3d', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  # pu.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
  for i, T in enumerate(T_list):
    T_aux = T.copy()
    if i < 7:
      T_aux[:3, :3] = np.linalg.inv(T_aux[:3, :3])
    T_aux[:3, 3:] = -T_aux[:3, 3:]
    
    pu.drawRefSystem(ax, T_aux, '-', f'C{i+1}')
  
  if T_list_2 is not None:
    for i, T in enumerate(T_list_2):
      T_aux2 = T.copy()
      T_aux2[:3, 3:] = - T_aux2[:3, 3:] - T_list[6][:, 3:]
      T_aux2[:3, :3] = np.linalg.inv(T_aux[:3, :3])
      pu.drawRefSystem(ax, T_aux2, '-', f'C_Old')

  ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
  if X_w_2 is not None:
    ax.scatter(X_w_2[0, :], X_w_2[1, :], X_w_2[2, :], marker='.', color='r')

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
  for i in range(7):
    T = np.loadtxt(f"../../data/processed/buildings/pilar/T_opt_{i}.txt")
    T_list.append(T)

  plot(X_w, T_list)
  K = np.loadtxt("../../data/calibration/K_z2.txt")
  distCoeffs = np.loadtxt("../../data/calibration/radial_distorsion_z1.txt")

  # img1 = cv2.cvtColor(cv2.imread('../../data/raw/buildings/pilar/1.jpg'), cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(cv2.imread('../../data/raw/buildings/pilar/0.jpg'), cv2.COLOR_BGR2RGB)

  img1 = cv2.imread('../../data/raw/buildings/pilar/7.jpg')
  # img2 = cv2.imread('../../data/raw/buildings/pilar/0.jpg')

  path = f"../../data/processed/buildings/pilar/7_0_matches.npz"
  npz = np.load(path)
  mask = npz['matches'] > -1
  idxs = npz['matches'][mask]
  matches1 = npz['keypoints0'][mask]
  matches2 = npz['keypoints1'][idxs]

  print(f"Number of matches: {len(matches2)}")

  P1 = gu.compute_projection_matrix(K, T_list[0][:3, :])
  P2 = gu.compute_projection_matrix(K, T_list[6][:3, :])

#carga en x los puntos 2D cada cada camara alojada en processed/.../x_p_opt_i.txt
  x = []
  for i in range(7):
    x.append(np.loadtxt(f"../../data/processed/buildings/pilar/x_p_opt_{i}.txt"))

  print(f"matches1 shape: {matches1.shape}")
  print(f"x[0] shape: {x[0].shape}")
  print(f"matches1[0]: {matches1[0]}")
  print(f"x[0][:2, :].T[0]: {x[0][:2, :].T[0]}")

  threshold = 1.0  # Define a threshold for similarity

  def is_similar(p1, p2, threshold):
      return np.linalg.norm(p1 - p2) < threshold

  mask = np.array([np.any([is_similar(matches1[i], x[6].T[j, :2], threshold) for j in range(x[6].T.shape[0])]) for i in range(matches1.shape[0])])
  matches1 = matches1[mask]
  matches2 = matches2[mask]

  mask2 = np.array([np.any([is_similar(matches1[j], x[6].T[i, :2], threshold) for j in range(matches1.shape[0])]) for i in range(x[6].T.shape[0])])
  max_size = x[0].shape[1]
  x0 = x[0].T[~mask2].T
  x1 = x[6].T[~mask2].T
  x[0] = x[0].T[mask2].T
  x[6] = x[6].T[mask2].T


  sorted_indices = np.lexsort((matches1[:, 1], matches1[:, 0]))
  matches1 = matches1[sorted_indices]
  matches2 = matches2[sorted_indices]

  sorted_indices_x = np.lexsort((x[6][1, :], x[6][0, :]))
  x[0] = x[0][:, sorted_indices_x]
  x[6] = x[6][:, sorted_indices_x]

  X_w_ = gu.triangulate_points(P1, P2, x[0].T,  x[6].T)
  X_w_not = gu.triangulate_points(P1, P2, x0.T,  x1.T)
  X_w_not = np.vstack([X_w_not.T, np.ones((1, X_w_not.shape[0]))])
  X_w_old_photo = np.vstack([X_w_.T, np.ones((1, X_w_.shape[0]))])
  T_wc, K2 = estimate_pose_and_intrinsics(matches1[:max_size], matches2[:max_size], K, distCoeffs, X_w_.T, 1, 1)#img2.shape[0], img2.shape[1])
  T_list_old_photo = []
  T_list_old_photo.append(T_wc)
  print(f"T_wc: {T_wc}")

  plot(X_w_not, T_list, X_w_2=X_w_old_photo, T_list_2=T_list_old_photo)

  print(f"K2.shape: {K2.shape}")
  # P_old = gu.compute_projection_matrix(K2, T_wc)
  x_7_proy = gu.project_points(K, T_list[6], X_w_old_photo)
  x_7_proy_not = gu.project_points(K, T_list[6], X_w_not)
  # Multiplica la coordenada X por 100 de x_old_p -> x_old_p.shape: (3, 159)
  # T_aux2 = T.copy()
  # T_wc[:3, 3:] = -T_wc[:3, 3:]
  # T_wc[:3, :3] = np.linalg.inv(T_wc[:3, :3])
  
  T_7_0 = T_wc
  
  T_1_7 = np.vstack([T_list[6], [0, 0, 0, 1]])
  # T_1_0 = ?
  T_1_0 = T_1_7 @ T_7_0
  x_old_p = gu.project_points(K,  T_1_0, X_w_old_photo)

  # x_old_p = K2 @ np.eye(3, 4) @ np.linalg.inv(T_wc) @ X_w_old_photo


  print(f"matches2.shape: {matches2.T.shape}")
  print(f"x_old_p.shape: {x_old_p.shape}")
  # Imagen 1
  # Resize img1 to have a maximum dimension of 752 while maintaining aspect ratio
  max_dim = 752
  scale = max_dim / max(img1.shape[:2])
  img1 = cv2.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
  scale2 = max_dim / max(img2.shape[:2])
  img2 = cv2.resize(img2, (int(img2.shape[1] * scale2), int(img2.shape[0] * scale2)))

  #The elements of x_7_proy are not in order

  # Plot the results
  plt.figure(4)
  plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
  # pu.plotResidual(matches1.T, x_7_proy, 'k-')  # Residuals originales
  plt.plot(x_7_proy[0, :], x_7_proy[1, :], 'ro', label='Changes')
  plt.plot(x_7_proy_not[0, :], x_7_proy_not[1, :], 'bo', label='No changes')
  # plt.plot(matches1.T[0, :], matches1.T[1, :], 'rx')
  plt.legend()
  plt.title('Image 1')
  plt.show()
  
  plt.figure(4)
  plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
  pu.plotResidual(matches2.T, x_old_p, 'k-')  # Residuals originales
  plt.plot(x_old_p[0, :], x_old_p[1, :], 'bo', label='Projection')
  # plt.plot(x1_p_opt[0, :], x1_p_opt[1, :], 'go', label='Optimized Projection')  # Proyecciones optimizadas
  plt.plot(matches2.T[0, :], matches2.T[1, :], 'rx')
  # Convert matches2 to float
  matches2 = matches2.astype(np.float32)
  # pu.plotNumberedImagePoints(matches2[0:2, :].T, 'r', 4)
  plt.legend()
  plt.title('Image 1')
  plt.show()
