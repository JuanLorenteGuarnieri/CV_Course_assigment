import numpy as np
import cv2
import geometry_utils as gu
import plot_utils as pu
import argparse
import os
import time
import matplotlib.pyplot as plt

def plot(X_w, T_list):
  fig3D = plt.figure(1)
  ax = plt.axes(projection='3d', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  pu.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
  for i, T in enumerate(T_list):
    T_aux = T.copy()
    if i > 0:
      T_aux[:, 3:] = -T_aux[:, 3:]
      T_aux[:, :3] = np.linalg.inv(T_aux[:, :3])

    pu.drawRefSystem(ax, T_aux, '-', f'C{i+1}')

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

def load_and_filter_matches(num_images, output_path, max_matches=200):
  matches = [[] for _ in range(num_images+1)]
  for i in range(2, num_images+1):
    path = os.path.join(output_path, f'1_{i}_matches.npz')
    npz = np.load(path)
    mask = npz['matches'] > -1
    idxs = npz['matches'][mask]
    matches1 = npz['keypoints0'][mask]
    matches2 = npz['keypoints1'][idxs]
    if len(matches[0]) == 0:
      matches[0] = matches1
    else:

      print(f"matches2 shape: {matches1.shape}")
      print(f"matches2 shape: {matches2.shape}")
      print(f"matches[0] shape: {matches[0].shape}")
      mask_existing = np.array([np.any(np.all(matches1[i] == matches[0], axis=1)) for i in range(matches1.shape[0])])
      matches1 = matches1[mask_existing]
      matches2 = matches2[mask_existing]
      mask_existing = np.array([np.any(np.all(matches1 == matches[0][i], axis=1)) for i in range(matches[0].shape[0])])
      for j in range(i-1):
        matches[j] = matches[j][mask_existing]
    matches[i-1] = matches2

  if len(matches[0]) > max_matches:
    for i in range(num_images):
      matches[i] = matches[i][:max_matches]

  # Swap matches[1] with matches[num_images-1] to use more different images in the first iteration
  # matches[1], matches[num_images-2] = matches[num_images-2], matches[1]

  return matches

def basic_transformations(K, matches):
    transformations = []
    transformations.append(np.eye(4))
    F, inliers = gu.ransac_fundamental_matrix(matches[0], matches[1])
    E = K.T @ F @ K
    R1, R2, t = gu.decompose_essential_matrix(E)

  # Choose the correct R and t by checking the validity of the solution
    if gu.is_valid_solution(R1, t, K, matches[0], matches[1]):
      R, t = R1, t
    elif gu.is_valid_solution(R2, t, K, matches[0], matches[1]):
      R, t = R2, t
    elif gu.is_valid_solution(R1, -t, K, matches[0], matches[1]):
      R, t = R1, -t
    else:
      R, t = R2, -t

  # Compute the transformation matrix
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    transformations.append(T)
    return transformations

def generate_transformation(i, K, distCoeffs, matches, X_w_):
    retval, rvec, tvec = cv2.solvePnP(X_w_, matches[i-1], K, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
    R, _ = cv2.Rodrigues(rvec)

    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = tvec.flatten()
    return T_wc

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Feature detection and matching')
  parser.add_argument('--K_path', type=str, required=True, help='Path to the calibration matrix K ex. ../../data/calibration/K.txt')
  parser.add_argument('--imgs_path', type=str, required=True, help='Path to the directory containing images ex. ../../data/raw/buildings/pilar/')
  parser.add_argument('--output_path', type=str, required=True, help='Path to the directory to save the results ex. ../../data/processed/buildings/pilar/')
  args = parser.parse_args()

  start_time = time.time()
  K = np.loadtxt(args.K_path)
  distCoeffs = np.loadtxt("../../data/calibration/radial_distorsion_z2.txt")
  print(f"K: {K}")
  # Load images and feature matches (for illustration, using SIFT here)
  valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
  image_files = [name for name in os.listdir(args.imgs_path) if name.lower().endswith(valid_extensions)]
  num_images = len(image_files)
  imgs = [cv2.cvtColor(cv2.imread(os.path.join(args.imgs_path, img)), cv2.COLOR_BGR2RGB) for img in image_files]
  print(f"Number of images: {num_images}")
  matches = load_and_filter_matches(num_images, args.output_path, max_matches=2000)
  print(f"Matches shape: {matches[0].shape}")

#carga en x los puntos 2D cada cada camara alojada en processed/.../x_p_opt_i.txt

  X_w = np.loadtxt("../../data/processed/buildings/pilar/X_w_opt.txt")
  
  T_list = []
  T_files = [f for f in os.listdir("../../data/processed/buildings/pilar/") if f.startswith("T_opt_") and f.endswith(".txt")]
  T_files.sort(key=lambda f: int(f.split('_')[2].split('.')[0]))  # Sort by the index number
  for T_file in T_files:
    T = np.loadtxt(os.path.join("../../data/processed/buildings/pilar/", T_file))
    T_list.append(T)
    

  P1 = gu.compute_projection_matrix(K, T_list[0][:3, :])
  P2 = gu.compute_projection_matrix(K, T_list[6][:3, :])
  
  x_p_opt_files = [f for f in os.listdir("../../data/processed/buildings/pilar/") if f.startswith("x_p_opt_") and f.endswith(".txt")]
  x_p_opt_files.sort(key=lambda f: int(f.split('_')[3].split('.')[0]))  # Sort by the index number
  x = []
  for x_file in x_p_opt_files:
    x.append(np.loadtxt(os.path.join("../../data/processed/buildings/pilar/", x_file)))
    
  
  threshold = 2.0  # Define a threshold for similarity

  def is_similar(p1, p2, threshold):
      return np.linalg.norm(p1 - p2) < threshold
  mask = np.array([np.any([is_similar(matches[0][i], x[6].T[j, :2], threshold) for j in range(x[0].T.shape[0])]) for i in range(matches[0].shape[0])])
  # matches = matches[:][mask]
  matches[0] = matches[0][mask]
  matches[1] = matches[1][mask]
  matches[2] = matches[2][mask]
  matches[3] = matches[3][mask]
  matches[4] = matches[4][mask]
  matches[5] = matches[5][mask]
  matches[6] = matches[6][mask]
  matches[7] = matches[7][mask]
  matches[8] = matches[8][mask]
  matches[9] = matches[9][mask]
  matches[10] = matches[10][mask]
  matches[11] = matches[11][mask]
  matches[12] = matches[12][mask]
  matches[13] = matches[13][mask]
  matches[14] = matches[14][mask]
  matches[15] = matches[15][mask]
  matches[16] = matches[16][mask]
  matches[17] = matches[17][mask]
  matches[18] = matches[18][mask]

  mask2 = np.array([np.any([is_similar(matches[0][j], x[6].T[i, :2], threshold) for j in range(matches[0].shape[0])]) for i in range(x[0].T.shape[0])])
  max_size = x[0].shape[1]
  x[0] = x[0].T[mask2].T
  x[6] = x[6].T[mask2].T

  X_w_ = gu.triangulate_points(P1, P2, x[0].T,  x[6].T)

  plot(X_w, T_list)

  for i in range(7, num_images + 1):
    T_list.append(generate_transformation(i, K, distCoeffs, matches, X_w_))
    
  x_p_opt = [gu.project_points(K, T_wc_opt, X_w) for T_wc_opt in T_list]
  

  # Save T_opt, X_w_opt, and x_p_opt to text files
  for i, T in enumerate(T_list):
    if i > 6:
      np.savetxt(os.path.join(args.output_path, f'T_opt_{i}.txt'), T)
  # np.savetxt(os.path.join(args.output_path, 'X_w_opt.txt'), X_w)
  for i, x_p in enumerate(x_p_opt):
    if i > 6:
      np.savetxt(os.path.join(args.output_path, f'x_p_opt_{i}.txt'), x_p)

  print("--- Total time: %s seconds ---" % (time.time() - start_time))
  plot(X_w, T_list)