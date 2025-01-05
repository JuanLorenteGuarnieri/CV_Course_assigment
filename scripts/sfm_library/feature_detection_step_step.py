import numpy as np
import cv2
import geometry_utils as gu
import plot_utils as pu
import argparse
import os
import matplotlib.pyplot as plt

def plot(X_w, T_list):
  fig3D = plt.figure(1)
  ax = plt.axes(projection='3d', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  pu.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
  for i, T in enumerate(T_list):
    pu.drawRefSystem(ax, T, '-', f'C{i+1}')

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

  np.random.seed(42)  # Set seed for reproducibility
  for i in range(num_images):
    # if len(matches[i]) > max_matches:
    #   indices = np.random.choice(len(matches[i]), max_matches, replace=False)
    #   matches[i] = matches[i][indices]
    matches[i] = matches[i][:max_matches]

  # Swap matches[1] with matches[num_images-1]
  matches[1], matches[num_images-1] = matches[num_images-1], matches[1]
    
  return matches

def save_colmap_files(output_path, T_opt, X_w_opt, x_p_opt):
    # Save cameras.txt
    with open(os.path.join(output_path, 'cameras.txt'), 'w') as f:
      for i, T in enumerate(T_opt):
        f.write(f"{i+1} PINHOLE {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

    # # Save images.txt
    # with open(os.path.join(output_path, 'images.txt'), 'w') as f:
    #   for i, T in enumerate(T_opt):
    #     R = T[:3, :3]
    #     t = T[:3, 3]
    #     q = gu.rotation_matrix_to_quaternion(R)
    #     f.write(f"{i+1} {' '.join(map(str, q))} {' '.join(map(str, t))} {i+1}.jpg\n")

    # Save points3D.txt
    with open(os.path.join(output_path, 'points3D.txt'), 'w') as f:
      for i, X in enumerate(X_w_opt.T):
        f.write(f"{i+1} {' '.join(map(str, X[:3]))} 1.0\n")
        
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
    T = gu.ensamble_T(R, t)
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

  K = np.loadtxt(args.K_path)
  distCoeffs = np.loadtxt("../../data/calibration/radial_distorsion_z2.txt")
  print(f"K: {K}")
  # Load images and feature matches (for illustration, using SIFT here)
  valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
  image_files = [name for name in os.listdir(args.imgs_path) if name.lower().endswith(valid_extensions)]
  num_images = len(image_files)
  imgs = [cv2.cvtColor(cv2.imread(os.path.join(args.imgs_path, img)), cv2.COLOR_BGR2RGB) for img in image_files]

  print(f"Number of images: {num_images}")
  matches = load_and_filter_matches(num_images, args.output_path, max_matches=500)
  print(f"Matches shape: {matches[0].shape}")

  # translations = [np.random.randn(3) * 0.5 for _ in range(num_images)]

  # rotations = [np.eye(3) for _ in range(num_images)]

  # transformations = [gu.ensamble_T(rotations[i], translations[i]) for i in range(num_images)]

  # Initialize transformations list
  print(f"Estimating transformations")
  transformations = basic_transformations(K, matches)

  print(f"Transformations computed: {len(transformations)}")

  P1 = gu.compute_projection_matrix(K, transformations[0][:3, :])
  P2 = gu.compute_projection_matrix(K, transformations[1][:3, :])

  x = [np.hstack([matches[i], np.ones((matches[i].shape[0], 1))]).T for i in range(num_images)]

  X_w_ = gu.triangulate_points(P1, P2, x[0].T,  x[1].T)
  X_w = np.vstack([X_w_.T, np.ones((1, X_w_.shape[0]))])

  for i in range(3, num_images + 1):
    print(f"X_w shape: {X_w[:3, :].T.shape}")
    print(f"X_w_ shape: {X_w_.shape}")
    transformations.append(generate_transformation(i, K, distCoeffs, matches, X_w_))#X_w[:3, :].T))

    x_no_opt = [gu.project_points(K, T_wc, X_w) for T_wc in transformations]

    plot(X_w, transformations)

    T_opt, X_w_opt = gu.run_bundle_adjustment(transformations, K, X_w, x[:len(transformations)],max_iter=500, tol=1e-5)

    x_p_opt = [gu.project_points(K, T_wc_opt, X_w_opt) for T_wc_opt in T_opt]
  
    transformations = T_opt
    X_w = X_w_opt
    X_w_ = X_w[:3, :].T

    plot(X_w_opt, T_opt)

  # Save T_opt, X_w_opt, and x_p_opt to text files
  for i, T in enumerate(T_opt):
    np.savetxt(os.path.join(args.output_path, f'T_opt_{i}.txt'), T)
  np.savetxt(os.path.join(args.output_path, 'X_w_opt.txt'), X_w_opt)
  # np.savetxt(os.path.join(args.output_path, 'x_p_opt.txt'), x_p_opt)
  for i, x_p in enumerate(x_p_opt):
    np.savetxt(os.path.join(args.output_path, f'x_p_opt_{i}.txt'), x_p)

  save_colmap_files(args.output_path, T_opt, X_w_opt, x_p_opt)
