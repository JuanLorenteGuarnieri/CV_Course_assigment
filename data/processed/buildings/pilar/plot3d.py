from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

points = []

with open('./points3D.txt', 'r') as f:
  for line in f:
    parts = line.strip().split(' ')
    if len(parts) != 5:
      continue
    _, x, y, z, _ = parts
    points.append((float(x), float(y), float(z)))

if points:
  xs, ys, zs = zip(*points)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Scatter Plot of Points')
  cbar = plt.colorbar(sc, ax=ax, pad=0.1)
  cbar.set_label('Z value')
  plt.show()
else:
  print("No valid points found in points3D.txt")
