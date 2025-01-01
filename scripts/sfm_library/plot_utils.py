#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.5
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2

def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def draw_epipolar_lines(img1, img2, matches1, matches2, F, refined_matches=None):
    """
    Draw epipolar lines on the images corresponding to the points
    Args:
        img1, img2: Images
        matches1, matches2: Corresponding points in the two images
        F: Fundamental matrix
    """
    def draw_lines(img, lines, pts, colors):
        '''
        Draw the epilines for the points in one image over the other image.
            img - image on which we draw the epilines for the points in img2.
            lines - corresponding epilines.
            pts - corresponding points.
            color - color of the epilines.
        '''
        r, c = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r, pt, color in zip(lines, pts, colors):
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(map(int, pt)), 5, color, -1)
        return img

    # Generate a random colro for each line
    num_lines = matches1.shape[0]
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_lines)]

    # Compute the epipolar lines in both images
    lines1 = cv2.computeCorrespondEpilines(matches2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines = draw_lines(img1, lines1, matches1, colors)

    lines2 = cv2.computeCorrespondEpilines(matches1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines = draw_lines(img2, lines2, matches2, colors)

    if refined_matches is not None:
        print ("Drawing refined matches, ", refined_matches.shape)
        for i in range(refined_matches.shape[0]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # draw a cross on the points
            img1_with_lines = cv2.drawMarker(img1_with_lines, (int(refined_matches[i, 0, 0]), int(refined_matches[i, 0, 1])), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            img2_with_lines = cv2.drawMarker(img2_with_lines, (int(refined_matches[i, 1, 0]), int(refined_matches[i, 1, 1])), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    plt.subplot(121), plt.imshow(img1_with_lines)
    plt.subplot(122), plt.imshow(img2_with_lines)
    plt.show()

def draw_matches(img1, img2, matches1, matches2, inliers):
    """
    Display matches and inliers between two images.
    Args:
        img1, img2: Input images.
        matches1, matches2: Matched points between two images (Nx2 arrays).
        inliers: List of indices of inliers.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_combined[:h1, :w1] = img1
    img_combined[:h2, w1:] = img2

    for i, (pt1, pt2) in enumerate(zip(matches1, matches2)):
        pt1 = tuple(np.round(pt1).astype(int))
        pt2 = tuple(np.round(pt2).astype(int) + np.array([w1, 0]))
        color = (0, 255, 0) if i in inliers else (0, 0, 255)
        cv2.line(img_combined, pt1, pt2, color, 1)
        cv2.circle(img_combined, pt1, 4, color, -1)
        cv2.circle(img_combined, pt2, 4, color, -1)

    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.show()

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)
        
def plot_sparse_optical_flow(img1, points_selected, flow_est_sparse, error_sparse, flow_est_sparse_norm, error_sparse_norm):
    """
    Plots the results of sparse optical flow and error with respect to ground truth.

    Parameters:
        img1 (ndarray): The first image frame.
        points_selected (ndarray): The selected points for sparse optical flow.
        flow_est_sparse (ndarray): The estimated sparse optical flow vectors.
        error_sparse (ndarray): The error vectors with respect to the ground truth.
        flow_est_sparse_norm (ndarray): Norms of the estimated sparse optical flow vectors.
        error_sparse_norm (ndarray): Norms of the error vectors.
    """
    fig, axs = plt.subplots(1, 2)

    # Plot the optical flow
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow')

    # Plot the error with respect to ground truth
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]), color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[1].title.set_text('Error with respect to GT')

    plt.show()

def generate_wheel(size):
    """
     Generate wheel optical flow for visualizing colors
     :param size: size of the image
     :return: flow: optical flow for visualizing colors
     """
    rMax = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    u = x - size / 2
    v = y - size / 2
    r = np.sqrt(u ** 2 + v ** 2)
    u[r > rMax] = 0
    v[r > rMax] = 0
    flow = np.dstack((u, v))

    return flow

def plot_dense_optical_flow(img1, img2, flow_12, flow_est, flow_error, error_norm, scale):
    """
    Plots the results of dense optical flow including ground truth, estimated flow, and error norm.

    Parameters:
        img1 (ndarray): The first image frame.
        img2 (ndarray): The second image frame.
        flow_12 (ndarray): The ground truth optical flow.
        flow_est (ndarray): The estimated optical flow.
        flow_error (ndarray): The error between the estimated and ground truth optical flow.
        error_norm (ndarray): Norms of the error vectors.
        scale (float): Scaling factor for flow visualization.
    """
    wheelFlow = generate_wheel(256)
    fig, axs = plt.subplots(2, 3)

    axs[0, 0].imshow(img1)
    axs[0, 0].title.set_text('image 1')
    axs[1, 0].imshow(img2)
    axs[1, 0].title.set_text('image 2')
    axs[0, 1].imshow(draw_hsv(flow_12 * np.bitwise_not(binUnknownFlow), scale))
    axs[0, 1].title.set_text('Optical flow ground truth')
    axs[1, 1].imshow(draw_hsv(flow_est, scale))
    axs[1, 1].title.set_text('LK estimated optical flow ')
    axs[0, 2].imshow(error_norm, cmap='jet')
    axs[0, 2].title.set_text('Optical flow error norm')
    axs[1, 2].imshow(draw_hsv(wheelFlow, 3))
    axs[1, 2].title.set_text('Color legend')
    axs[1, 2].set_axis_off()
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def draw_hsv(flow, scale):
    """
    Draw optical flow data (Middlebury format)
    :param flow: optical flow data in matrix
    :return: scale: scale for representing the optical flow
    adapted from https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * scale, 255)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return rgb


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    R_w_c1 = np.loadtxt('./p1/ext/R_w_c1.txt')
    R_w_c2 = np.loadtxt('./p1/ext/R_w_c2.txt')

    t_w_c1 = np.loadtxt('./p1/ext/t_w_c1.txt')
    t_w_c2 = np.loadtxt('./p1/ext/t_w_c2.txt')

    T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
    T_w_c2 = ensamble_T(R_w_c2, t_w_c2)


    K_c = np.loadtxt('./p1/ext/K.txt')

    X_A = np.array([3.44, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])

    print(np.array([[3.44, 0.80, 0.82]]).T) #transpose need to have dimension 2
    print(np.array([3.44, 0.80, 0.82]).T) #transpose does not work with 1 dim arrays

    # Example of transpose (need to have dimension 2)  and concatenation in numpy
    X_w = np.vstack((np.hstack((np.reshape(X_A,(3,1)), np.reshape(X_C,(3,1)))), np.ones((1, 2))))

    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    plotLabelled3DPoints(ax, X_w, ['A','C'], 'r', (-0.3, -0.3, 0.1)) # For plotting with labels (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    #Drawing a 3D segment
    draw3DLine(ax, X_A, X_C, '--', 'k', 1)

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread("Image1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)

    x1 = np.array([[527.7253,334.1983],[292.9017,392.1474]])

    plt.figure(1)
    plt.imshow(img1)
    plt.plot(x1[0, :], x1[1, :],'+r', markersize=15)
    plotLabeledImagePoints(x1, ['a','c'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    plotNumberedImagePoints(x1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
