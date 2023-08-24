import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d


def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:, :, 0], kernel, "valid")
    green = convolve2d(image[:, :, 1], kernel, "valid")
    blue = convolve2d(image[:, :, 2], kernel, "valid")
    return np.stack([red, green, blue], axis=2)


identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


def display_image(conv_im1, kernel):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(kernel, cmap="gray")
    ax[1].imshow(abs(conv_im1), cmap="gray")


def apply_kernel(img, kernel, display=False):
    conv_im1 = rgb_convolve2d(img, kernel)
    if display:
        display_image(conv_im1, kernel)
    else:
        return conv_im1


kernel0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# Edge Detection1
kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# Edge Detection2
kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# Bottom Sobel Filter
kernel3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# Top Sobel Filter
kernel4 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# Left Sobel Filter
kernel5 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# Right Sobel Filter
kernel6 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


# Sharpen
kernel7 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# Emboss
kernel8 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
# Box Blur
kernel9 = (1 / 9.0) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# Gaussian Blur 3x3
kernel10 = (1 / 16.0) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
# Gaussian Blur 5x5
kernel11 = (1 / 256.0) * np.array(
    [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
)
# Unsharp masking 5x5
kernel12 = -(1 / 256.0) * np.array(
    [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
)
kernel13 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
kernel14 = np.array([[-3, -2, 0], [-2, 1, 2], [0, 2, 3]])
kernel15 = np.array([[3, 2, 0], [2, 1, -2], [0, -2, -3]])
kernel16 = np.array([[0, 0, 0], [0, 1.5, 0], [0, 0, 0]])


# import the opencv library
import cv2
from time import sleep
from win32gui import SetForegroundWindow
from pywinauto.findwindows import find_window


vid = cv2.VideoCapture(0)


kernels = [
    kernel0,
    kernel1,
    kernel2,
    kernel3,
    kernel4,
    kernel5,
    kernel6,
    kernel7,
    kernel8,
    kernel9,
    kernel10,
    kernel11,
    kernel12,
    kernel13,
    kernel14,
    kernel15,
    kernel16,
]

i = 0

min_val = False
max_val = False

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = apply_kernel(frame, kernels[i], False).astype("float32")
    # frame = apply_kernel(frame, kernel13, False).astype("float32")
    frame /= 255
    if min_val == False:
        min_val = np.min(frame)
    if max_val == False:
        max_val = np.max(frame)

    frame = (frame - min_val) / (max_val - min_val)
    # frame[frame < 0.5] = 0.5
    # frame[frame > 0.5] = 1

    # frame[frame < 0] = 0
    # frame = abs(frame)
    # print(frame)
    # break

    # plt.imshow(frame)
    # plt.show()
    # break
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("n"):
        i = i + 1
        if i >= len(kernels):
            break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
plt.show()
