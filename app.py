import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours


# Loading the Image
image = plt.imread("image.png")

# Converting the image into GrayScale
def rgb_to_gray(img: np.array) -> np.array:
    r, g, b, = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

# Reversing the channels Increases the quality as well
image_gray = rgb_to_gray(image[..., ::-1])

# Convolution
def convolve2d(src: np.array, k: np.array) -> np.array:
    ih, iw = src.shape
    kh, kw = k.shape
    oh, ow = ih - kh + 1 , iw - kw + 1
    output = np.zeros((oh, ow))

    for x in range(ow):
        for y in range(oh):
            output[y, x] = np.sum(src[y:y + kh, x:x + kw] * k)

    return output 

# Gaussian Filter
def gaussian_filter(img: np.array, kernel_size: tuple[int, int], sigma: float) -> np.array:
    kernel = np.fromfunction(
        lambda x, y : np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2),
        kernel_size
    )
    k = kernel / np.sum(kernel)
    return convolve2d(img, k)

blured_image = gaussian_filter(image_gray, (3,3), 1)

# Sobel Edge Detection
def sobel_edge_detection(img: np.array):
    Gx = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )
    Gy = Gx.T
    gradient_magnitude = np.sqrt(convolve2d(img, Gx) ** 2 + convolve2d(img, Gy) ** 2)
    return 255 * gradient_magnitude / np.max(gradient_magnitude)

# Thresholding
def threshold(img: np.array, thresh: float) -> np.array:
    return np.where(img > thresh, 255, 0)


# Detecting the edges
edges = sobel_edge_detection(blured_image)
thresh = threshold(edges, 10)

# FiDetectingnding the Counrours
countors = find_contours(thresh)

# Finding the countour with max coords point
max_polygon = max(countors, key=len)

# Ploting the result
fig, axes = plt.subplots(1, 3, figsize=(15,8))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(thresh, cmap="gray")
axes[1].set_title("Detected Edges")

axes[2].imshow(np.zeros(thresh.shape), cmap="gray")
axes[2].plot(max_polygon[:, 1], max_polygon[:, 0], linewidth=1, color='red')
axes[2].set_title("Biggest Polygon")

plt.show()