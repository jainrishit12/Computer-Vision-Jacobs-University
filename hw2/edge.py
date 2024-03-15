import numpy as np
from collections import deque

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of temp at each
    pixel.

    Args -
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns -
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    1
    input_height, input_width = padded.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate output dimensions
    output_height = int((input_height - kernel_height) / 1) + 1
    output_width = int((input_width - kernel_width) / 1) + 1
    
    # Initialize output array
    output_array = np.zeros((output_height, output_width))
    
    # Perform convolution with zero-padding
    for i in range(0, input_height - kernel_height + 1, 1):
        for j in range(0, input_width - kernel_width + 1, 1):
            # Extract the region of interest from padded input array
            roi = padded[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and sum
            output_array[i // 1, j // 1] = np.sum(roi * kernel)
    ### END YOUR CODE
    return output_array

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE]
    k = (size - 1) // 2
    # size = 2k + 1 -> k = (size - 1) // 2

    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(- (((i - k) ** 2 + (j - k) ** 2) /(2 * (sigma ** 2))))
    
    ### END YOUR CODE
    return kernel

def partial_x(image):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    ### YOUR CODE HERE
    Dx = np.array([[-1, 0, 1]]) / 2
    out = conv(image, Dx)

    ### END YOUR CODE

    return out

def partial_y(image):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        image: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    ### YOUR CODE HERE
    Dy = np.array([[-1], [0], [1]]) / 2
    out = conv(image, Dy)
    ### END YOUR CODE

    return out

def gradient(image):
    """ Returns gradient magnitude and direction of input img.

    Args:
        image: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(image.shape)
    theta = np.zeros(image.shape)

    ### YOUR CODE HERE
    Gx = partial_x(image)
    Gy = partial_y(image)

    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)
    theta[theta < 0] += (np.pi * 2)
    theta = np.rad2deg(theta)
    # this converts the negative angles to a range between 0 and 2 pi, as tan(theta) is periodic
    # then we convert radians to degrees
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    theta = theta * np.pi / 180
    # converted back to rad for np.sin and np.cos
    dx = ((np.cos(theta) + 0.5) // 1).astype(int)
    dy = ((np.sin(theta) + 0.5) // 1).astype(int)
    
    padded = np.pad(G, ((1, 1), (1, 1)), mode='constant')
    tempI = np.indices((H, W)) + 1
    temp2 = (G >= padded[tempI[0] + dy, tempI[1] + dx]) & (G >= padded[tempI[0] - dy, tempI[1] - dx])
    out[temp2] = G[temp2]
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array which represents strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)
    # np.bool changed to bool due to deprecated asset error:
    # AttributeError: module 'numpy' has no attribute 'bool'.
    # `np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    # The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    # https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

    ### YOUR CODE HERE
    # edges stronger than threshold are strong edges
    strong_edges[img >= high] = True
    # edges between low and high threshold are weak edges
    weak_edges[(img >= low) & (img < high)] = True
    # all other edges (less than low threshold) are 0 or False
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def dfs(y, x, weak, strong, H, W):
    stack = deque([(y, x)])
    while stack:
        i, j = stack.pop()
        for k, l in get_neighbors(i, j, H, W):
            if weak[k, l] and not strong[k, l]:
                strong[k, l] = True
                stack.append((k, l))

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    for y in range(H):
        for x in range(W):
            if edges[y, x]:
                dfs(y, x, weak_edges, edges, H, W)
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # smoothed image
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)

    # gradient extraction
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    G, theta = gradient(smoothed)

    # edge detection, suppression and linking
    nms = non_maximum_suppression(G, theta)
    strong, weak = double_thresholding(nms, high, low)
    edge = link_edges(strong, weak)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2.0) + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    xS = xs.reshape(-1, 1)
    yS = ys.reshape(-1, 1)
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordinate
    ### YOUR CODE HERE
    r_cos = xS * cos_t.reshape(1, -1)
    r_sin = yS * sin_t.reshape(1, -1)
    r = (r_cos + r_sin).reshape(-1) + diag_len
    r = r.astype(int)

    theta_indices = np.tile(np.arange(len(thetas)), len(xs))
    accumulator_indices = (r, theta_indices)
    np.add.at(accumulator, accumulator_indices, 1)
    ### END YOUR CODE

    return accumulator, rhos, thetas
