import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    # out = None

    ### YOUR CODE HERE
    # dimming function
    tempImage = 0.5 * np.square(image)
    return tempImage
    ### END YOUR CODE

    # return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    # conversion using library function
    tempImage = np.array(image)
    out = color.rgb2gray(tempImage)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel out

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    tempImage = np.array(image)
    # isolate the channels
    r = tempImage[:, :, 0]
    g = tempImage[:, :, 1]
    b = tempImage[:, :, 2]
    
    # replace the required channel with array of 0s, and merge channels back
    if channel == 'R':
        tempImage2 = np.stack([np.zeros_like(r), g, b], axis=-1)
    elif channel == 'G':
        tempImage2 = np.stack([r, np.zeros_like(g), b], axis=-1)
    elif channel == 'B':
        tempImage2 = np.stack([r, g, np.zeros_like(b)], axis=-1)
    else:
        pass
    ### END YOUR CODE

    return tempImage2


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel out.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel out.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    # isolate channels using input
    if channel == 'H':
        out = hsv[:, :, 0]
    elif channel == 'S':
        out = hsv[:, :, 1]
    elif channel == 'V':
        out = hsv[:, :, 2]
    else:
        pass
    
    return out
    ### END YOUR CODE


def mixImages(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    image1 = np.array(image1)
    image2 = np.array(image2)

    # gets width in case they do not match
    w = min(image1.shape[1], image2.shape[1]) // 2

    temp1 = rgb_exclusion(image1, channel1)
    temp2 = rgb_exclusion(image2, channel2)
    left = temp1[:, :w, :]
    right = temp2[:, w:, :]
    # concatenate the two temp images
    out = np.concatenate((left, right), axis = 1)
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    image = np.array(image)

    #TL quadrant
    tl = rgb_exclusion(image, 'R')

    #TR quadrant
    tr = dim_image(image)

    #BL quadrant
    bl = imageBrighten(image)
    
    #BR quadrant
    br = rgb_exclusion(image, 'R')

    # get dimensions
    w = image.shape[1] // 2
    h = image.shape[0] // 2

    # merge tl and tr horizontally
    temp1 = np.concatenate((tl[:, :w, :], tr[:, w:, :]), axis=1)
    
    # merge bl and br horizontally
    temp2 = np.concatenate((bl[:, :w, :], br[:, w:, :]), axis=1)
    
    # merge all vertically
    out = np.concatenate((temp1[:h, :, :], temp2[h:, :, :]), axis=0)

    ### END YOUR CODE
    return out

def imageBrighten(image):
    image = np.array(image)
    out = np.power(image, 0.5)
    return out