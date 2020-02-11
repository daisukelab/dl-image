import numpy as np
import cv2


def get_edge(img):
    """
    Calculate edges on the image.
    Thanks to https://qiita.com/supersaiakujin/items/494cc16836738b5394c8

    Args:
        img: numpy array in HWC format.

    Returns:
        2D uint8 image, non-zero elements are the edge pixels.
        > img[get_edge(img) > 0] = 0 # Put black edge pixels on the image.
    """
    org_shape = img.shape

    # preprocess: blur image to suppress small noise edges
    blur_kernel = np.array([1,1,1, 1,3,1, 1,1,1]) / 11
    img = cv2.filter2D(img, 256, blur_kernel)

    # preprocess: sharpen edges
    k = -1.
    sharpningKernel8 = np.array([k, k, k, k, 9.0, k, k, k, k])
    img = cv2.filter2D(img, 256, sharpningKernel8)

    # extract edges by image pyramid
    L = 2
    tmp = img.copy()
    edges = [cv2.Canny(tmp.astype(np.uint8),100,200 )]
    for idx in range(L-1):
        tmp = cv2.pyrDown(tmp)
        edges.append(cv2.Canny(tmp.astype(np.uint8),100,200 ))

    # recover size (edges[1] is currently half of org_img)
    edge = cv2.resize(edges[1], org_shape[:2])

    return edge


def simplify_by_kmeans(img, K=10, grayscale=True):
    """
    Simplify image by using kmeans clustering colors.

    Args:
        img: Input image in HWC format.
        K: Number of clusters (= colors).
        grayscale: Set false to handle color image as it is.

    Returns:
        RGB numpy image array [HWC]
    """
    # dilation & erusion to connect small piecies all together
    kernel = np.ones((10,10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # apply kmeans()
    Z = img.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))

    return img


def abstract_image(img_file, K=10, grayscale=True, add_edge=True, resize=None):
    """
    """
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize)
    org_img = img.copy()

    # apply grayscale to make clustering easier
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # apply clustering
    if K is not None:
        img = simplify_by_kmeans(img, K=K, grayscale=grayscale)

    # emphasize edge
    if add_edge:
        edge = get_edge(org_img)
        img[edge > 0] = 0

    # convert back to color image
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)        
    
    return org_img, img
