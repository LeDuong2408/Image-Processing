import numpy as np
import cv2

L = 256

def Negative(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        # Convert color image to grayscale
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = s
    return imgout

def Logarit(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L - 1) / np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c * np.log(1 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

# Add similar modifications for the remaining functions...

# (omitting Power function for brevity)

def PiecewiseLinear(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < r1:
                s = s1 / r1 * r
            elif r < r2:
                s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
            else:
                s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
            imgout[x, y] = np.uint8(s)
    return imgout

# (omitting Histogram function for brevity)

def HistEqual(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    return imgout

# (omitting HistEqualColor, LocalHist, HistStat, MyBoxFilter, BoxFilter, and Threshold functions for brevity)

# Update the rest of the functions similarly...

def Sharpen(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    gx = cv2.Sobel(imgin, cv2.CV_32FC1, dx=1, dy=0)
    gy = cv2.Sobel(imgin, cv2.CV_32FC1, dx=0, dy=1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout
