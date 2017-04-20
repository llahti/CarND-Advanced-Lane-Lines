def color_threshold(image):
    # Test out HSL thresholding
    import numpy as np
    import cv2

    img = image

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # Detect and threshold yellow line on Hue-Channel
    thresh = (20, 70)
    gray = H
    binary_yh = np.zeros_like(gray)
    binary_yh[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    # Detect and threshold yellow and white line on Saturation-Channel
    thresh = (110, 255)
    gray = S
    binary_yws = np.zeros_like(gray)
    binary_yws[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    # Detect and threshold white line on Lightness-Channel
    thresh = (200, 255)
    gray = L
    binary_wl = np.zeros_like(gray)
    binary_wl[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    # Find yellow lines
    binary_yellow = np.zeros_like(gray)
    binary_yellow[(binary_yws == 1)& (binary_yh == 1)] = 1

    # Find white lines
    binary_white = np.zeros_like(gray)
    binary_white[(binary_yws == 1)& (binary_wl == 1)] = 1
    #binary_white = binary_yws + binary_wl

    # Combine Yellow and white lines
    binary_both = np.zeros_like(gray)
    binary_both[(binary_yellow==1)|(binary_white==1)]=1

    return binary_both