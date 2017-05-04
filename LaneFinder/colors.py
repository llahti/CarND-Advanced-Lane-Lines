from scipy.special import expit # expit is  a sigmoid function
import numpy as np
import cv2


def bgr2cpaces(bgr):
    """This function converts float BGR image into color spaces.
    Output 'tensor' is shape (W, H, N) where in is number of color spaces.
    
    Colorplanes are in following order: 
    RGB_R, RGB_G, RGB_B
    HLS_H, HLS_L, HLS_S
    Lab_L, Lab_a, Lab_b,
    Luv_L, Luv_u, Luv_v
    """

    # 1. BGR split
    rgb_b, rgb_g, rgb_r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    # 2. HLS split
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hls_h, hls_l, hls_s = hls[:, :, 0], hls[:, :, 1], hls[:,:, 2]
    # 3. Lab split
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lab_l, lab_a, lab_b = lab[:, :, 0], lab[:, :, 1], lab[:,:, 2]
    # 4. LUV split
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2Luv)
    luv_l, luv_u, luv_v = luv[:, :, 0], luv[:, :, 1], luv[:,:, 2]

    cpaces = np.dstack((rgb_r, rgb_g, rgb_b,
                         hls_h, hls_l, hls_s,
                         lab_l, lab_a, lab_b,
                         luv_l, luv_u, luv_v))
    return cpaces


def bgr_uint8_2_cpaces_float32(bgr):
    """This function converts float BGR image into color spaces.
    Output 'tensor' is shape (W, H, N) where in is number of color spaces.

    Colorplanes are in following order: 
    RGB_R, RGB_G, RGB_B
    HLS_H, HLS_L, HLS_S
    Lab_L, Lab_a, Lab_b,
    Luv_L, Luv_u, Luv_v
    """
    # Convert to float 32
    bgr = bgr.copy().astype('float32') / 255

    # 1. BGR split
    rgb_b, rgb_g, rgb_r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    # 2. HLS split
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hls_h, hls_l, hls_s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    # 3. Lab split
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lab_l, lab_a, lab_b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    # 4. LUV split
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2Luv)
    luv_l, luv_u, luv_v = luv[:, :, 0], luv[:, :, 1], luv[:, :, 2]

    cpaces = np.dstack((rgb_r, rgb_g, rgb_b,
                        hls_h, hls_l, hls_s,
                        lab_l, lab_a, lab_b,
                        luv_l, luv_u, luv_v))
    return cpaces


def normalize_plane(color_plane):
    """This function normalizes array values between values 0...1"""
    arr = color_plane.copy()
    # Shift minimum to zero
    arr_min = arr.min()
    arr = arr - arr_min
    # Scale maximum to 1
    arr_max = arr.max()
    arr = arr / arr_max

    return arr


def yellow_centric_hls_h(hls_h, gamma=8):
    """Turns h plane of HLS-colorspace image into yellow centric where yellow 
    colors have values near 1.
    :param hls_h: HLS color space's H-plane.
    :param gamma: Defines the spread. I.e. how much near by colors are taken into account. Low gamma values take only pure yellow color where as higher gamma takes also near by colors.
    :return: image where 1 is yellow color and 0 is most far away from yellow."""

    # 1. In HLS_h color plane yellow color is centered around value 40.
    # 2. Values are subtracted by 40 to get yellow color near zero. TODO: Check exact value of yellow!
    # 3. Division by gamma defines how sensitive we are to near by colors. Low gamma means that we are focusing on pure yellow where as high gamma includes also near by colors..
    # 4. Sigmoid functions centers yellow color around 0.5 and we compensate it by substracting 0.5 --> centered again to zero
    # 5. Multiply absolute value by 2 to get range 0..1
    # 6. subtract value from 1 to flip yellow colors near 1
    #
    yellow = 1 - (np.abs(expit((hls_h - 40) / gamma) - 0.5) * 2)  # Using sigmoid
    return yellow


def yellow_color_plane(cplanes):
    """This function returns a yellow centric color plane, i.e. amount of yellow color.
    :param cplanes: Color planes tensor
    :return: yellow color plane
    """
    hls_h = yellow_centric_hls_h(cplanes[:, :, 3])
    lab_b = normalize_plane(cplanes[:, :, 8].copy())
    luv_v = normalize_plane(cplanes[:, :, 11].copy())
    return hls_h * lab_b * luv_v


def white_centric_lab_a(lab_a, gamma=1):
    white = 1 - (np.abs(expit(lab_a / gamma) - 0.5) * 2)
    return white


def white_color_plane(cplanes):
    """This function returns a white centric color plane. i.e. amount of whiteness.
    :param cplanes: Color planes tensor
    :return: white color plane
    """
    # Following color planes are potential indicators of whiteness
    # hls-l, hls-s, rgb, inverse lab-a, inverse luv-u
    hls_l_norm = normalize_plane(cplanes[:, :, 4])
    hls_l = np.power(cplanes[:, :, 4], 2)
    hls_s = normalize_plane(cplanes[:, :, 5])
    rgb_r = normalize_plane(cplanes[:, :, 0])
    rgb_g = cplanes[:, :, 1]
    rgb_b = normalize_plane(cplanes[:, :, 2])
    lab_a_norm = normalize_plane(
        white_centric_lab_a(cplanes[:, :, 7], gamma=1))
    lab_a = white_centric_lab_a(cplanes[:, :, 7], gamma=1)

    return normalize_plane(hls_l * rgb_r)
