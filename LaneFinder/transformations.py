import numpy as np
import cv2

class Perspective:
    def __init__(self, psource, pdest, dst_size=(128, 256)):

        self.dst_size = dst_size
        # Compute perspective transform
        self.M = cv2.getPerspectiveTransform(psource, pdest)
        # Compute inverse perspective transform
        self.Minv = cv2.getPerspectiveTransform(pdest, psource)

    def apply(self, image):
        warped = cv2.warpPerspective(image, self.M, self.dst_size,
                                     flags=cv2.INTER_LINEAR)
        return warped

if __name__ == "__main__":
    pass
    # warp source points
    # TR, BR, BL, TL
    # These coordinates assumes that camera is exactly on horizontal center of vehicle
    # Original image size is 1280, 720
    yt = 460    # Y-top
    yb = 670    # Y-bottom
    src = np.array([[715, yt],     # Top-Right
                    [1080, yb],    # Bottom-Right
                    [200, yb],     # Bottom-Left
                    [565, yt]],    # Top-Left
                   dtype=np.float32)
    image_size = (256, 512)
    dst = np.array([[image_size[0]*0.8, image_size[1]*0.1],
                    [image_size[0]*0.8, image_size[1]],
                    [image_size[0]*0.2, image_size[1]],
                    [image_size[0]*0.2, image_size[1]*0.1]], dtype=np.float32)

    image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    #image = cv2.imread(
    #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pt = Perspective(src, dst, image_size)
    warped = pt.apply(image)

    cv2.imshow('image', warped)
    # import matplotlib.pyplot as plt
    # plt.imshow(gradx, cmap='gray')
    cv2.waitKey(5000)
    cv2.destroyWindow('image')