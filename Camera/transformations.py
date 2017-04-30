import numpy as np
import cv2

class Perspective:
    def __init__(self, psource, pdest, src_size=(1280, 720), dst_size=(128, 256)):
        # Define source and destination image sizes
        self.dst_size = dst_size
        self.src_size = src_size
        self.src_points = psource
        self.dst_points = pdest
        # Compute perspective transform
        self.M = cv2.getPerspectiveTransform(psource, pdest)
        # Compute inverse perspective transform
        self.Minv = cv2.getPerspectiveTransform(pdest, psource)

    def apply(self, image):
        """Transformation from source to destination perspective.
        :param image: Non-warped image
        :return warped_image. """

        warped = cv2.warpPerspective(image, self.M, self.dst_size,
                                     flags=cv2.INTER_LINEAR)
        return warped

    def apply_inverse(self, image):
        """Transformation from destination to source perspective.
        :param image: warped image
        :return non-warped image. """
        # Inverse perspective transformation
        non_warped = cv2.warpPerspective(image, self.Minv, self.src_size,
                                     flags=cv2.INTER_LINEAR)
        return non_warped

if __name__ == "__main__":
    pass
    # Define image sizes
    input_image_size = (1280, 720)
    warped_image_size = (256, 512)

    # Define Perspective transformation
    yt = 460  # Y-top
    yb = 670  # Y-bottom
    src = np.array([[710, yt],  # Top-Right
                    [1080, yb],  # Bottom-Right
                    [200, yb],  # Bottom-Left
                    [569, yt]],  # Top-Left
                   dtype=np.float32)

    dst = np.array(
        [[warped_image_size[0] * 0.8, warped_image_size[1] * 0],   # * 0.1
         [warped_image_size[0] * 0.8, warped_image_size[1]],       # * 0.9
         [warped_image_size[0] * 0.2, warped_image_size[1]],       # * 0.9
         [warped_image_size[0] * 0.2, warped_image_size[1] * 0]],  # * 0.1
        dtype=np.float32)

    image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
    #image = cv2.imread(
    #    './test_images/project_video.mp4_border_of_dark_and_bright_road.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pt = Perspective(src, dst, input_image_size, warped_image_size)
    warped = pt.apply(image)
    cv2.imshow('image', warped)
    cv2.waitKey(5000)
    non_warped = pt.apply_inverse(warped)
    cv2.imshow('image', non_warped)
    cv2.waitKey(5000)
    # import matplotlib.pyplot as plt
    # plt.imshow(gradx, cmap='gray')

    cv2.destroyWindow('image')