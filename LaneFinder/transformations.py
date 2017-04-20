import numpy as np
import cv2
class Perspective:
    def __init__(self, psource, pdest, sample_img):

        self.image_size = (sample_img[1], sample_img[0])
        # Compute perspective transform
        self.M = cv2.getPerspectiveTransform(psource, pdest)
        # Compute inverse perspective transform
        self.Minv = cv2.getPerspectiveTransform(pdest, psource)

    def apply(self, image):
        return cv2.warpPerspective(image, self.M, self.image_size, flags=cv2.INTER_LINEAR)

if __name__ == "__main__":
    pass
# warp source points
# TR, BR, BL, TL
src = np.array([[650, 428],
                [1042, 678],
                [252, 686],
                [627, 428]], dtype=np.float32)

dst = np.array([[1042, 0],
                [1042, 1280],
                [252, 1280],
                [252, 9]], dtype=np.float32)

image = cv2.imread('./test_images/challenge_video.mp4_tarmac_edge_separates.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pt = Perspective(src, dst, image)
pt.apply(image)
cv2.imshow('image', image)
# import matplotlib.pyplot as plt
# plt.imshow(gradx, cmap='gray')
cv2.waitKey(5000)
cv2.destroyWindow('image')