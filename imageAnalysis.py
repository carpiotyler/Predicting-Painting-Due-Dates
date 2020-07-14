import numpy as np, argparse, imutils, cv2, math, cpbd
from PIL import Image, ImageStat

class ImageAnalysis:

    def __init__(self, image, url):
        statRGB = ImageStat.Stat(image)
        r,g,b = statRGB.rms
        perceivedBrightness = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
        colorfulness = self.image_colorfulness(np.array(image))
        cpbdSharpness = cpbd.compute(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY))
        
        self.data = {
            'url': url,
            'perceivedBrightness': perceivedBrightness,
            'colorfulness': colorfulness,
            'redLevel': r,
            'greenLevel': g,
            'blueLevel': b,
            'cpbdSharpness': cpbdSharpness
        }

    def getData(self):
        return self.data

    # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    def image_colorfulness(self, image):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype("float"))
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)