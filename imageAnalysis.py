import numpy as np, argparse, imutils, cv2, math, cpbd
from PIL import Image, ImageStat
from skydetectorConverted import detect_sky

class ImageAnalysis:

    def __init__(self, image, url):
        npimg = np.array(image)
        statRGB = ImageStat.Stat(image)
        r,g,b = statRGB.rms
        perceivedBrightness = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
        colorfulness = self.image_colorfulness(npimg)
        greyImg = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)

        # cpbd algorithm
        cpbdSharpness = cpbd.compute(greyImg)

        # num frontal faces in the image
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces_frontal = face_cascade.detectMultiScale(greyImg, 1.1, 35)
        # num face 'alt' faces in the image
        face_alt_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces_alt = face_alt_cascade.detectMultiScale(greyImg, 1.1, 20)
        # num face profile faces in the image
        face_profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
        faces_profile = face_profile_cascade.detectMultiScale(greyImg, 1.1, 20)

        thinEdges = cv2.Canny(greyImg, 20, 60)
        medEdges = cv2.Canny(greyImg, 50, 100)
        largeEdges = cv2.Canny(greyImg, 100, 200)
        # percentage of painting that is thin, med, large edges (an estimate for number of small brushstrokes)
        pixelCount = len(thinEdges) * len(thinEdges[0])
        thinEdgePixels = 0
        for row in thinEdges:
            for col in row:
                if col > 0:
                    thinEdgePixels += 1
        
        medEdgePixels = 0
        for row in medEdges:
            for col in row:
                if col > 0:
                    medEdgePixels += 1
        
        largeEdgePixels = 0
        for row in largeEdges:
            for col in row:
                if col > 0:
                    largeEdgePixels += 1
                
        skyValue = 0
        try:
            skyValue = detect_sky(npimg)
        except:
            print('Error during sky detection for: ' + url)
            
        self.data = {
            'url': url,
            'perceivedBrightness': perceivedBrightness,
            'colorfulness': colorfulness,
            'redLevel': r,
            'greenLevel': g,
            'blueLevel': b,
            'cpbdSharpness': cpbdSharpness,
            'numFacesFrontal': len(faces_frontal),
            'numFacesAlt': len(faces_alt),
            'numFacesProfile': len(faces_profile),
            'skyValue': skyValue,
            'thinEdgePercentage': thinEdgePixels / pixelCount * 100.0,
            'medEdgePercentage': medEdgePixels / pixelCount * 100.0,
            'largeEdgePercentage': largeEdgePixels / pixelCount * 100.0
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