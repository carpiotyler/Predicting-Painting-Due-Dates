import urllib, json, os, math, numpy, threading, itertools, cv2
from PIL import Image, ImageStat
import requests
from io import BytesIO
from mySQL import MySQL
from imageAnalysis import ImageAnalysis

class FastWriteCounter(object):
    def __init__(self):
        self._number_of_read = 0
        self._counter = itertools.count()
        self._read_lock = threading.Lock()

    def increment(self):
        return next(self._counter)

    def value(self):
        with self._read_lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
        return value

mySql = MySQL()
result = mySql.query('SELECT  * FROM AMERICAN_PAINTINGS A WHERE (SELECT thinEdgePercentage FROM PAINTING_COMPUTED_VALUES P WHERE P.url = A.url) = 0')

paintings = []
for row in result:
    paintings.append({
        "url": row[0],
        "recordUrl": row[1],
        "title": row[2],
        "painter": row[3],
        "earliestDate": row[4],
        "latestDate": row[5]
    })

allPaintingData = []
index = FastWriteCounter()

def thread_function(index):
    while index.increment() < len(paintings):
        paintingIndex = index.value() - 1
        painting = paintings[paintingIndex]
        response = requests.get(painting['url'])
        img = Image.open(BytesIO(response.content))
        paintingData = ImageAnalysis(img, painting['url']).getData()
        allPaintingData.append(paintingData)
        print(f'Analyzed {paintingIndex} / {paintings.__len__()} paintings')
    
# 20 threads to analayze paintings
if __name__ == "__main__":
    threadList = []
    for i in range(1, 2):
        thread = threading.Thread(target=thread_function, args=(index,))
        thread.start()
        threadList.append(thread)
    while len(threadList) > 0:
        threadList.pop().join()
    
    replaceOrInsertString = """
    REPLACE INTO PAINTING_COMPUTED_VALUES
        (url, perceivedBrightness, colorfulness, redLevel, blueLevel, greenLevel, cpbdSharpness, numFacesFrontal, numFacesAlt, numFacesProfile, skyValue, thinEdgePercentage, medEdgePercentage, largeEdgePercentage)
    VALUES
    """
    comma = False
    for painting in allPaintingData:
        if comma == True:
            replaceOrInsertString += ','
        replaceOrInsertString += '("{url}", {perceivedBrightness}, {colorfulness}, {redLevel}, {greenLevel}, {blueLevel}, {cpbdSharpness}, {numFacesFrontal}, {numFacesAlt}, {numFacesProfile}, {skyValue}, {thinEdgePercentage}, {medEdgePercentage}, {largeEdgePercentage})'.format(
            url=painting['url'], 
            perceivedBrightness=painting['perceivedBrightness'], 
            colorfulness=painting['colorfulness'], 
            redLevel=painting['redLevel'], 
            greenLevel=painting['greenLevel'], 
            blueLevel=painting['blueLevel'], 
            cpbdSharpness=painting['cpbdSharpness'],
            numFacesFrontal=painting['numFacesFrontal'],
            numFacesAlt=painting['numFacesAlt'],
            numFacesProfile=painting['numFacesProfile'],
            skyValue=painting['skyValue'],
            thinEdgePercentage=painting['thinEdgePercentage'],
            medEdgePercentage=painting['medEdgePercentage'],
            largeEdgePercentage=painting['largeEdgePercentage']
            )
        comma = True

    mySql.write(replaceOrInsertString)

