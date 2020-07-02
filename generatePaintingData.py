import urllib, json, os, math, numpy, threading, itertools, cv2
from PIL import Image, ImageStat
import requests
from io import BytesIO

class FastWriteCounter(object):
    def __init__(self):
        self._number_of_read = 0
        self._counter = itertools.count()
        self._read_lock = threading.Lock()

    def increment(self):
        next(self._counter)

    def value(self):
        with self._read_lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
        return value

paintings = json.load(open('paintings.json'))

allPaintingData = []
index = FastWriteCounter()

def thread_function(index):
    while index.value() < len(paintings):
        try:
            painting = paintings[index.value()]
            response = requests.get(painting['imageURL'])
            img = Image.open(BytesIO(response.content))
            statRGB = ImageStat.Stat(img)
            r,g,b = statRGB.rms
            perceivedBrightness = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            paintingData = {
                'earliestDate': painting['earliestDate'],
                'latestDate': painting['latestDate'],
                'perceivedBrightness': perceivedBrightness,
                'saturation': saturation
            }
            allPaintingData.append(paintingData)
        except:
            print(f'Failed to analyze painting #{index}')
        print(f'Analyzed {index.value()} / {paintings.__len__()} paintings')
        index.increment()
    
# 20 threads to analayze paintings
if __name__ == "__main__":
    for i in range(1, 100):
        thread = threading.Thread(target=thread_function, args=(index,))
        thread.start()
    thread.join()
    

with open('paintingData.json', 'w') as outfile:
    json.dump(allPaintingData, outfile, indent=4)