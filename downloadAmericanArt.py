import requests, json, os, math
from PIL import Image

imageData = []
'''
    Image Data format:
    Title, URL to image, URL to details, objectType, Painter, earliestDate, latestDate
'''

filterString = 'images object type paintings online media data source smithsonian museum american art topic landscapes'

response = requests.get("https://api.si.edu/openaccess/api/v1.0/search",
            {
                'q': filterString, 
                'api_key': os.getenv('SMITHSONIAN_API_KEY'),
                'rows': 500
                })
numRequests = 1
print(f"Made {numRequests} requests")

totalRows = response.json()['response']['rowCount']
start = 0

while start <= totalRows:
    resDict = response.json()
    for row in resDict['response']['rows']:
        try:
            imageDetails={}
            if('Paintings' not in row['content']['indexedStructured']['object_type']):
                continue
            imageDetails['objectType'] = row['content']['indexedStructured']['object_type']
            imageDetails['title'] = row['title']
            imageDetails['imageURL'] = row['content']['descriptiveNonRepeating']['online_media']['media'][0]['content']
            imageDetails['recordLink'] = row['content']['descriptiveNonRepeating']['record_link']
            if(row['content']['indexedStructured']['name'].__len__() > 1):
                continue
            imageDetails['painter'] = row['content']['indexedStructured']['name'][0]
            earliestDate = math.inf
            latestDate = -math.inf
            for dateEntry in row['content']['indexedStructured']['date']:
                if dateEntry.index('s') > -1:
                    rawDate = int(dateEntry[:dateEntry.__len__()-1])
                    eDate = rawDate
                    lDate = rawDate + 10
                    latestDate = max(lDate, latestDate)
                    earliestDate = min(eDate, earliestDate)
                else:
                    rawDate = int(dateEntry)
                    latestDate = max(rawDate, latestDate)
                    earliestDate = min(rawDate, earliestDate)
            imageDetails['earliestDate'] = earliestDate
            imageDetails['latestDate'] = latestDate
            imageData.append(imageDetails)
        except:
            pass
    start += 500
    response = requests.get("https://api.si.edu/openaccess/api/v1.0/search",
                {
                    'q': filterString,
                    'api_key': os.getenv('SMITHSONIAN_API_KEY'),
                    'rows': 500,
                    'start': start
                    })
    numRequests += 1
    print(f"Made {numRequests} requests / {round(totalRows / 500)}")

with open('paintings.json', 'w') as outfile:
    json.dump(imageData, outfile, indent=4)
