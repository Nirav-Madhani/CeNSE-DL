import os
import sys
import threading

def commandThread(command):
    os.system(command)

totalImages, numWorkers = 0,0
imgs = []
totalImages = int(sys.argv[1])
numWorkers = int(sys.argv[2])

if totalImages % numWorkers != 0:
    numImagePerWoker = totalImages // numWorkers
    imgs = [numImagePerWoker] * (numWorkers-1) + [totalImages - numImagePerWoker * (numWorkers-1)]
else:
    imgs = [totalImages // numWorkers] * numWorkers

offset = [0]
for i in range(numWorkers-1):
    offset.append(offset[i] + imgs[i])
print(imgs,offset)
os.mkdir("Dataset")
for i in range(numWorkers):
    print(f"DataGenerator.exe {imgs[i]} {offset[i]}")
    T = threading.Thread(target=commandThread, args=(f"DataGenerator.exe {imgs[i]} {offset[i]}",))
    T.start()