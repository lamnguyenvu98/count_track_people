from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0 #counter id gán cho từng object
        self.objects = OrderedDict() # một dictionary lưu trữ các tuple (objectID, centroid) ~ (key, value)
        self.disappeared = OrderedDict() # dictionary lưu trữ số lần của object đã mất khỏi khung hình
        self.maxDisappeared = maxDisappeared # số frame cho phép đối tượng mất khỏi khung hình
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid  #lưu centroid vào dict objects
        self.disappeared[self.nextObjectID] = 0 #khởi tạo số lần object biến mất
        self.nextObjectID += 1 #tăng counter id lên 1 vì mỗi object cần 1 id
                               # #khi có object mới cùng centroid thì tạo thêm tuple

    def deregister(self, objectID):
        del self.objects[objectID] #xoa object id khoi dict objects
        del self.disappeared[objectID] # xoa object id khoi dict disappeared

    def update(self, rects):  #rects có dạng (x1,y1,x2,y2) hoặc (x, y, x+w, y+h)
        if len(rects) == 0: # check list input bounding box có hay không, tức xem có đối tượng mới hay không
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1  #lặp qua list object được tracked và đánh dấu là disappeared
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID) # mất quá số frame quy định thì xóa object id đi
            return self.objects

        inputCentroid = np.zeros((len(rects), 2), dtype=int)
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroid[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroid)):
                self.register(inputCentroid[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroid = list(self.objects.values())
            D = distance.cdist(np.array(objectCentroid), inputCentroid)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row,col) in zip(rows,cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > self.maxDistance: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroid[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroid[col])
        return self.objects