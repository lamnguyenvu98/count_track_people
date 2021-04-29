import cv2
import numpy as np
from centroidTracker import CentroidTracker
from trackableobject import TrackableObject
import pafy
from random import randint
import imutils
from imutils.video import VideoStream
from collections import deque
import dlib, time, schedule, csv
from yolo import *
import config, thread
from itertools import zip_longest
import datetime
import argparse

def video_type(type='cam'):
    if type == 'cam':
        vid = cv2.VideoCapture("highway.mp4")
    if type == 'video':
        url = 'https://www.youtube.com/watch?v=H7BrVzdOzc4'  # 'https://www.youtube.com/watch?v=wqctLW0Hb_0&t=369s' #'https://www.youtube.com/watch?v=WvhYuDvH17I'   # ""
        video = pafy.new(url)
        play = video.getbest(preftype="mp4")
        vid = cv2.VideoCapture(play.url)
    return vid

t0 = time.time()
def run():
    classes = [c.strip() for c in open('coco.names').readlines()]
    conf_threshold = 0.6  # lay confidence > 0.5
    nmsThreshold = 0.4  # > 0.5 se ap dung Non-max Surpression
    shape = 288
    colors = []
    colors.append([(randint(0, 255), randint(0, 255), randint(0, 255)) for i in range(1000)])
    detected_classes = ['cell phone']
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    pts = [deque(maxlen=10) for _ in range(1000)]
    counter = 0
    center = None
    trackers = []
    totalIn = []
    empty = []
    empty1 = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    (W, H) = (None, None)
    net = yolo_net("yolov3.weights", "yolov3.cfg")
    if config.Thread:
        vid = thread.ThreadingClass(0)
    else: vid = cv2.VideoCapture(0)
    while True:
        if config.Thread:
            img = vid.read()
        else:
            _, img = vid.read()
        img = cv2.resize(img, (600,500))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = img.shape[:2]
        status = "Waiting"
        rects = []
        if totalFrames % 30 == 0:
            status = "Detecting"
            trackers = []
            outputs = yolo_output(net, img, shape)
            bbox, classIds, confs = yolo_predict(outputs, conf_threshold, H, W)
            indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nmsThreshold)
            for i in indices:
                i = i[0]
                if classes[classIds[i]] not in detected_classes: continue
                box = bbox[i]
                color = colors[0][i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))
        cv2.line(img, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        obj = ct.update(rects)
        for (objectID, centroid) in obj.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                oy = [c[1] for c in to.centroids]
                directionY = centroid[1] - np.mean(oy)
                to.centroids.append(centroid)
                if not to.counted:
                    if directionY < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        empty.append(totalUp)
                        to.counted = True

                    elif directionY > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        empty1.append(totalDown)
                        # print(empty1[-1])
                        totalIn = []
                        # compute the sum of total people inside
                        totalIn.append(len(empty1) - len(empty))
                        print("Total people inside:", totalIn)
                        # if the people limit exceeds over threshold, send an email alert
                        if sum(totalIn) >= config.Threshold:
                            cv2.putText(img, "-ALERT: People limit exceeded-", (10, img.shape[0] - 80),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                print("[INFO] Sending email alert..")
                                # Mailer().send(config.MAIL)
                                print("[INFO] Alert sent")

                        to.counted = True

            trackableObjects[objectID] = to

            text1 = "ID {}".format(objectID)
            colorID = colors[0][objectID]
            cv2.circle(img, (centroid[0], centroid[1]), 4, colorID, -1)
            # cv2.putText(img, "Direction: {}".format(direction), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            center = (centroid[0], centroid[1])
            pts[objectID].append(center)
            for i in range(1, len(pts[objectID])):
                if pts[objectID][i - 1] is None or pts[objectID][i] is None:
                    continue
                thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
                cv2.line(img, pts[objectID][i - 1], pts[objectID][i], colorID, thickness)
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
            ]
        info2 = [
            ("Total people inside", totalIn),
        ]


        for (i, (k, v)) in enumerate(info):
            text2 = "{}: {}".format(k, v)
            cv2.putText(img, text2, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for (i, (k, v)) in enumerate(info2):
            text3 = "{}: {}".format(k, v)
            cv2.putText(img, text3, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if config.Log:
            datetimee = [datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")]
            d = [datetimee, empty1, empty, totalIn]
            print("D: ", d)
            export_data = zip_longest(*d, fillvalue='')
            print("Export Data: ", export_data)
            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)
                myfile.close()
            with open('Log.csv', 'a', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerows(export_data)

        cv2.imshow('Result', img)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        totalFrames += 1
        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds = (t1 - t0)
            if num_seconds > 28800:
                break
    if not config.Thread:
        vid.release()
    cv2.destroyAllWindows()


if config.Scheduler:
    ##Runs for every 1 second
    # schedule.every(1).seconds.do(run)
    ##Runs at every day (9:00 am). You can change it.
    schedule.every().day.at("9:00").do(run)

    while 1:
        schedule.run_pending()

else:
    run()
