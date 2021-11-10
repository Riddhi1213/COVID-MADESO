import time
import cv2
import numpy as np

confid = 0.5
thresh = 0.5


def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0


labelsPath = "./dist_files/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

# weightsPath = "./yolov3.weights"
# configPath = "./yolov3.cfg"

###### use this for faster processing (caution: slighly lower accuracy) ###########

weightsPath = "./dist_files/yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
configPath = "./dist_files/yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg



class DistVideo(object):
    def __init__(self,filename):
        self.fname=filename
        self.vid_path = "./dist_files/uploads/"+self.fname
        print("hello in distancing  ",self.vid_path)
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.video = cv2.VideoCapture(self.vid_path)
        self.writer = None

        (self.W, self.H) = (None, None)

        self.fl = 0
        self.q = 0
        

    
    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        # success, image = self.video.read()
        # image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        # 	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # 	break
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        (grabbed, frame) = self.video.read()

        if not grabbed:
            print('No vid')


        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
            self.q = self.W

        #frame = frame[0:self.H, 200:self.q]
        frame = frame[0:self.H, 0:self.q]
        (self.H, self.W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if LABELS[classID] == "person":

                    if confidence > confid:
                        box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

        if len(idxs) > 0:

            status = list()
            idf = idxs.flatten()
            close_pair = list()
            s_close_pair = list()
            center = list()
            dist = list()
            for i in idf:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                center.append([int(x + w / 2), int(y + h / 2)])

                status.append(0)
            for i in range(len(center)):
                for j in range(len(center)):
                    g = isclose(center[i], center[j])

                    if g == 1:

                        close_pair.append([center[i], center[j]])
                        status[i] = 1
                        status[j] = 1
                    elif g == 2:
                        s_close_pair.append([center[i], center[j]])
                        if status[i] != 1:
                            status[i] = 2
                        if status[j] != 1:
                            status[j] = 2

            total_p = len(center)
            low_risk_p = status.count(2)
            high_risk_p = status.count(1)
            safe_p = status.count(0)
            kk = 0

            for i in idf:

                sub_img = frame[10:170, 10:self.W - 10]
                # black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

                #res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 1.0)

                #frame[10:170, 10:self.W - 10] = res

                cv2.putText(frame, "Social Distancing Analyser", (100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # cv2.rectangle(frame, (20, 60), (510, 160), (170, 170, 170), 2)
                # cv2.putText(frame, "Connecting lines shows closeness among people. ", (30, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                # cv2.putText(frame, "-- YELLOW: CLOSE", (50, 110),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # cv2.putText(frame, "--    RED: VERY CLOSE", (50, 130),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # cv2.rectangle(frame, (535, 60), (self.W - 20, 160), (170, 170, 170), 2)
                # cv2.putText(frame, "Bounding box shows the level of risk to the person.", (545, 80),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                # cv2.putText(frame, "-- DARK RED: HIGH RISK", (565, 110),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
                # cv2.putText(frame, "--   ORANGE: LOW RISK", (565, 130),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

                # cv2.putText(frame, "--    GREEN: SAFE", (565, 150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                tot_str = "TOTAL: " + str(total_p)
                high_str = "RISK: " + str(high_risk_p+low_risk_p)
                #low_str = "LOW RISK COUNT: " + str(low_risk_p)
                safe_str = "SAFE: " + str(safe_p)

                sub_img = frame[self.H - 100:self.H, 0:100]
                black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

                res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)

                frame[self.H - 100:self.H, 0:100] = res

                cv2.putText(frame, tot_str, (10, self.H - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(frame, safe_str, (10, self.H - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
               # cv2.putText(frame, low_str, (10, self.H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 255), 1)
                cv2.putText(frame, high_str, (10, self.H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if status[kk] == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

                elif status[kk] == 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

                kk += 1
            for h in close_pair:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
            for b in s_close_pair:
                cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)

            #cv2.imshow('Social distancing analyser', frame)
            #cv2.waitKey(1)

        # if self.writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     self.writer = cv2.VideoWriter("output.mp4", fourcc, 30,
        #                              (frame.shape[1], frame.shape[0]), True)

        # self.writer.write(frame)
        print(frame.shape)
        rett, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
            
