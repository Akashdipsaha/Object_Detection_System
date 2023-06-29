get_ipython().system('pip install opencv-python')
import cv2
import matplotlib.pyplot as plt
config_file = "C:/Users/KIIT/Music/objdetec/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "C:/Users/KIIT/Music/objdetec/frozen_inference_graph.pb"


model = cv2.dnn_DetectionModel(frozen_model,config_file)


classLabels = []
file_name = "C:/Users/KIIT/Music/objdetec/label.txt"
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

print(len(classLabels))


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


img = cv2.imread("C:/Users/KIIT/Music/objdetec/demo.jpg")


plt.imshow(img)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5)

print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img,boxes,(255, 0, 0), 2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0), thickness=3)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

screen_width, screen_height = 700, 480

cap = cv2.VideoCapture(r"C:\Users\KIIT\Music\objdetec\General_public.mov")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(classIndex) != 0:
        for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 8, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
        
        cv2.imshow('Object Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
