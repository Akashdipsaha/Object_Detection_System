#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system('pip install opencv-python')


# In[22]:


import cv2


# In[23]:


import matplotlib.pyplot as plt


# In[24]:


config_file = "C:/Users/KIIT/Music/objdetec/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "C:/Users/KIIT/Music/objdetec/frozen_inference_graph.pb"


# In[25]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[26]:


classLabels = []
file_name = "C:/Users/KIIT/Music/objdetec/label.txt"
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[27]:


print(classLabels)


# In[28]:


print(len(classLabels))


# In[29]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[30]:


img = cv2.imread("C:/Users/KIIT/Music/objdetec/demo.jpg")


# In[31]:


plt.imshow(img)


# In[32]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[33]:


ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5)


# In[34]:


print(ClassIndex)


# In[35]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img,boxes,(255, 0, 0), 2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0), thickness=3)


# In[36]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


screen_width, screen_height = 700, 480

cap = cv2.VideoCapture(r"C:\Users\KIIT\Music\objdetec\General_public.mov")

# Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Use webcam as fallback
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


# In[ ]:





# In[ ]:




