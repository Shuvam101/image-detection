import cv2
import matplotlib.pyplot as plt
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_file='frozen_inference_graph.pb'
model=cv2.dnn_DetectionModel(frozen_file,config_file)
classlabels=[] #empty list of python
file_name='LABELS.txt'
with open(file_name,'rt') as fpt:
    classlabels=fpt.read().rstrip('\n').split('\n')
    #classlabels.append(fpt.read())
print(classlabels)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
img=cv2.imread('cake.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
classindex,confidence,bbox=model.detect(img,confThreshold=0.55)
print(classindex)
font_scale=3
font=cv2.FONT_HERSHEY_TRIPLEX
for classind,conf,boxes in zip(classindex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,255,0),2)
    cv2.putText(img,classlabels[classind-1],(boxes[0]+10,boxes[1]+50),font,fontScale=font_scale,color=(0,0,255),thickness=2)
plt.figure(figsize=(12,10))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# vid=cv2.VideoCapture('video.mp4')
# # vid.set(3,640)
# # vid.set(4,480)

# # check if video is opened correctly
# if not vid.isOpened():
#     vid=cv2.VideoCapture(0)
# if not vid.isOpened():
#     raise IOError("Cannot open video")
    
# font_scale=3
# font=cv2.FONT_HERSHEY_TRIPLEX

# while True:
#     ref,frame=vid.read()
#     classindex,confidence,bbox=model.detect(frame,confThreshold=0.55)
#     print(classindex)
#     if (len(classindex)!=0):
#         for classind,conf,boxes in zip(classindex.flatten(),confidence.flatten(),bbox):
#             if(classind<=80):
#                 cv2.rectangle(frame,boxes,(0,255,0),2)
#                 cv2.putText(frame,classlabels[classind-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2) 
#     cv2.imshow('Image Detection',frame)    
#     if cv2.waitKey(2) & 0xFF==ord('d'):
#         break
# vid.release()
# vid.destroyAllWindows()
