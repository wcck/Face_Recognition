import cv2
import imutils
import argparse

# This xml is face model
face_model = '/home/ubuntu/opencv_contrib-3.1.0/modules/face/data/cascades/haarcascade_frontalface_default.xml'
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", help="path to the image")
args = ap.parse_args()
#loading 分類器
face_cascade = cv2.CascadeClassifier(face_model)
#reading image
img = cv2.imread(args.img)
# create a grayscale version to enhance detection
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#核心中的核心(偵測人臉個數)
faces = face_cascade.detectMultiScale(img_gray, 1.1, minNeighbors=5)

#标记--在脸部画圆
index = 0
FACE_PAD = 50
#Define seven color to tag face
color = [(255, 0, 0), (255, 97, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (160, 32, 240)]
#draw rectangle
for (x, y, width, height) in faces:
	# 画线标记
	index += 1
	upper_cut = [min(img.shape[0], y + height + FACE_PAD), min(img.shape[1], x + width + FACE_PAD)]
	lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
	cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), color[index%7], 2)

# #draw circle
# for (x, y, width, height) in faces:
# 	index += 1 
# 	center_x = int((x + width * 0.5))
# 	center_y = int((y + height * 0.5))
# 	radius = int((width + height) * 0.25)
# 	cv2.circle(img, (center_x, center_y), radius, color[index%7], 2)
# 显示图片
img = imutils.resize(img, width=600)
cv2.imshow('output.jpg', img)
cv2.waitKey(0)


