# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
prev_frame_time = 0
new_frame_time = 0

while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	global faceROI
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time
	fps = int(fps)
	fps = str(fps)
	font = cv2.FONT_HERSHEY_SIMPLEX = 1
	abc = "FPS :"
	hehe = "coordinate: "
	copy = frame.copy()

	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
 
	# loop over the bounding boxes
	print(rects)
	 	
	for (x, y, w, h) in rects:
		center = (x + w//2, y + h//2)
		cv2.putText(frame, abc, (5,20), font, 1, (139, 0, 0), 1, cv2.LINE_AA)
		cv2.putText(frame , fps, (50, 20), font, 1, (139, 0, 0), 1, cv2.LINE_AA)
		cv2.putText(frame, hehe , (6, 50), font, 1,(139, 0, 0),1,cv2.LINE_AA)
		cv2.putText(frame, str(center), (105, 50), font, 1,(139, 0, 0),1,cv2.LINE_AA)
		# draw the face bounding box on the image
		face = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		face[y:y+h, x:x+w] = cv2.medianBlur(face[y:y+h, x:x+w], 35)
		cv2.imshow('frame', face)
  
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
