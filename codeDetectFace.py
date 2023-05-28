from mtcnn import MTCNN
import cv2

# Create MTCNN detector
detector = MTCNN()

#Task1: to detect single face
def detectSingleFace():
	# Load image from file
	img = cv2.imread('Image1.jpeg')

	# Resize image to 400x400 pixels
	resized_img = cv2.resize(img, (400, 400))

	# Detect faces
	output = detector.detect_faces(resized_img)

	# Get coordinates of face bounding box
	x, y, w, h = output[0]['box']

	# Draw bounding box around face
	cv2.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display resized image in a window named "Resized Image"
	cv2.imshow('Resized Image', resized_img)

	# Wait for user to press a key
	cv2.waitKey(0)

	# Close window and exit program
	cv2.destroyAllWindows()



#Task2: Detect Facial Land Marks (Single Face)
def singleFaceLandMark():
	img = cv2.imread('Image1.jpeg')

	# Resize image to 400x400 pixels
	resized_img = cv2.resize(img, (640, 640))

	# Detect faces
	output = detector.detect_faces(resized_img)
	print(output)
	# Get coordinates of face bounding box
	x, y, w, h = output[0]['box']

	# Draw bounding box around face
	cv2.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	lefteye_x, lefteye_y = output[0]['keypoints']['left_eye']
	righteye_x, righteye_y = output[0]['keypoints']['right_eye']
	nose_x, nose_y = output[0]['keypoints']['nose']
	mouthleft_x, mouthleft_y = output[0]['keypoints']['mouth_left']
	mouthright_x, mouthright_y = output[0]['keypoints']['mouth_right']
	cv2.circle(resized_img, center = (nose_x,nose_y), color = (0,0,255), thickness= 8, radius = 1)
	cv2.circle(resized_img, center = (lefteye_x,lefteye_y), color = (0,0,255), thickness= 8, radius = 1)
	cv2.circle(resized_img, center = (righteye_x,righteye_y), color = (0,0,255), thickness= 8, radius = 1)
	cv2.circle(resized_img, center = (mouthleft_x,mouthleft_y), color = (0,0,255), thickness= 8, radius = 1)
	cv2.circle(resized_img, center = (mouthright_x,mouthright_y), color = (0,0,255), thickness= 8, radius = 1)

	cv2.imshow("image",resized_img)
	cv2.waitKey(0)
	# Close window and exit program
	cv2.destroyAllWindows()



#Task3: to detect Multiple Faces
def detectMultipleFace():
	img = cv2.imread('Image2.jpeg')

	# Get original image size
	height, width, _ = img.shape

	# Set desired image size
	new_size = 600

	# Calculate scaling factor
	scale = new_size / max(height, width)

	# Resize image
	resized_img = cv2.resize(img, None, fx=scale, fy=scale)

	# Detect faces
	output = detector.detect_faces(resized_img)

	# Draw bounding boxes around detected faces
	for face in output:
		# Get coordinates of face bounding box
		x, y, w, h = face['box']

		# Draw bounding box around face
		cv2.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display image with bounding boxes
	cv2.imshow('Image', resized_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



#Task4: Detect Facial Land Mark (Multiple Faces)
def multipleFaceLandMark():
	img = cv2.imread('Image2.jpeg')

	# Get original image size
	height, width, _ = img.shape

	# Set desired image size
	new_size = 600

	# Calculate scaling factor
	scale = new_size / max(height, width)

	# Resize image
	resized_img = cv2.resize(img, None, fx=scale, fy=scale)

	# Detect faces
	output = detector.detect_faces(resized_img)

	# Draw bounding boxes around detected faces
	for face in output:
		# Get coordinates of face bounding box
		x, y, w, h = face['box']

		# Draw bounding box around face
		cv2.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		lefteye_x, lefteye_y = face['keypoints']['left_eye']
		righteye_x, righteye_y = face['keypoints']['right_eye']
		nose_x, nose_y = face['keypoints']['nose']
		mouthleft_x, mouthleft_y = face['keypoints']['mouth_left']
		mouthright_x, mouthright_y = face['keypoints']['mouth_right']
		cv2.circle(resized_img, center = (nose_x,nose_y), color = (0,0,255), thickness= 8, radius = 1)
		cv2.circle(resized_img, center = (lefteye_x,lefteye_y), color = (0,0,255), thickness= 8, radius = 1)
		cv2.circle(resized_img, center = (righteye_x,righteye_y), color = (0,0,255), thickness= 8, radius = 1)
		cv2.circle(resized_img, center = (mouthleft_x,mouthleft_y), color = (0,0,255), thickness= 8, radius = 1)
		cv2.circle(resized_img, center = (mouthright_x,mouthright_y), color = (0,0,255), thickness= 8, radius = 1)

	# Display image with bounding boxes
	cv2.imshow('Image', resized_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



#Task5: Detecting Faces in a Video
def detectVideoFace():
	video = cv2.VideoCapture('vid.mp4')

	while True:
		# Read frame from video
		ret, frame = video.read()

		if ret:
			# Resize frame
			frame = cv2.resize(frame, (1280, 720))
		
			# Detect faces
			faces = detector.detect_faces(frame)

			# Draw bounding boxes around detected faces
			for face in faces:
				# Get coordinates of face bounding box
				x, y, w, h = face['box']

				# Draw bounding box around face
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			# Display frame with bounding boxes
			cv2.imshow('Video', frame)

			# Exit on ESC key
			if cv2.waitKey(1) == 27:
				break
		else:
			break

	# Release video capture and close all windows
	video.release()
	cv2.destroyAllWindows()



#Task6: Detecting Facial Land Mark (in a Video)
def videoLandMark():
	video = cv2.VideoCapture('vid.mp4')

	while True:
		# Read frame from video
		ret, frame = video.read()

		if ret:
			# Resize frame
			frame = cv2.resize(frame, (1280, 720))

			# Detect faces
			faces = detector.detect_faces(frame)

			# Draw bounding boxes around detected faces
			for face in faces:
				# Get coordinates of face bounding box
				x, y, w, h = face['box']
				# Draw bounding box around face
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				
				lefteye_x, lefteye_y = face['keypoints']['left_eye']
				righteye_x, righteye_y = face['keypoints']['right_eye']
				nose_x, nose_y = face['keypoints']['nose']
				mouthleft_x, mouthleft_y = face['keypoints']['mouth_left']
				mouthright_x, mouthright_y = face['keypoints']['mouth_right']
				cv2.circle(frame, center = (nose_x,nose_y), color = (0,0,255), thickness= 8, radius = 1)
				cv2.circle(frame, center = (lefteye_x,lefteye_y), color = (0,0,255), thickness= 8, radius = 1)
				cv2.circle(frame, center = (righteye_x,righteye_y), color = (0,0,255), thickness= 8, radius = 1)
				cv2.circle(frame, center = (mouthleft_x,mouthleft_y), color = (0,0,255), thickness= 8, radius = 1)
				cv2.circle(frame, center = (mouthright_x,mouthright_y), color = (0,0,255), thickness= 8, radius = 1)
			# Display frame with bounding boxes
			cv2.imshow('Video', frame)

			# Exit on ESC key
			if cv2.waitKey(1) == 27:
				break
		else:
			break

	# Release video capture and close all windows
	video.release()
	cv2.destroyAllWindows()




detectSingleFace()
singleFaceLandMark()
detectMultipleFace()
multipleFaceLandMark()
detectVideoFace()
videoLandMark()