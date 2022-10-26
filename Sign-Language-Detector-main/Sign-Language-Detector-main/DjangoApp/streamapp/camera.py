import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
import cv2
import urllib.request
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
				
		self.BASE = os.path.dirname(os.path.abspath(__file__))
		self.model_path = os.path.join(self.BASE, "night_actions.h5")

		self.model = load_model(self.model_path)
		self.mp_holistic = mp.solutions.holistic # Holistic model
		self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities

		# Actions that we try to detect
		self.actions = np.array(['hello', 'thanks', 'iloveyou', 'like'])
		# Thirty videos worth of data
		self.no_sequences = 30
		# Videos are going to be 30 frames in length
		self.sequence_length = 10

		self.sequence = []
		self.sentence = []
		self.threshold = 0.8

	def __del__(self):
		self.video.release()
	
	def mediapipe_detection(self, image, model):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
		image.flags.writeable = False                  # Image is no longer writeable
		results = model.process(image)                 # Make prediction
		image.flags.writeable = True                   # Image is now writeable 
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
		return image, results

	def draw_landmarks(self, image, results):
		self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
		self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
		self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
		self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
	
	def draw_styled_landmarks(self, image, results):
		# Draw face connections
		self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, 
								self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
								self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
								) 
		# Draw pose connections
		self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
								self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
								self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
								) 
		# Draw left hand connections
		self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
								self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
								self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
								) 
		# Draw right hand connections  
		self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
								self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
								self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
								) 
		
	def extract_keypoints(self, results):
		pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
		face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
		lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
		rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
		return np.concatenate([pose, face, lh, rh])

	colors = [(245,117,16), (117,245,16), (16,117,245), (50,117,245)]
	def prob_viz(self, res, actions, input_frame, colors):
		output_frame = input_frame.copy()
		for num, prob in enumerate(res):
			cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
			cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
			
		return output_frame

	def get_frame(self):
		success, frame = self.video.read()

		#frame = cv2.flip(img, 1)

		# Do operation......

		# Set mediapipe model 
		with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
			# Make detections
			image, results = self.mediapipe_detection(frame, holistic)
			print(results)
			
			# Draw landmarks
			self.draw_styled_landmarks(image, results)
			
			# 2. Prediction logic
			keypoints = self.extract_keypoints(results)
			self.sequence.append(keypoints)
			self.sequence = self.sequence[-10:]
			
			if len(self.sequence) == 10:
				res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
				print(self.actions[np.argmax(res)])
				
				
			#3. Viz logic
				if res[np.argmax(res)] > self.threshold: 
					if len(self.sentence) > 0: 
						if self.actions[np.argmax(res)] != self.sentence[-1]:
							self.sentence.append(self.actions[np.argmax(res)])
					else:
						self.sentence.append(self.actions[np.argmax(res)])

				if len(self.sentence) > 5: 
					self.sentence = self.sentence[-5:]

				# Viz probabilities
				image = self.prob_viz(res, self.actions, image, self.colors)
				
			cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
			cv2.putText(image, ' '.join(self.sentence), (3,30), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

		# End Operation.....

		ret, jpeg = cv2.imencode('.jpg', image) 
		return jpeg.tobytes()


class IPWebCam(object):
	def __init__(self):
		self.url = "http://172.20.10.2:8080/shot.jpg"

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
		img = cv2.imdecode(imgNp, -1)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_webcam.detectMultiScale(
			gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h),
			              color=(255, 0, 0), thickness=2)
		resize = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
		frame_flip = cv2.flip(resize, 1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()
