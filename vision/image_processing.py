#! /usr/bin/env python
# -*- coding: utf-8 -*-

import imutils
import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Categories
EMOTIONS = ['angry' ,'disgust','scared',
            'happy', 'sad',
            'surprised', 'neutral']
AGELIST = ['(0-2)', '(4-6)', '(8-12)',
           '(15-20)', '(25-32)', '(38-43)',
           '(48-53)', '(60-100)']

class ImageProcesser:

    def __init__(self, haarcascade_path, xception_path, age_paths, face_paths):
        self.haarcascade_path = haarcascade_path
        self.xception_path = xception_path
        self.age_paths = age_paths
        self.face_paths = face_paths
        self.load_models()

    def load_models(self):
        """Load all required models.
        """
        self.face_detection = cv2.CascadeClassifier(self.haarcascade_path)
        self.emotion_classifier = load_model(self.xception_path, compile=False)
        self.age_net = cv2.dnn.readNet(*self.age_paths)
        self.face_net = cv2.dnn.readNet(*self.face_paths)

    def preprocess_image(self, image, resize_dims):
        image_res = imutils.resize(image, width=resize_dims)
        return image_res, cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)

    def detect_faces(self, image):
        faces = self.face_detection.detectMultiScale(image, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            return sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

    def detect_faces2(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

    def detect_age(self, image):
        image_clone = image.copy()
        canvas = np.zeros((250, 300, 3), dtype=np.uint8)

        # Detect face
        padding = 20
        frame_face, bboxes = self.detect_faces2(image)
        for bbox in bboxes:
            face = image_clone[max(0, bbox[1]-padding):min(bbox[3]+padding, image.shape[0]-1),
                               max(0, bbox[0]-padding):min(bbox[2]+padding, image.shape[1]-1)]

            # Make blob
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)

            # Predict age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = AGELIST[age_preds[0].argmax()]
            print("Age Output : {}".format(age_preds))
            print("Age : {}, conf = {:.3f}".format(age, age_preds[0].max()))
            cv2.putText(frame_face, 'AGE: {}'.format(age), (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            for (i, (emotion, prob)) in enumerate(zip(AGELIST, age_preds.squeeze())):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        return frame_face, canvas

    def detect_emotions(self, img):
        # Preprocess image
        image, gray = self.preprocess_image(img, 300)

        # Build probability canvas
        canvas = np.zeros((250, 300, 3), dtype=np.uint8)
        image_clone = image.copy()

        # Detect faces
        face = self.detect_faces(image)

        preds = None
        if face is not None:
            fX, fY, fW, fH = face
            # Extract the ROI of the face from the grayscale image,
            # resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Compute predictions and label
            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

        if preds is not None:
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(image_clone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(image_clone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
        return image_clone, canvas
