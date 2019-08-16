#! /usr/bin/env python
# -*- coding: utf-8 -*-

import imutils
import argparse
import cv2

from vision.image_processing import ImageProcesser

if __name__ == '__main__':
    # Demo arguments
    parser = argparse.ArgumentParser(description='Python-OpenCV-(Keras) face detail demos.')
    parser.add_argument('-c', help='Path to haarcascade classifier model file.',
                        default='models/haarcascade_frontalface_default.xml')
    parser.add_argument('-x', help='Path to pretrained XCEPTION model for emotion detection',
                        default='models/_mini_XCEPTION.102-0.66.hdf5')
    parser.add_argument('-am', help='Path to age detection model',
                        default='models/age_net.caffemodel')
    parser.add_argument('-ac', help='Path to age detection model configuration',
                        default='models/age_deploy.prototxt')
    parser.add_argument('-fm', help='Path to face detection model',
                        default='models/opencv_face_detector_uint8.pb')
    parser.add_argument('-fc', help='Path to face detection model configuration',
                        default='models/opencv_face_detector.pbtxt')
    args = parser.parse_args()

    # Build image processing object
    iproc = ImageProcesser(args.c, args.x, (args.am, args.ac), (args.fm, args.fc))

    # Video streaming
    camera = cv2.VideoCapture(0)

    # Loop and process frames on real time
    while True:
        _, frame = camera.read()

        # Estimate emotion probabilities
        e_detection, e_probs = iproc.detect_emotions(frame)

        # Estimage age
        a_detection, a_probs = iproc.detect_age(frame)

        # Show and break when required by the user
        cv2.imshow('your emotion', e_detection)
        cv2.imshow("emotion probabilities", e_probs)
        cv2.imshow('your age', a_detection)
        cv2.imshow("age probabilities", a_probs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy cv2 camera object and free memory
    camera.release()
    cv2.destroyAllWindows()
