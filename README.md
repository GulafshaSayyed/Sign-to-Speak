# Sign-to-Speak
A Vision-Based Real-Time Sign Language to Text and Speech Translation System

The Sign-to-Speak project is a real-time vision-based assistive communication system designed to translate sign language gestures into readable text and audible speech. The system aims to reduce the communication gap between deaf or hard-of-hearing individuals and non-signers by leveraging computer vision and deep learning techniques. It operates without the need for wearable sensors, relying only on a standard camera for gesture capture.

### Objectives
To recognize hand gestures corresponding to sign language in real time
To convert recognized gestures into meaningful text
To generate speech output from the recognized text
To create an accessible and user-friendly assistive communication tool

### Problem Statement
Sign language is not widely understood by the general population
Communication barriers exist in social, educational, and professional environments
Existing solutions often require expensive hardware or lack real-time performance

### System Architecture
The system captures live video input through a camera and processes each frame to detect hand movements. Mediapipe is used to extract hand landmarks, enabling a skeleton-based representation of gestures. These landmarks are normalized and passed to a Convolutional Neural Network, which classifies the gesture into the corresponding sign language character.

### Key Components
Camera-based real-time gesture capture
Mediapipe for hand landmark extraction
CNN for gesture classification
Rule-based disambiguation for similar signs
Text generation module
Text-to-speech conversion module

### Dataset and Training
Dataset includes 26 American Sign Language (ASL) fingerspelling gestures
Images are captured under varied lighting and background conditions
Data preprocessing includes landmark normalization and noise reduction
CNN is trained using labeled gesture data

### Performance Evaluation
Achieves up to 99% accuracy in controlled environments
Maintains around 97% accuracy in real-world conditions
Shows minimal overfitting due to robust preprocessing techniques

### Technologies Used
Python
OpenCV
Mediapipe
TensorFlow and Keras
NumPy
Text-to-Speech (pyttxs) library
Pychant
Tkinter

### Applications
Assistive communication for deaf and hard-of-hearing individuals
Educational tools for learning sign language
Public service and customer support systems
Healthcare and emergency communication

### Future Enhancements
Support for dynamic and continuous gestures
Sentence-level recognition and grammar correction
Integration of facial expressions for emotion recognition

Deployment as a web or mobile application

Expansion to multiple sign languages
