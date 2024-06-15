import cv2
import numpy as np
from keras.models import load_model
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load the trained model
model = load_model("Suspicious_Human_Activity_Detection_LRCN_Model.h5")

# Define the classes for activities (modify as per your model)
activity_classes = ["handwaving","handclapping","jogging","running","walking","boxing"]

# Parameters for preprocessing
FRAME_COUNT = 30  # Number of frames per sequence
FRAME_HEIGHT = 64  # Height of each frame
FRAME_WIDTH = 64  # Width of each frame

# Email configuration
sender_email = "majorproject2324@gmail.com"
sender_password = "Major@project"
#receiver_email = "philipsdsouza12345@gmail.com"
receiver_email = "philipsdsouza12345@gmail.com"

def preprocess_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    preprocessed_frame = resized_frame / 255.0  # Normalize pixel values
    return preprocessed_frame

def detect_activity(frames):
    # Preprocess each frame in the sequence
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    
    # Stack frames into a 4D tensor
    input_sequence = np.stack(preprocessed_frames, axis=0)
    
    # Make predictions using the model
    predictions = model.predict(np.expand_dims(input_sequence, axis=0))
    
    # Get the predicted activity class
    predicted_class_index = np.argmax(predictions)
    predicted_activity = activity_classes[predicted_class_index]
    
    return predicted_activity

def send_email(subject, body):
    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)

    # Send email
    server.send_message(message)
    server.quit()

def capture_and_detect_activity():
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    frame_buffer = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to capture frame.")
            break
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Add frame to buffer
        frame_buffer.append(frame)
        
        # Maintain frame buffer size
        if len(frame_buffer) > FRAME_COUNT:
            frame_buffer.pop(0)
        
        # Detect human activity when enough frames are collected
        if len(frame_buffer) == FRAME_COUNT:
            activity = detect_activity(frame_buffer)
            
            # Display the frame with the predicted activity
            cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Suspicious Human Activity Detection', frame)
            
            # If boxing activity detected, send email
            if activity == "handwaving":
                send_email("Suspicious Activity Detected", "handwaving activity detected in the video.")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing and detecting activity
capture_and_detect_activity()



















# import cv2
# import numpy as np
# from keras.models import load_model

# # Load the trained model
# model = load_model("Suspicious_Human_Activity_Detection_LRCN_Model.h5")

# # Define the classes for activities (modify as per your model)
# activity_classes = ["handwaving","handclapping","jogging","running","walking","boxing"]

# # Parameters for preprocessing
# FRAME_COUNT = 30  # Number of frames per sequence
# FRAME_HEIGHT = 64  # Height of each frame
# FRAME_WIDTH = 64  # Width of each frame

# def preprocess_frame(frame):
#     # Preprocess the frame (resize, normalize, etc.)
#     resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
#     preprocessed_frame = resized_frame / 255.0  # Normalize pixel values
#     return preprocessed_frame

# def detect_activity(frames):
#     # Preprocess each frame in the sequence
#     preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    
#     # Stack frames into a 4D tensor
#     input_sequence = np.stack(preprocessed_frames, axis=0)
    
#     # Make predictions using the model
#     predictions = model.predict(np.expand_dims(input_sequence, axis=0))
    
#     # Get the predicted activity class
#     predicted_class_index = np.argmax(predictions)
#     predicted_activity = activity_classes[predicted_class_index]
    
#     return predicted_activity

# def capture_and_detect_activity():
#     # Open the default camera (usually the webcam)
#     cap = cv2.VideoCapture(0)
    
#     # Check if the camera opened successfully
#     if not cap.isOpened():
#         print("Error: Unable to open camera.")
#         return
    
#     frame_buffer = []
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
        
#         if not ret:
#             print("Error: Unable to capture frame.")
#             break
#         # Flip the frame horizontally
#         frame = cv2.flip(frame, 1)

#         # Add frame to buffer
#         frame_buffer.append(frame)
        
#         # Maintain frame buffer size
#         if len(frame_buffer) > FRAME_COUNT:
#             frame_buffer.pop(0)
        
#         # Detect human activity when enough frames are collected
#         if len(frame_buffer) == FRAME_COUNT:
#             activity = detect_activity(frame_buffer)
            
#             # Display the frame with the predicted activity
#             cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.imshow('Suspicious Human Activity Detection', frame)
        
#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the capture
#     cap.release()
#     cv2.destroyAllWindows()

# # Call the function to start capturing and detecting activity
# capture_and_detect_activity()

