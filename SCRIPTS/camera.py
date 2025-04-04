import cv2
import numpy as np
import os
from detection import AccidentDetectionModel
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_video(video_path, num_frames=32, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = img_to_array(frame) / 255.0
            frames.append(frame)
    
    cap.release()
    return np.expand_dims(np.array(frames), axis=0)

def startapplication():
    model = AccidentDetectionModel("accident_model.h5")  # Just pass the .h5 file
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    video_path = 'car5.gif'  # Change to your video path or 0 for webcam
    cap = cv2.VideoCapture(video_path)
    
    # For real-time processing, we'll analyze clips of 32 frames
    frame_buffer = []
    clip_size = 32
    frame_skip = 2  # Process every 2nd frame to reduce computation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        normalized_frame = img_to_array(resized_frame) / 255.0
        
        # Add to buffer
        frame_buffer.append(normalized_frame)
        if len(frame_buffer) > clip_size:
            frame_buffer.pop(0)
            
        # When we have enough frames, make prediction
        if len(frame_buffer) == clip_size and len(frame_buffer) % frame_skip == 0:
            clip = np.expand_dims(np.array(frame_buffer), axis=0)
            pred_prob = model.predict_accident(clip)[1]
            
            if pred_prob > 0.7:  # High probability threshold for accident
                cv2.putText(frame, f"Accident Detected: {pred_prob[0][0]:.2f}", 
                          (20, 50), font, 1, (0, 0, 255), 2)
                # Optional: Add alert sound
                # os.system("say beep")
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()