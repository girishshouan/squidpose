import cv2
import mediapipe as mp
import numpy as np

# 定义关键点索引
keypoint_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
connections = [
    (11, 13), (12, 14),  # 肩膀到肘部
    (13, 15), (14, 16),  # 肘部到手腕
    (23, 25), (24, 26),  # 臀部到膝盖
    (25, 27), (26, 28),  # 膝盖到脚踝
    (11, 12),            # 左肩到右肩
    (23, 24)             # 左臀到右臀
]

#画线
def draw_custom_landmarks(image, landmarks, indices, connections, distances, max_distance=0.4):
    """ Draw keypoints and lines on the image with colors based on individual distances. """
    if landmarks and distances:  # Ensure there are landmarks and distances are not empty
        # Draw each connection with a color based on its individual distance
        for i, (idx1, idx2) in enumerate(connections):
            point1 = landmarks.landmark[idx1]
            point2 = landmarks.landmark[idx2]
            # Use red if the distance exceeds the threshold, green otherwise
            color = (0, 0, 255) if distances[i] > max_distance else (0, 255, 0)
            cv2.line(image, 
                     (int(point1.x * image.shape[1]), int(point1.y * image.shape[0])),
                     (int(point2.x * image.shape[1]), int(point2.y * image.shape[0])),
                     color, 2)

        # Draw each keypoint. A keypoint is colored red if any of its connections exceed the threshold
        for idx in indices:
            point = landmarks.landmark[idx]
            involved_in_bad_connection = any(distances[j] > max_distance for j, (id1, id2) in enumerate(connections) if id1 == idx or id2 == idx)
            color = (0, 0, 255) if involved_in_bad_connection else (0, 255, 0)
            cv2.circle(image,
                       (int(point.x * image.shape[1]), int(point.y * image.shape[0])),
                       5, color, -1)

# Function to extract keypoints
def extract_keypoints(results, indices):
    if results.pose_landmarks:
        return [(results.pose_landmarks.landmark[idx].x,
                 results.pose_landmarks.landmark[idx].y,
                 results.pose_landmarks.landmark[idx].z) for idx in indices]
    return []

# Define a function to calculate the distance between two points
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1[0] - landmark2[0]) ** 2 + 
                   (landmark1[1] - landmark2[1]) ** 2 + 
                   (landmark1[2] - landmark2[2]) ** 2)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize camera and video file
cap = cv2.VideoCapture(0)  # Initialize the webcam

video = cv2.VideoCapture('yoga.mp4')  # Replace with your video file path
threshold = 0.18

# video = cv2.VideoCapture('pushups.mp4')  # Replace with your video file path
# threshold = 0.27

# video = cv2.VideoCapture('pilates.mp4')  # Replace with your video file path
# threshold = 0.34

frame_counter = 0  # Frame counter for keeping track of the number of processed frames
pose_match = True  # Assume initial pose match to start video playback

# Main loop to process frames from camera and video
while cap.isOpened() and video.isOpened():
    success_cam, frame_cam = cap.read()  # Read a frame from the camera
    if not success_cam:
        break  # Stop if the camera fails

    if pose_match:  # Update video frame only when poses match
        success_vid, frame_vid = video.read()  # Read a frame from the video
        if not success_vid:
            break

    # Resize frames to match dimensions
    height, width = 480, 720
    frame_cam = cv2.resize(frame_cam, (width, height))  # Resize camera frame
    frame_vid = cv2.resize(frame_vid, (width, height))  # Resize video frame

    # Process camera video frame
    frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)  # Convert camera frame to RGB
    frame_cam_rgb.flags.writeable = False  # Disable writing to frame to improve performance
    results_cam = pose.process(frame_cam_rgb)  # Process the frame through the pose estimation model

    # Process recorded video frame
    frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)  # Convert video frame to RGB
    frame_vid_rgb.flags.writeable = False  # Disable writing to frame to improve performance
    results_vid = pose.process(frame_vid_rgb)  # Process the frame through the pose estimation model
    
    '''
    # Draw keypoints and connection lines on both frames
    draw_custom_landmarks(frame_cam, results_cam.pose_landmarks, keypoint_indices, connections)
    draw_custom_landmarks(frame_vid, results_vid.pose_landmarks, keypoint_indices, connections)
    '''
    # Calculate the distances between corresponding keypoints in the two frames
    current_landmarks = extract_keypoints(results_cam, keypoint_indices)
    video_landmarks = extract_keypoints(results_vid, keypoint_indices)
    distances = [calculate_distance(lm1, lm2) for lm1, lm2 in zip(current_landmarks, video_landmarks)]

    # Check if distances list is not empty
    if distances:
        # Check if all distances are below a threshold to consider the poses matched
        # threshold = 0.4  # Set your desired threshold here
        pose_match = all(distance < threshold for distance in distances)
        if pose_match:
            print("Pose match successful")
        else:
            print("Pose not matched, pausing video playback")
    else:
        print("No keypoints detected, checking next frames")
        pose_match = False  # Continue to the next frame or handle accordingly

    # Draw keypoints and connection lines on the camera frame
    draw_custom_landmarks(frame_cam, results_cam.pose_landmarks, keypoint_indices, connections, distances)

    # If pose_match is True, also draw on the video frame
    if pose_match:
        draw_custom_landmarks(frame_vid, results_vid.pose_landmarks, keypoint_indices, connections, distances)

    combined_frame = cv2.hconcat([frame_cam, frame_vid])
    cv2.imshow('Combined Feed', combined_frame)

    # Break the loop if ESC key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    frame_counter += 1  # Increment frame counter

# Release the camera and video file handles and close all windows
cap.release()
video.release()
cv2.destroyAllWindows()