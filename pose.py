import cv2
import mediapipe as mp
import numpy as np
def process_pose_image(image_path, output_path):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Failed to load image {image_path}")
                return
            
            image_height, image_width, _ = image.shape
            
            # Resize the image
            image_resized = cv2.resize(image, (int(image_width * 0.5), int(image_height * 0.5)))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Process the image to detect pose
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"Warning: No pose landmarks found for {image_path}")
                return
            
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                keypoints.append((x, y))
            
            # Create an empty image for drawing the skeleton
            skeleton_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            
            # Draw connections between keypoints
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                cv2.line(skeleton_image, start_point, end_point, (0, 255, 0), 2)
                cv2.circle(skeleton_image, start_point, 4, (0, 0, 255), -1)
                cv2.circle(skeleton_image, end_point, 4, (0, 0, 255), -1)
            
            # Save the skeleton image
            cv2.imwrite(output_path, skeleton_image)
            print(f'Saved skeleton pose at {output_path}')
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    print("Processing complete.")