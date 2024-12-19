from flask import Flask, render_template, redirect, url_for, request, session, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import current_user, login_required, LoginManager, login_user, logout_user
from db import db  # Import db from db.py
from flask_migrate import Migrate
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pose import process_pose_image
from gen import generate_image
from reid import extract_features,fuse_features,compute_euclidean_distance,rank_gallery,compare_with_gallery,is_same_person, fuse_features2
from reidupdated import build_combined_model_with_gen_and_attention, preprocess_image,preprocess_imagewithimg
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from PIL import Image
import torch
from torchvision import transforms
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

migrate = Migrate(app, db)  # Add this line to initialize Flask-Migrate

# Bind the app to the db
db.init_app(app)

# Importing models
from models import User

# Initialize Flask-Login's LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Set the view for login

# Define user loader callback
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Flask-Login expects user_id to be an integer


# Helper function to get the logged-in user
def get_logged_in_user():
    if 'user_id' in session:
        return User.query.filter_by(id=session['user_id']).first()
    return None

# Homepage route
@app.route('/')
def index():
    user = get_logged_in_user()  # Get the logged-in user
    return render_template('index.html', user=user)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            if user.is_admin:
                return redirect(url_for('admin'))
            return redirect(url_for('index'))
        flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

# Admin route (admin only)
@app.route('/admin')
def admin():
    user = get_logged_in_user()  # Get the logged-in user
    if user and user.is_admin:
        users = User.query.all()
        return render_template('admin.html', users=users, user=user)
    return redirect(url_for('index'))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('admin'))

# Profile route
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        if request.method == 'POST':
            # Get form data from the request
            full_name = request.form.get('full_name')
            security_id = request.form.get('security_id')
            gender = request.form.get('gender')
            age = request.form.get('age')
            contact = request.form.get('contact')
            email = request.form.get('email')
            username = request.form.get('username')
            password = request.form.get('password')

            # Update user details
            user.full_name = full_name
            user.security_id = security_id
            user.gender = gender
            user.age = age
            user.contact = contact
            user.email = email
            user.username = username

            # Only update the password if a new one is provided
            if password:
                user.password = generate_password_hash(password)

            # Commit the changes to the database
            try:
                db.session.commit()
                flash('Profile updated successfully!', 'success')
            except Exception as e:
                db.session.rollback()
                flash('Error updating profile: ' + str(e), 'danger')

        return render_template('profile.html', user=user)
    
    return redirect(url_for('login'))

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

# Register new user (admin only)
@app.route('/register_user', methods=['POST'])
def register_user():
    user = get_logged_in_user()  # Get the logged-in user
    if user and user.is_admin:
        username = request.form['username']
        password = generate_password_hash(request.form['password'])  # Hash the password
        new_user = User(username=username, password=password, is_admin=False)  # Register as a regular user
        db.session.add(new_user)
        db.session.commit()
        flash('User registered successfully!')
        return redirect(url_for('admin'))
    return redirect(url_for('index'))

# Register route (for admin creation only)
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if an admin user already exists
    existing_admin = User.query.filter_by(is_admin=True).first()
    
    if existing_admin:
        flash('Admin user already exists. Please log in.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        new_user = User(username=username, password=password, is_admin=True)  # Register as admin
        db.session.add(new_user)
        db.session.commit()
        flash('Admin user registered successfully!')
        return redirect(url_for('login'))

    return render_template('register.html')
def resize_image(image, width=None, height=None, scale_factor=None):
    """
    Resize image with multiple options
    
    Args:
        image (numpy.ndarray): Input image
        width (int): Desired width
        height (int): Desired height
        scale_factor (float): Scale factor to resize image
    
    Returns:
        numpy.ndarray: Resized image
    """
    if image is None:
        return None
    
    if scale_factor:
        # Resize by scale factor
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
    
    if width and height:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return image


def imagepath_to_tensor(image_path):
    """
    Loads an image, resizes it, converts it to a tensor, and normalizes it.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tf.Tensor: Transformed image tensor.
    """
    # Read the image file
    image = tf.io.read_file(image_path)

    # Decode the image to a tensor and convert it to RGB
    image = tf.image.decode_image(image, channels=3)

    # Resize the image to the desired dimensions (256x128)
    image = tf.image.resize(image, [256, 128])

    # Normalize the image to the range [0, 1]
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    return image

def image_to_tensor(image):
    """
    Loads an image, resizes it, converts it to a tensor, and normalizes it.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tf.Tensor: Transformed image tensor.
    """
    image = tf.image.resize(image, [256, 128])

    # Normalize the image to the range [0, 1]
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    return image
import os
import cv2
import tensorflow as tf
import numpy as np

def process_images(folder_path, cam2):
    """
    Process all images in a folder and pass them to the ReID model for prediction.

    :param folder_path: Path to the folder containing images (e.g., 'cam1').
    :param cam2: Second input image (e.g., as a NumPy array) for the ReID model.
    """
    predictions = []  # To store predictions with file paths

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No valid image files found in the folder.")
        return predictions

    # Load the model once outside the loop
    upmodel = tf.keras.models.load_model('updatedreid_model.h5')
    #model_a = tf.keras.models.load_model('reid3modelcuhk_a.h5')
    #model_b = tf.keras.models.load_model('reid3modelcuhk_b.h5')
    #model_c = tf.keras.models.load_model('reid2modelcuhk_c.h5')

    # Loop through each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(image_path)

        # Validate file path and read the image
        if not os.path.isfile(image_path):
            print(f"Warning: {image_path} is not a valid file. Skipping...")
            continue

        # Preprocess images for the model
        upcam1 = preprocess_image(image_path)
        upcam2 = preprocess_image(cam2)
        #query_features = extract_features([model_a], image_path)
        #gallery_features = extract_features([model_c], cam2)

        #fused_query_features = fuse_features2(query_features)
        #is_same, distance = is_same_person(fused_query_features, gallery_features, threshold=0.39)

        # Predict using the ReID model
        predicts = upmodel.predict([upcam1, upcam2])

        # Append prediction and file path
        #predictions.append((image_path, distance))
        predictions.append((image_path, predicts))

    # Sort predictions in descending order based on the prediction value
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Ensure the output folder exists
    reid_folder = "reid_folder"

    # Process top 3 predictions
    for idx, (image_pat, pred) in enumerate(predictions_sorted[:3], start=1):
        # Read raw images for concatenation
        cam1_img = cv2.imread(image_pat) # Read cam1 image directly
        img = cv2.resize(cam1_img, (128, 256))
        cam2_img = cv2.imread(cam2)
        img2 = cv2.resize(cam2_img, (128, 256))
        if cam1_img is None:
            print(f"Warning: Unable to read {image_pat}. Skipping concatenation...")
            continue

        # Resize cam2 to match cam1's dimensions
        cam2_resized = cv2.resize(img2, (img.shape[1], img.shape[0]))

        # Concatenate cam1 and cam2 horizontally
        concatenated_image = cv2.hconcat([img, cam2_resized])

        # Save the concatenated image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(reid_folder, f"concatenated_{timestamp}_{idx}.png")
        
        # Save the concatenated image
        cv2.imwrite(output_path, concatenated_image)
        print(f"Concatenated image saved to {output_path}")

    # Print sorted predictions
    print("Sorted Predictions:")
    for image_path, pred in predictions_sorted:
        print(f"{image_path}: {pred}")

    return predictions_sorted



from flask import Flask, request, jsonify, send_from_directory
import os

# Define folder paths
CAM1_FOLDER = os.path.join(os.getcwd(), 'snapshots_cam1')
CAM2_FOLDER = os.path.join(os.getcwd(), 'snapshots_cam2')

# Variable to store the clicked image for Re-ID model
selected_image_path = None

@app.route('/get-images/<folder_name>')
def get_images(folder_name):
    """Fetch image list from the given folder."""
    if folder_name == 'snapshots_cam1':
        folder_path = CAM1_FOLDER
    elif folder_name == 'snapshots_cam2':
        folder_path = CAM2_FOLDER
    else:
        return jsonify({'error': 'Invalid folder'}), 400

    images = [
        f"/{folder_name}/{file}" for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg', 'gif'))
    ]
    return jsonify({'images': images})

@app.route('/<folder_name>/<filename>')
def serve_image(folder_name, filename):
    """Serve images from the corresponding folder."""
    if folder_name == 'snapshots_cam1':
        folder_path = CAM1_FOLDER
    elif folder_name == 'snapshots_cam2':
        folder_path = CAM2_FOLDER
    else:
        return "Invalid folder", 404

    return send_from_directory(folder_path, filename)


selected_image_path = None
@app.route('/save-image', methods=['POST'])
def save_image():
    """Save the clicked image path to a variable for Re-ID processing."""

    global selected_image_path
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'success': False, 'message': 'No image URL provided'}), 400

    # Extract folder and file name from the URL
    parts = image_url.split('/')
    folder_name = parts[-2]
    filename = parts[-1]

    if folder_name == 'snapshots_cam1':
        folder_path = CAM1_FOLDER
    elif folder_name == 'snapshots_cam2':
        folder_path = CAM2_FOLDER
    else:
        return jsonify({'success': False, 'message': 'Invalid folder name'}), 400

    # Construct full image path
    image_path = os.path.join(folder_path, filename)

    # Check if the file exists
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'message': 'Image not found'}), 404

    # Save the selected image path in a global variable
    selected_image_path = image_path
    image_tensor = imagepath_to_tensor(selected_image_path)
    preds = process_images('snapshots_cam1',selected_image_path)
    



# Print tensor details
    print(f"Tensor Shape: {image_tensor.shape}")
    print(image_tensor)
    print(f"Selected image for Re-ID: {selected_image_path}")

    return jsonify({'success': True, 'message': 'Image path saved successfully', 'image_path': selected_image_path})



net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

net1 = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes1 = [line.strip() for line in f.readlines()]

layer_names1 = net.getLayerNames()
output_layers1 = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Globals for video and object detection
 # Replace with 0 for webcam
 
cap1 = cv2.VideoCapture('A04_20241215162738.mp4')
cap2 = cv2.VideoCapture('A02_20241215162738part2.mp4')
frame = None
frame2 = None
boxes,boxes2, confidences,confidences2, class_ids,class_ids2 = [], [], [],[],[],[]
snapshot_taken = []
def create_snapshot_folder(video_name):
    snapshot_dir = f'snapshots_{video_name}'
    os.makedirs(snapshot_dir, exist_ok=True)  # Ensure the directory exists
    return snapshot_dir

# Increase the threshold to reduce duplicate snapshots for the same person
def is_duplicate(new_box, existing_centers, threshold=250):
    new_center_x = new_box[0] + new_box[2] // 2
    new_center_y = new_box[1] + new_box[3] // 2

    for center_x, center_y in existing_centers:
        if abs(new_center_x - center_x) < threshold and abs(new_center_y - center_y) < threshold:
            return True

    return False

def detect_objects(frame, snapshot_dir,gen):
    """
    Detect objects in the given frame and return bounding boxes for 'person'.
    """
    global boxes, confidences, class_ids
    final_boxes = []
    height, width = frame.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes.clear()
    confidences.clear()
    class_ids.clear()
    

    # Loop over detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person":  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])

        # Create a center point for the detected person
        center_point = (x + w // 2, y + h // 2)

        # Draw the bounding box for the detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not is_duplicate(box, snapshot_taken):
            person_snapshot = frame[y:y + h, x:x + w]
            if person_snapshot.size > 0:  # Ensure valid region
                #imageten = image_to_tensor(person_snapshot)
                #snapimage = generate_image2(person_snapshot, "0475_c2_4629.png")

                # Save the `person_snapshot`
                snapshot_path = os.path.join(snapshot_dir, f'snapshot_person_{len(snapshot_taken)}.png')
                cv2.imwrite(snapshot_path, person_snapshot)
                print(f'Snapshot saved as {snapshot_path}')
                snapimage = generate_image(snapshot_path, "0475_c2_4629.png")

                # Save the `snapimage` to the 'gen' folder
                snapimage_path = os.path.join(gen_dir, f'gen_image_{len(snapshot_taken)}.png')
                cv2.imwrite(snapimage_path, snapimage)
                print(f'Generated image saved as {snapimage_path}')

                # Append the center point of the person to the list
                snapshot_taken.append(center_point)



import time  # Import for delay handling

# Global variable to track the last snapshot time
last_snapshot_time = 0  

def detect_objects2(frame, snapshot_dir):
    """
    Detect objects in the given frame and return bounding boxes for 'person'.
    """
    global boxes2, confidences2, class_ids2, last_snapshot_time
    final_boxes2 = []
    height2, width2 = frame.shape[:2]

    # Prepare the frame for YOLO
    blob2 = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net1.setInput(blob2)
    detections = net1.forward(output_layers)

    boxes2.clear()
    confidences2.clear()
    class_ids2.clear()

    # Loop over detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id2 = np.argmax(scores)
            confidence2 = scores[class_id2]

            if confidence2 > 0.5 and classes[class_id2] == "person":  
                center_x = int(detection[0] * width2)
                center_y = int(detection[1] * height2)
                w = int(detection[2] * width2)
                h = int(detection[3] * height2)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes2.append([x, y, w, h])
                confidences2.append(float(confidence2))
                class_ids2.append(class_id2)

    # Apply Non-Max Suppression
    indices2 = cv2.dnn.NMSBoxes(boxes2, confidences2, score_threshold=0.5, nms_threshold=0.4)

    for i in indices2:
        box = boxes2[i]
        x, y, w, h = box
        label = str(classes[class_ids2[i]])

        # Create a center point for the detected person
        center_point = (x + w // 2, y + h // 2)

        # Draw the bounding box for the detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        if not is_duplicate(box, snapshot_taken) and (current_time - last_snapshot_time > 2):  # Check 2-second delay
            person_snapshot = frame[y:y + h, x:x + w]
            if person_snapshot.size > 0:  # Ensure valid region
                snapshot_path = os.path.join(snapshot_dir, f'snapshot_person_{len(snapshot_taken)}.png')
                cv2.imwrite(snapshot_path, person_snapshot)
                print(f'Snapshot saved as {snapshot_path}')
                snapshot_taken.append(center_point)
                last_snapshot_time = current_time  # Update the last snapshot time


window_names = []

def generate_frames(rcap,dirr,gen, delay=0.2):
    """
    Generate frames for video streaming.
    """
    global  frame
    while True:
        ret, frame = rcap.read()
        if not ret:
            break

        detect_objects(frame,dirr,gen)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(delay)
        
def generate_frames2(rcap,dirr,delay=0.2):
    """
    Generate frames for video streaming.
    """
    global  frame2
    while True:
        ret, frame2 = rcap.read()
        if not ret:
            break

        detect_objects2(frame2,dirr)

        _, buffer = cv2.imencode('.jpg', frame2)
        frame2 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
        time.sleep(delay)

dir1 = create_snapshot_folder("cam1")
dir2 = create_snapshot_folder("cam2")
gen_dir = 'gen'
os.makedirs(gen_dir, exist_ok=True)
@app.route('/video_feed1')
def video_feed1():
    """
    Route for live video feed.
    """ 
    return Response(generate_frames(cap1,dir1,gen_dir), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    """
    Route for live video feed.
    """
    return Response(generate_frames2(cap2,dir2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click_event', methods=['POST'])
def click_event():
    """
    Handle click event on the video stream to save the selected person's screenshot.
    """
    global frame, boxes
    if frame is None:
        return "No frame available", 400

    # Get click coordinates from POST request
    x_click = int(request.form['x'])
    y_click = int(request.form['y'])

    # Check if the click is within any bounding box
    for i, (x, y, w, h) in enumerate(boxes):
        if x <= x_click <= x + w and y <= y_click <= y + h:
            screenshot = np.frombuffer(frame, dtype=np.uint8)
            screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
            cropped_image = screenshot[y:y + h, x:x + w]
            filename = f'screenshot_yolo_person_{i}.png'
            cv2.imwrite(filename, cropped_image)
            return f"Screenshot saved as {filename}", 200

    return "No object selected", 400

# Livefeed route
@app.route('/livefeed')
def livefeed():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        return render_template('livefeed.html', user=user)
    return redirect(url_for('login'))


# Tools route
@app.route('/tools')
def tools():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        return render_template('tools.html', user=user)
    return redirect(url_for('login'))


# footage route
@app.route('/footage')
def footage():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        return render_template('footage.html', user=user)
    return redirect(url_for('login'))

# Camera Management route
@app.route('/cameramanagement')
def cameramanagement():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        return render_template('cameramanagement.html', user=user)
    return redirect(url_for('login'))







# History route
@app.route('/history')
def history():
    user = get_logged_in_user()  # Get the logged-in user
    if user:
        return render_template('history.html', user=user)
    return redirect(url_for('login'))

# generator route
UPLOAD_FOLDER = './static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['STATIC_FOLDER'] = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generator', methods=['GET', 'POST'])
def generator():
    user = get_logged_in_user()  # Get the logged-in user
    if not user:
            return redirect(url_for('login'))

    output_image_path = None
    reid_results = None

    if request.method == 'POST':
        if 'queryImage' not in request.files or 'galleryImage' not in request.files:
            print('No file part')
            return redirect(request.url)

        query_image = request.files['queryImage']
        gallery_image = request.files['galleryImage']

        # Secure the filenames
        query_image_filename = secure_filename(query_image.filename)
        gallery_image_filename = secure_filename(gallery_image.filename)

        # Save the uploaded images temporarily in the upload folder
        query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], query_image_filename)
        gallery_image_path = os.path.join(app.config['UPLOAD_FOLDER'], gallery_image_filename)

        query_image.save(query_image_path)
        gallery_image.save(gallery_image_path)

        # Process images and generate output
        cam1 = query_image_path
        cam2 = gallery_image_path
        process_pose_image(cam2, "pose8.png")
        output_image = generate_image(cam1, "pose8.png")

        # Save the generated image
        output_image_path = 'gen_image.png'
        tf.keras.preprocessing.image.save_img(os.path.join(app.config['STATIC_FOLDER'], output_image_path), output_image)

        # Load reid models
        model_a = tf.keras.models.load_model('reid3modelcuhk_a.h5')
        model_b = tf.keras.models.load_model('reid3modelcuhk_b.h5')
        model_c = tf.keras.models.load_model('reid2modelcuhk_c.h5')
        umodel= tf.keras.models.load_model('updatedreid_model.h5')
        ucam1 = preprocess_image(cam1)
        ucam2 = preprocess_image(cam2)
        pred = umodel.predict([ucam1,ucam2])

        # Extract features and perform reid
        query_features = extract_features([model_a], cam1)
        synthesized_features = extract_features([model_b], output_image_path)
        gallery_features = extract_features([model_c], cam2)

        fused_query_features = fuse_features(query_features, synthesized_features, query_weight=0.5, synthesized_weight=0.5)
        is_same, distance = is_same_person(fused_query_features, gallery_features, threshold=0.39)
        # Format reid results
        if pred > 0.68 :
            reid_results = f"The query and the provided images (gallery and generated) correspond to the same individual{pred}"
        else:
            reid_results = f"The query image and the provided images (gallery and generated) are from different individual {pred}"

    return render_template('generator.html', output_image_path=output_image_path, reid_results=reid_results, user=user)



@app.route('/get_user_info', methods=['POST'])
def get_user_info():
    user_id = request.json['userId']
    user = User.query.get(user_id)
    if user:
        return jsonify({
            'id': user.id,
            'full_name': user.full_name,
            'security_id': user.security_id,
            'gender': user.gender,
            'age': user.age,
            'contact': user.contact,
            'email': user.email
        })
    else:
        return jsonify({'error': 'User not found'}), 404

if __name__ == "__main__":
    app.run(debug=True)
