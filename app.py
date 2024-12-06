from flask import Flask, render_template, redirect, url_for, request, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import current_user, login_required, LoginManager, login_user, logout_user
from db import db  # Import db from db.py
from flask_migrate import Migrate
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
from pose import process_pose_image
from gen import generate_image
from reid import extract_features,fuse_features,compute_euclidean_distance,rank_gallery,compare_with_gallery,is_same_person
import tensorflow as tf



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

        # Extract features and perform reid
        query_features = extract_features([model_a], cam1)
        synthesized_features = extract_features([model_b], output_image_path)
        gallery_features = extract_features([model_c], cam2)

        fused_query_features = fuse_features(query_features, synthesized_features, query_weight=0.5, synthesized_weight=0.5)
        is_same, distance = is_same_person(fused_query_features, gallery_features, threshold=0.39)

        # Format reid results
        if is_same:
            reid_results = f"The query and the provided images (gallery and generated) correspond to the same individual"
        else:
            reid_results = f"The query image and the provided images (gallery and generated) are from different individual"

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
