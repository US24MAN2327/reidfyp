from db import db  # Import db from db.py

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)  # Full Name
    security_id = db.Column(db.String(50), unique=True, nullable=True)  # Security ID
    gender = db.Column(db.String(10), nullable=True)  # Gender
    age = db.Column(db.Integer, nullable=True)  # Age
    contact = db.Column(db.String(15), nullable=True)  # Contact
    email = db.Column(db.String(120), unique=True, nullable=True)  # Email
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<User {self.username}>'

