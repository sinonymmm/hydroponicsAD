import firebase_admin
from firebase_admin import credentials, auth

def initialize_firebase():
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

def create_user(email, password):
    user = auth.create_user(email=email, password=password)
    return user

def verify_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except firebase_admin.auth.UserNotFoundError:
        return None
