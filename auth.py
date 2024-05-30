import json
import os

USER_DATA_FILE = 'users.json'

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_user_data(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

def register_user(username, password):
    users = load_user_data()
    if username in users:
        return False
    users[username] = password
    save_user_data(users)
    return True

def login_user(username, password):
    users = load_user_data()
    if username in users and users[username] == password:
        return True
    return False
