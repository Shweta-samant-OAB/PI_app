import json
import random
import string

filename = 'authentication.json'

def generate_random_string(length=50):
    all_characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choices(all_characters, k=length))
    return random_string

def add_new_user_cred():
    with open(filename, "r") as json_file:
        USER_CREDENTIALS = json.load(json_file)
    
    add_new_user = input("Enter Email ID")
    create_password = generate_random_string()
    USER_CREDENTIALS.update({add_new_user:create_password})
    
    with open(filename, 'w') as json_file:
        json.dump(USER_CREDENTIALS, json_file)
        
    print(f'User : {add_new_user} has been added.')
    
    
def rotate_password():
    pass
        
if __name__ == "__main__":
    add_new_user_cred()
