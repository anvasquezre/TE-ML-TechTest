from passlib.context import CryptContext
import json

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def get_password_hash(password):
    """
    Generate the hash of a password.

    Parameters:
    - password (str): The password to be hashed.

    Returns:
    - str: The hashed password.
    """
    return pwd_context.hash(password)

def create_user(db,user_data: dict[str, str]) -> None:
    """
    This function takes a dictionary with user data and appends it to the dummy_users_database.json file.
    The user_data dictionary should have the following keys: username, full_name, email, hashed_password, disabled.

    Args:
        db (str): The path to the dummy_users_database.json file.
        user_data (Dict[str, str]): A dictionary containing the user data.

    Returns:
        None
    """
    with open(db, 'r+') as file:
        data = json.load(file)
        new_id = str(int(max(data.keys())) + 1)  # Get the next id
        data[new_id] = user_data
        # Move the pointer to the beginning of the file and overwrite it with the updated data
        file.seek(0)
        json.dump(data, file, indent=4)

    return ("User created successfully.")



## This script allow to create a new user in the dummy_users_database.json file. You must execute this script directly from the terminal.
# Usage: python create_users.py
# Note: Don't update the repository showing the password in the code. This is just an example. It could be a security issue.
db = r"../data/dummy_users_database.json"
# Example usage:
new_user = {
    "username": "usernamex",  # Change this to the desired username
    "full_name": "full_name", # Change this to the desired full name
    "email": "email",        # Change this to the desired email
    "hashed_password": get_password_hash("new_password"),  # Change this to the desired password
    "disabled": False
}
create_user(db,new_user)

