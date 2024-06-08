# Setup
Setup should be pretty straightforward. The vectorstore is already created in the repo, but if you want to see how to create it check `chatbot.ipynb`.

Install requirements

```
pip install -r requirements.txt
```

You will also need an Open AI API key in the environment variables
```
OPENAI_API_KEY = your_key
```
## Setup login
To setup the login page follow the following steps. 
Create a secret key and add it to your environment variables
```
chainlit create-secret
```

```
CHAINLIT_AUTH_SECRET= your_secret
```
Once you do this you can run the app

**user:** admin

**password:** admin
## Use with no Login page
Alternatively, you can comment the section under the auth decorator and no login will be needed.
```
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
```

# Run the programm
This will host the application locally in port 8000.

```
chainlit run chat-ethicalai.py
```
