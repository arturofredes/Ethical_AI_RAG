# Setup
Setup should be prettu straightforward. The vectorstore is already created in the repo, but if you want to see how to create it check **chatbot.ipynb.**

Install requirements

```
pip install -r requirements.txt
```
## Setup login
Create a secret key and add it to your environment variables
```
chainlit create-secret
```

```
CHAINLIT_AUTH_SECRET=
```
Once you do this you can run the app

**user:** admin

**password:** admin

# Run the programm
This will host the application locally in port 8000.

```
chainlit run chat-ethicalai.py
```
