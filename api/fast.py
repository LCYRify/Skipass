from fastapi import FastAPI

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"greeting":'halo'}

@app.get("/predict")
def predict(query):
    pass

