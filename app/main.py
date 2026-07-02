# importing required dependancies
from fastapi import FastAPI

# initiating fastapi app
app = FastAPI(
    title="FastAPI Template",
    description="A simple FastAPI template",
    version="1.0.0"
)

# endpoints
@app.get("/")
def read_root():
    return {"message": "This is the root of house price prediction API"}


@app.get("/health")
def read_health():
    return{"status": "ok"}

