from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run some code on application startup and shutdown.
    1. Code before 'yield' runs on startup.
    2. Code after 'yield' runs on shutdown.
    """
    print("Hello")
    yield
    print("Goodbye")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST model inference API!"}


# item_id is a `path parameter`
@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}


# bpm is a `query parameter`
@app.get("/health")
def health_check(bpm: int):
    """Health check endpoint."""
    if bpm < 30 or bpm > 200:
        return {"status": "unhealthy"}
    return {"status": "healthy"}
