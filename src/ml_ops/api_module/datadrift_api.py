import pickle
from collections.abc import Generator
from datetime import datetime, timezone

import anyio
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from sklearn import datasets


def lifespan(app: FastAPI) -> Generator[None]:
    """Load model and classes, and create database file."""
    global model, classes
    classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        # In case model is not found, we might want to handle it,
        # but for now we follow the structure.
        # Ideally this should fail if model is required.
        print("Warning: model.pkl not found")
        model = None

    with open("prediction_database.csv", "w") as file:
        file.write("time, sepal_length, sepal_width, petal_length, petal_width, prediction\n")

    yield

    if model:
        del model


app = FastAPI(lifespan=lifespan)


def add_to_database(
    now: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    prediction: int,
) -> None:
    """Simple function to add prediction to database."""
    with open("prediction_database.csv", "a") as file:
        file.write(f"{now}, {sepal_length}, {sepal_width}, {petal_length}, {petal_width}, {prediction}\n")


@app.post("/predict")
async def iris_inference(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    background_tasks: BackgroundTasks,
):
    """Version 2 of the iris inference endpoint."""
    if model is None:
        return {"error": "Model not loaded"}

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = prediction.item()

    now = str(datetime.now(tz=timezone.utc))
    background_tasks.add_task(add_to_database, now, sepal_length, sepal_width, petal_length, petal_width, prediction)
    return {"prediction": classes[prediction], "prediction_int": prediction}


@app.get("/monitoring", response_class=HTMLResponse)
async def iris_monitoring():
    """Simple get request method that returns a monitoring report."""
    reference_data: pd.DataFrame = datasets.load_iris(as_frame=True).frame
    reference_data = reference_data.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "prediction",
        }
    )
    current_data = pd.read_csv("prediction_database.csv")
    current_data = current_data.drop(columns=["time"])

    # Use available presets: DataDriftPreset and DataSummaryPreset
    data_drift_report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    data_drift_report.run(current_data=current_data, reference_data=reference_data)
    data_drift_report.save_html("monitoring.html")

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
