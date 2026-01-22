import pandas as pd
from evidently import Report
from evidently.metrics import DatasetMissingValueCount
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.tests import eq
from sklearn import datasets

reference_data = datasets.load_iris(as_frame=True).frame
reference_data = reference_data.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "prediction",
    }
)

current_data = pd.read_csv("prediction_database.csv", skipinitialspace=True)
current_data = current_data.drop(columns=["time"])

report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
snapshot = report.run(reference_data=reference_data, current_data=current_data)
snapshot.save_html("report.html")

data_test = Report(metrics=[DatasetMissingValueCount(tests=[eq(0)])])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
# Accessing test results from metric results is different in Report API
# We iterate through metrics to find tests
all_passed = True
for metric_result in result["metrics"]:
    if "tests" in metric_result:
        for test in metric_result["tests"]:
            if test["status"] != "SUCCESS":
                all_passed = False
                break

print("All tests passed: ", all_passed)
