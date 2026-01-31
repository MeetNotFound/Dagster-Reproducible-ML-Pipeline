from dagster import asset
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =========================
# DATA INGESTION
# =========================
@asset
def raw_data():
    start = time.time()
    data = load_breast_cancer(as_frame=True)
    df = data.frame.sample(frac=0.9, random_state=42)
    print(f"[raw_data] Loaded in {time.time() - start:.2f} seconds")
    return df

# =========================
# EDA
# =========================
@asset
def eda_summary(raw_data):
    return raw_data.describe()


# =========================
# TRAIN / TEST SPLIT
# =========================
@asset
def train_test(raw_data):
    X = raw_data.drop("target", axis=1)
    y = raw_data["target"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# =========================
# MODELS
# =========================
@asset
def decision_tree(train_test):
    X_train, X_test, y_train, y_test = train_test
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


@asset
def random_forest(train_test):
    X_train, X_test, y_train, y_test = train_test
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


@asset
def logistic_regression(train_test):
    X_train, X_test, y_train, y_test = train_test
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


@asset
def knn(train_test):
    X_train, X_test, y_train, y_test = train_test
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc
