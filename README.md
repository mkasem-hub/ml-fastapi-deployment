# 🛠️ ML Model Deployment with FastAPI (Course 4 Project)

This repository contains the final project for the **Udacity Machine Learning DevOps Engineer Nanodegree**, which demonstrates deploying a classification model using **FastAPI**, **GitHub Actions**, **DVC**, and **Render**.

---

## 🔍 Project Overview

The project includes:
- Training and saving an ML model
- Creating API endpoints for inference
- Automating testing and deployment
- Deploying the service publicly via Render

---

## 📁 Environment Setup

To set up the development environment:

```bash
conda create -n ml-api-env python=3.8 scikit-learn pandas numpy pytest jupyterlab fastapi uvicorn -c conda-forge
conda activate ml-api-env
pip install -r requirements.txt  # optional
```

Install Git and DVC:
```bash
sudo apt install git
pip install dvc
```

If using Windows, WSL is recommended.

---

## 📊 Data Handling

- Download `census.csv` and commit it to DVC:
```bash
dvc add data/census.csv
git add data/census.csv.dvc .gitignore
git commit -m "Track census dataset with DVC"
```

- Clean data using Pandas (strip extra spaces from features and column names).
- Save cleaned version as `census_clean.csv`.

---

## 🤖 Model Training

- Implemented a `RandomForestClassifier` with preprocessing using OneHotEncoder.
- Wrote reusable `train_model`, `inference`, and `process_data` functions.
- Wrote performance evaluation for **data slices** (e.g., by `education`, `sex`, etc.).
- Wrote 3+ unit tests for model pipeline.
- Included a model card documenting model assumptions and limitations.

---

## 🚀 API with FastAPI

### Endpoints:
- `GET /` → Returns welcome message.
- `POST /inference` → Accepts JSON input and returns prediction (`<=50K` or `>50K`).

### Input model:
Used `Pydantic` with an example schema. Handles features with hyphens using FastAPI aliasing.

---

## ✅ API Testing

- Test script: `test_live_api.py`  
- Tests cover:
  - `GET /` root
  - `POST /inference` with valid input (positive test)
  - `POST /inference` with edge cases or invalid input (negative test)

Run:
```bash
python test_live_api.py
```

---

## 🔁 CI/CD with GitHub Actions

- `.github/workflows/main.yml` includes:
  - `pytest` for unit testing
  - `flake8` for linting
- CI passes required before deployment
- GitHub repo connected to Render for auto-deployment on push

---

## 🌐 Deployment on Render

**Live URL**:  
[https://ml-fastapi-deployment-xxxxx.onrender.com](https://ml-fastapi-deployment-xxxxx.onrender.com)

Deployment uses:
- `render.yaml`
- `runtime.txt`
- `gunicorn` + `uvicorn`

---

## 🧪 Model Inference Example (Live)

```python
import requests

data = {
    "age": 37,
    "workclass": "Private",
    "fnlwgt": 284582,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post("https://ml-fastapi-deployment-xxxxx.onrender.com/inference", json=data)
print(response.status_code)
print(response.json())
```

---

## 📄 Files Included

- `starter/main.py` – FastAPI app
- `starter/ml/` – Model training and preprocessing code
- `starter/model/` – Saved `model.pkl`, `encoder.pkl`, `lb.pkl`
- `requirements.txt`, `runtime.txt`, `render.yaml`
- `.github/workflows/main.yml` – GitHub Actions config
- `test_live_api.py` – API test script
- `README.md` – this documentation

---

## 📌 Notes

- Python version: `3.8` (set in `runtime.txt`)
- scikit-learn version: `1.3.2` (must match for `joblib` compatibility)
- API hosted on Render instead of Heroku (Heroku billing required)

---

## 👤 Author

**Mostafa Kasem**  
GitHub: [@mkasem-hub](https://github.com/mkasem-hub)
