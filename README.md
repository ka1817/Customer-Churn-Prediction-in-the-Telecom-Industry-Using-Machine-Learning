# 📉 Customer Churn Prediction in the Telecom Industry Using Machine Learning

## 🧠 Problem Statement

Telecom companies face customer churn as a significant challenge, directly impacting revenue and business continuity. By leveraging historical customer behavior data, we aim to build a machine learning system to **predict churn** proactively, enabling companies to implement targeted retention strategies.

---

## 🧾 Introduction

This project delivers a comprehensive machine learning solution aimed at predicting customer churn in the telecom industry. By analyzing historical customer data, the system identifies individuals who are likely to discontinue services, enabling telecom providers to implement proactive retention strategies and reduce revenue loss.

The entire workflow is structured around the complete machine learning lifecycle. It begins with data ingestion and preprocessing using robust pipelines, followed by model training and hyperparameter tuning across multiple algorithms. Experiment tracking and model versioning are managed through MLflow to ensure reproducibility and transparency. A FastAPI-based backend enables real-time predictions, while a responsive frontend built with HTML and CSS provides an accessible user interface. The entire application is containerized using Docker, ensuring consistency across development and deployment environments. To maintain high code quality and automation, continuous integration and deployment are configured using GitHub Actions. Finally, the solution is deployed on an AWS EC2 instance, making it accessible and scalable in a cloud environment.

This project exemplifies the transition from exploratory data analysis to a fully operational, production-ready system, aligning with modern MLOps practices and software engineering standards.


---

## 🗂️ Project Structure

```bash
TELECOM_CUSTOMER_CHURN/
│
├── data/                     # Raw and processed datasets
├── images/                   # Project-related visual assets
├── logs/                     # Logging output
├── mlartifacts/              # ML artifacts saved during runs
├── mlruns/                   # MLflow experiment logs
├── models/                   # Serialized trained models (.pkl)
├── notebooks/                # EDA and experimentation notebooks
├── src/                      # Source scripts
│   ├── data_ingestion.py     # Load and return the dataset
│   └── data_preprocessing.py # Preprocessing logic (pipelines)
│   └── model_training.py     # Training and model selection
├── static/                   # Static CSS files
│   └── css/
├── templates/                # Frontend templates
│   └── index.html            # Main UI page
├── tests/                    # Unit and functional tests
├── .github/                  # GitHub Actions workflow
├── Dockerfile                # Container build definition
├── main.py                   # FastAPI backend app
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── README.md                 # Project documentation
```

---

## 📊 Features

* End-to-end machine learning pipeline
* MLflow tracking for experiments and models
* Frontend (HTML + CSS) for model inference
* FastAPI backend for real-time prediction
* Dockerized application
* CI/CD pipeline with GitHub Actions
* Deployed on AWS EC2 with Docker

---

## ⚙️ Tech Stack

* **Languages**: Python, HTML, CSS
* **Libraries**: scikit-learn,pandas,seaborn,matplotlib,joblib, FastAPI, MLflow
* **Tools**: Docker, GitHub Actions, EC2
* **CI/CD**: Pytest + GitHub Actions + DockerHub

---

## 🚀 Setup Instructions

### 1. 🔁 Clone the Repository

```bash
git clone https://github.com/ka1817/Customer-Churn-Prediction-in-the-Telecom-Industry-Using-Machine-Learning.git
cd Customer-Churn-Prediction-in-the-Telecom-Industry-Using-Machine-Learning
```

### 2. 🐍 Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📈 Run ML Pipeline Locally

```bash
python src/model_training.py
```

* Logs will be stored in the `logs/` directory.
* Trained models are saved in the `models/` directory.
* MLflow UI: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Start MLflow UI separately:

```bash
mlflow ui --port 5000
```

---

## 🥪 Run Unit Tests

```bash
pytest tests/
```

---

## 🌐 Run FastAPI App Locally

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🐳 Docker Commands

### Build Docker Image

```bash
docker build -t pranavreddy123/telecom-churn:latest .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 pranavreddy123/telecom-churn:latest
```

---

## ☁️ AWS EC2 Deployment Steps

### 1. SSH into EC2

```bash
ssh -i your-key.pem ec2-user@your-ec2-public-ip
```

### 2. Install Docker on EC2

```bash
sudo yum update -y
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

### 3. Pull and Run Docker Image

```bash
docker pull pranavreddy123/telecom-churn:latest
docker run -d -p 8000:8000 pranavreddy123/telecom-churn:latest
```

Access the app at: `http://<your-ec2-ip>:8000`

---

## 🔀 CI/CD with GitHub Actions

* On every `push` or `pull request` to `main`:

  * Runs unit tests using `pytest`
  * Starts temporary MLflow server
  * Builds and pushes Docker image to DockerHub

Workflow defined in: `.github/workflows/ci-cd.yml`

---

## 🔗 Project Links

* 🐙 GitHub: [Customer-Churn-Prediction-in-the-Telecom-Industry-Using-Machine-Learning](https://github.com/ka1817/Customer-Churn-Prediction-in-the-Telecom-Industry-Using-Machine-Learning)
* 🐳 DockerHub: [pranavreddy123/telecom-churn](https://hub.docker.com/r/pranavreddy123/telecom-churn)

---

## 🧑‍💻 Author

**Pranav Reddy**
*Machine Learning Engineer | Open Source Contributor*
