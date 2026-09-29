# 📰 AI-Powered Fake News Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**An AI-powered web application that detects whether a news article is REAL or FAKE using Machine Learning and Natural Language Processing (NLP).**

</div>

---

# 📌 Overview

Fake news spreads rapidly through social media and online platforms, making it difficult for users to distinguish between genuine and misleading information.

This project presents an **AI-Powered Fake News Detection System** that analyzes textual news content using **Natural Language Processing (NLP)** techniques and predicts whether the article is **Real** or **Fake**.

The application provides an intuitive web interface built with **Streamlit**, allowing users to input news articles and receive instant predictions along with confidence scores and visual analytics.

---

# ✨ Features

✅ User-friendly Streamlit Dashboard

✅ Modern UI with Custom Styling

✅ Real-Time Fake News Detection

✅ NLP-based Text Preprocessing

✅ TF-IDF Feature Extraction

✅ Logistic Regression Classifier

✅ Confidence Score

✅ Interactive Confidence Gauge

✅ Prediction Probability Chart

✅ Prediction History

✅ Download Prediction History (CSV)

✅ Sample News Loader

---

# 🛠 Technologies Used

| Category | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| NLP | NLTK |
| Feature Extraction | TF-IDF Vectorizer |
| Visualization | Plotly |
| Data Handling | Pandas |
| Model Serialization | Joblib |

---

# 📂 Project Structure

```
AI_FAKE_NEWS_DETECTION
│
├── app
│   ├── app.py
│   ├── sample_news.py
│   └── style.css
│
├── data
│   ├── GlobalFakeNews_Research2026_v1.csv
│   └── cleaned_news.csv
│
├── images
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── label_distribution.png
│   ├── category_distribution.png
│   ├── country_distribution.png
│   └── source_distribution.png
│
├── models
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── results
│   └── prediction_history.csv
│
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── eda_visualization.py
│   └── utils.py
│
├── report
├── presentation
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙ Machine Learning Workflow

```
News Article
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
      │
      ▼
REAL / FAKE
```

---

# 📊 Dataset Information

- Total Articles: **1200**
- Real News: **600**
- Fake News: **600**
- Missing Values: **None**
- Balanced Dataset

---

# 🤖 Machine Learning Model

The project evaluates multiple machine learning algorithms and selects the best-performing model.

Models evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Multi-Layer Perceptron (MLP)

Selected Model:

**Logistic Regression**

---

# 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

> **Note:** These results were obtained on the provided project dataset. Performance on unseen, real-world news may vary depending on the diversity and quality of the input data.

---

# 🖥 Application Preview

Add screenshots here after uploading them.

## Home Page

```
images/home.png
```

## Prediction Result

```
images/prediction.png
```

## Confidence Dashboard

```
images/dashboard.png
```

## Prediction History

```
images/history.png
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI_Fake_News_Detection.git
```

Go to the project folder

```bash
cd AI_Fake_News_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

# 💡 Future Enhancements

- BERT-based Fake News Detection
- Deep Learning Models
- Live News API Integration
- Multilingual Fake News Detection
- Explainable AI (XAI)
- User Authentication
- Cloud Deployment
- Mobile Application

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

- Machine Learning
- Natural Language Processing
- Data Preprocessing
- TF-IDF Feature Engineering
- Model Evaluation
- Streamlit Web Application Development
- Data Visualization
- Software Project Organization

---

# 👩‍💻 Author

**Lahya Sree Bollinkala**

B.Tech Computer Science and Engineering

GitHub: https://github.com/lahyasreebollinkala

LinkedIn: https://linkedin.com/in/lahyasreebollinkala

project_URL: https://aifakenewspredictioniictnewdelhi-mvwgckvhvrymudzsmu5vbr.streamlit.app/
---

# 📄 License

This project is intended for educational and learning purposes.
