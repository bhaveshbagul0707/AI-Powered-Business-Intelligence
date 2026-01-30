🚀 AI-Powered Business Intelligence & Forecasting Platform

An end-to-end Business Intelligence platform that combines Machine Learning and Generative AI to convert raw sales data into clear insights, explanations, recommendations, and automated reports.

This project focuses on real-world usability, not just model accuracy — making insights understandable for non-technical decision-makers.

📌 Project Overview

Traditional dashboards show charts and numbers.
This platform goes a step further by answering:

What is happening? (EDA & KPIs)

What will happen next? (ML prediction)

Why did the model predict this? (Explainable AI)

What should the business do next? (Decision AI)

How can this be shared professionally? (Auto PDF report)

🔍 Features
📊 Exploratory Data Analysis (EDA)

Sales & profit overview

Category-wise sales analysis

Region-wise profit analysis

Business KPIs (Total Sales, Profit, Orders)

🤖 Machine Learning

Sales Prediction using Regression

Profit Classification (High / Low)

Feature scaling for improved model performance

🧠 Generative AI (Explainable & Decision AI)

Explains ML predictions in plain business language

Generates actionable business recommendations

Creates structured business reports automatically

📄 Auto Report Generation

Professional PDF reports

Ready to share with stakeholders

🔐 Secure Backend Design

API keys managed using .env

No secrets exposed in UI or source code

🏗️ Project Architecture
CSV Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
EDA & KPI Dashboard
   ↓
ML Prediction
   ↓
GenAI Explanation
   ↓
Business Recommendations
   ↓
Auto PDF Report
🛠️ Tech Stack

Programming: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib

Machine Learning: Scikit-learn

Generative AI: Groq (LLaMA 3.1)

Web App: Streamlit

Reporting: ReportLab

📂 Dataset Requirements

This project is designed for transactional business datasets such as:

Retail sales

E-commerce orders

Superstore sales data

Required columns:

Sales

Profit

Category

Region

Quantity

Discount

Order Date

Each row should represent one transaction.
▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/AI-Powered-Business-Intelligence.git
cd AI-Powered-Business-Intelligence

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here


⚠️ Do not commit .env to GitHub

5️⃣ Run the application
streamlit run app/app.py
📸 Screenshots
<img width="1887" height="880" alt="Screenshot 2026-01-30 195940" src="https://github.com/user-attachments/assets/72f5ef51-008a-4f84-bec9-d33caf3edeaa" />


🎯 Key Learnings

Building end-to-end ML pipelines

Feature engineering & scaling

Streamlit session state management

Explainable AI using GenAI

Secure backend API handling

Designing AI systems for business users

🚀 Future Improvements

Time-series forecasting (ARIMA / Prophet)

SHAP-based feature importance

Database integration

Role-based dashboards

Cloud deployment

👤 Author

Bhavesh Bagul
Computer Engineering Student
Interested in Data Science, Machine Learning & AI-driven systems

⭐ Support

If you find this project interesting, feel free to ⭐ the repository or share feedback.
