# 🤖 AI Customer Support Agent

An AI-powered customer support system that understands customer queries, identifies the issue, retrieves relevant historical support information, generates helpful replies, and decides whether to handle or escalate the issue.

## 🎯 Objectives

- Automate customer query classification
- Retrieve relevant historical support solutions
- Generate context-based AI replies
- Detect issues requiring human escalation
- Evaluate system performance

## ✨ Features

- 🔍 Intent Classification
- 📚 Retrieval-Augmented Generation (RAG)
- 🚨 Automatic Escalation
- 💬 AI Response Generation
- 📊 Golden Set Evaluation
- 🔎 Failure Analysis

## 🧠 Supported Intents

- `general_issue`
- `update_problem`
- `battery_issue`
- `app_feature_problem`
- `device_hardware_issue`

## 🏗️ Workflow

```text
Customer Message
      ↓
Intent Classification
      ↓
RAG Retrieval
      ↓
Escalation Check
      ↓
AI Response
      ↓
Final Customer Reply

🛠️ Technologies
Python
OpenAI API
Scikit-learn
TF-IDF
Logistic Regression
Pandas
Streamlit

📊 Evaluation
System	Accuracy	Macro F1
Majority Baseline	36.7%	10.7%
TF-IDF + Logistic Regression	50.0%	23.1%
AI Customer Support Agent	99.3%	99.2%

Golden Set: 150 reviewed examples


🔮 Future Enhancements
Multimodal image support
Larger independent datasets
Multilingual support
Improved RAG
Conversation memory
Advanced human escalation
