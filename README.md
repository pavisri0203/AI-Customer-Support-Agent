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

## 🛠️ Technologies

- Python
- OpenAI API
- Scikit-learn
- TF-IDF
- Logistic Regression
- Pandas
- Streamlit

## 📊 Evaluation

The system was evaluated using a reviewed golden set of **150 customer support examples**.

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority Baseline | 36.7% | 10.7% |
| TF-IDF + Logistic Regression | 50.0% | 23.1% |
| AI Customer Support Agent | 99.3% | 99.2% |

**Decision Accuracy:** 99.3%

The result is based on an AI-assisted reviewed dataset and should be considered an optimistic evaluation.

## 🔎 Failure Analysis

One intent error was observed involving an image/link-only customer message.

Future versions can improve this using multimodal image understanding and human escalation for unclear queries.

## ⚠️ Limitations

- Evaluation dataset contains 150 reviewed examples.
- AI-assisted labeling may make the result optimistic.
- Text-based classification may struggle with image-only messages.
- Full LLM-based reply quality evaluation was not completed due to API request limits.

## ▶️ How to Run

Install dependencies:

`pip install openai pandas scikit-learn streamlit`

Set your OpenAI API key:

`export OPENAI_API_KEY="your_api_key"`

Run the application:

`streamlit run app.py`

## 🔮 Future Enhancements

- Multimodal support for image-based issues
- Larger independently labeled datasets
- Multilingual customer support
- Improved RAG with vector search
- Conversation history and memory
- Advanced human escalation
- LLM-based response quality evaluation
- Real-time support analytics

## 👩‍💻 Project

**AI Customer Support Agent**

AI-based customer support automation using **Intent Classification, RAG, Escalation, and AI Response Generation**.
