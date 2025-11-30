💬 Comment Categorization & Reply Assistant

📌 Project Overview
This tool is an AI-powered application designed to help content creators and community managers efficiently handle user feedback. It uses Natural Language Processing (NLP) to analyze comments and categorize them into 8 distinct intents (such as Praise, Hate, or Constructive Criticism). It also generates empathetic, context-aware reply suggestions.

🎯 Key Features

Smart Classification: Distinguishes between similar categories (e.g., differentiating "Hate" from Constructive Criticism).

Reply Assistant: Suggests appropriate responses based on the sentiment (e.g., thanking for praise, acknowledging criticism).

Visual Analytics: Displays a dynamic Pie Chart showing the distribution of comment types.

Batch Processing: Allows users to upload a CSV file to categorize hundreds of comments instantly.

🛠️ Tech Stack

Language: Python 3.x

Machine Learning: Scikit-Learn (Logistic Regression + TF-IDF Vectorization)

Web Interface: Streamlit

Data Manipulation: Pandas

Visualization: Plotly Express

🚀 How to Run the Project

1. Install Dependencies

Open your terminal/command prompt and run:

pip install -r requirements.txt



2. Train the Model

Before running the app, you must train the AI model. This script reads the dataset, trains the classifier, and saves the model as comment_model.pkl.

python train.py



Current Performance:

Training Accuracy: ~97.28%

Test Set Accuracy: ~55-60% (Expected for small datasets)

3. Launch the Application

Start the web interface:

streamlit run app.py



The application will open in your default web browser (usually at http://localhost:8501).

📂 Project Structure

| File | Description |
| app.py | The main application code containing the Streamlit UI and logic. |
| train.py | The machine learning script. Loads data, trains the Pipeline, and saves the model. |
| comment_dataset.csv | The labeled dataset containing 250+ comments across 8 categories. |
| requirements.txt | List of Python libraries required to run the project. |
| comment_model.pkl | The saved binary file of the trained model (generated after running train.py). |

📊 Categories & Logic

The model classifies comments into the following buckets:

Praise: Positive feedback (e.g., "Amazing work!").

Support: Encouragement (e.g., "Keep going, you got this").

Constructive Criticism: Useful negative feedback (e.g., "Great video, but the audio was low").

Hate/Abuse: Toxic content (e.g., "This is trash").

Threat: Dangerous language (e.g., "I will hack you").

Emotional: Personal stories or deep connections.

Irrelevant/Spam: Self-promotion or off-topic links.

Question/Suggestion: Inquiries or content ideas.

Handling Logic in UI

🔴 Red Alert: For Hate and Threats. Suggests blocking/reporting.

🟡 Yellow Alert: For Constructive Criticism. Suggests taking notes for improvement.

🟢 Green Status: For Praise, Support, etc. Suggests engaging positively.

📈 Future Improvements

Expand the dataset to 1000+ rows to improve test-set generalization.

Integrate a Deep Learning model (like BERT) for even higher nuance detection.

Add a "Download Report" PDF feature.
