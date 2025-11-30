import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import random
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Comment Categorization Tool", layout="wide")
st.title("💬 Comment Categorization & Reply Assistant")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists('comment_model.pkl'):
        return joblib.load('comment_model.pkl')
    else:
        return None

model = load_model()

if model is None:
    st.error("🚨 Model not found! Please run `python train.py` in your terminal first.")
    st.stop()

# --- REPLY TEMPLATES ---
reply_templates = {
    "Praise": ["Thank you so much! 😊", "We really appreciate the love! ❤️", "Thanks! Glad you liked it."],
    "Support": ["Thanks for sticking with us! 🙏", "Your support means the world.", "We'll keep doing our best!"],
    "Constructive Criticism": ["Thanks for the feedback, we'll look into it. 📝", "Good point! We'll try to improve.", "Appreciate the honesty."],
    "Hate/Abuse": ["(Blocked User) 🚫", "Please stay respectful.", "Comment flagged."],
    "Threat": ["(Reported to Safety Team) ⚠️", "We take safety seriously.", "This behavior is not tolerated."],
    "Emotional": ["Sending virtual hugs! 🤗", "Thank you for sharing your story. ❤️", "We're touched by this."],
    "Irrelevant/Spam": ["(Deleted Spam)", "Please no spam.", "Let's stay on topic."],
    "Question/Suggestion": ["Great question! We'll answer soon.", "Thanks for the suggestion! Added to our list.", "Good idea!"]
}

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

# LEFT: Analysis Tool
with col1:
    st.subheader("📝 Analyze Comment")
    user_input = st.text_area("Enter a comment:", placeholder="e.g., Great video but the audio is low.")
    
    if st.button("Analyze"):
        if user_input:
            # Predict
            prediction = model.predict([user_input])[0]
            # Generate Reply
            reply = random.choice(reply_templates.get(prediction, ["Thanks!"]))
            
            st.info(f"**Category:** {prediction}")
            
            # Logic for color-coded feedback
            if prediction in ["Hate/Abuse", "Threat"]:
                st.error(f"🔴 **Action:** Block/Report\n\n**Reply:** {reply}")
            elif prediction == "Constructive Criticism":
                st.warning(f"🟡 **Action:** Take Feedback\n\n**Reply:** {reply}")
            else:
                st.success(f"🟢 **Action:** Engage\n\n**Reply:** {reply}")
        else:
            st.warning("Please type something.")

# RIGHT: Visualization
with col2:
    st.subheader("📊 Training Data Stats")
    
    # Check if file exists before trying to read it
    if os.path.exists("comment_dataset.csv"):
        try:
            df = pd.read_csv("comment_dataset.csv")
            if not df.empty and 'category' in df.columns:
                counts = df['category'].value_counts().reset_index()
                counts.columns = ['Category', 'Count']
                
                # Create Pie Chart
                fig = px.pie(counts, names='Category', values='Count', title="Distribution of Comments", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dataset is empty or missing 'category' column.")
        except Exception as e:
            st.error(f"Error loading chart: {e}")
    else:
        st.warning("Dataset 'comment_dataset.csv' not found to generate charts.")

# --- BATCH UPLOAD ---
st.divider()
st.subheader("📂 Batch Analysis (CSV)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        if 'comment_text' in batch_df.columns:
            batch_df['Category'] = model.predict(batch_df['comment_text'])
            batch_df['Reply'] = batch_df['Category'].apply(lambda x: random.choice(reply_templates.get(x, [""])))
            
            st.dataframe(batch_df)
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "results.csv", "text/csv")
        else:
            st.error("CSV must have a 'comment_text' column.")
    except Exception as e:
        st.error(f"Error processing file: {e}")