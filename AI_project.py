import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set up the page
st.set_page_config(page_title="Spotify Dashboard", layout="wide")
st.title("🎵 Spotify Song Data Dashboard")
st.subheader("Explore, Visualize, and Predict Song Success")

# Load dataset
df = pd.read_csv("Spotify_final_dataset.csv")

# Clean whitespace
df['Artist Name'] = df['Artist Name'].str.strip()
df['Song Name'] = df['Song Name'].str.strip()

# Artist Filter
artist = st.selectbox("🎤 Choose an Artist", df['Artist Name'].unique())
filtered_data = df[df['Artist Name'] == artist]
st.write(f"🎧 Data for {artist}")
st.dataframe(filtered_data)

# Encoding with clean label encoders
artist_encoder = LabelEncoder()
song_encoder = LabelEncoder()

df['Artist Name Encoded'] = artist_encoder.fit_transform(df['Artist Name'])
df['Song Name Encoded'] = song_encoder.fit_transform(df['Song Name'])
df['Top 10 Binary'] = df['Top 10 (xTimes)'].apply(lambda x: 1 if x > 0 else 0)

X = df[['Days', 'Peak Position', 'Peak Streams', 'Total Streams', 'Artist Name Encoded', 'Song Name Encoded']]
y = df['Top 10 Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Selection
model_type = st.selectbox("🧠 Choose a model", ["Logistic Regression", "Random Forest"])
if model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report_dict).transpose()

# Display Metrics
st.write(f"✅ **Accuracy of {model_type}:** {accuracy:.2f}")
st.write("📊 **Confusion Matrix:**")
st.write(conf_matrix)
st.write("📋 **Classification Report:**")
st.dataframe(class_report_df.style.format("{:.2f}"))

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.matshow(conf_matrix, cmap='Blues', alpha=0.7)
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

# Scatter Plot
st.subheader("📈 Days vs Total Streams")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Days'], df['Total Streams'], c=df['Top 10 Binary'], cmap='viridis', s=100, alpha=0.7)
ax.set_xlabel("Days")
ax.set_ylabel("Total Streams")
ax.set_title("Days vs Total Streams")
fig.colorbar(scatter, label="Top 10 (1=Yes, 0=No)")
st.pyplot(fig)

# Pie Chart - Top 10 Frequency
st.subheader("🥧 Top 10 Frequency")
top10_counts = df['Top 10 Binary'].value_counts()
fig, ax = plt.subplots()
ax.pie(top10_counts, labels=["Not in Top 10", "Top 10"], autopct='%1.1f%%', colors=['lightgray', 'gold'], startangle=90)
ax.set_title("Distribution of Songs in Top 10")
st.pyplot(fig)

# Histogram - Genre-wise
if 'Genre' in df.columns:
    st.subheader("📊 Genre-wise Song Count")
    fig, ax = plt.subplots(figsize=(10, 5))
    df['Genre'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Genre Distribution")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Songs")
    st.pyplot(fig)

# Stream Filter
peak_streams = st.slider("🔍 Filter by Peak Streams", 0, int(df['Peak Streams'].max()), step=1000000)
st.write(f"🎶 Songs with Peak Streams ≥ {peak_streams:,}")
st.dataframe(df[df['Peak Streams'] >= peak_streams])

# Manual Prediction
st.subheader("🎯 Try Predicting a Song's Top 10 Potential Manually")

with st.form("manual_prediction_form"):
    days = st.number_input("Days on Chart", min_value=0, value=100)
    peak_position = st.number_input("Peak Position", min_value=1, value=10)
    peak_streams_input = st.number_input("Peak Streams", min_value=0, value=1000000)
    total_streams_input = st.number_input("Total Streams", min_value=0, value=5000000)
    
    artist_input = st.selectbox("Select Artist", df['Artist Name'].unique())
    song_input = st.selectbox("Select Song", df[df['Artist Name'] == artist_input]['Song Name'].unique())
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        artist_input = artist_input.strip()
        song_input = song_input.strip()
        try:
            artist_encoded = artist_encoder.transform([artist_input])[0]
            song_encoded = song_encoder.transform([song_input])[0]

            input_data = np.array([[days, peak_position, peak_streams_input, total_streams_input, artist_encoded, song_encoded]])
            prediction = model.predict(input_data)[0]
            result = "🎉 Yes, this song would make it to the Top 10!" if prediction == 1 else "❌ No, this song may not make it to the Top 10."
            st.success(f"📌 Prediction Result: {result}")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")
