import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Load the data
df = pd.read_csv('/home/maali/code/3Maali/VAK_ML/raw_data/dataset.csv')

# Check the distribution of styles before balancing
print("Distribution of styles before balancing:")
print(df['Type'].value_counts())

# Balance the data
min_samples = df['Type'].value_counts().min()  # Number of samples in the smallest class
df_auditory = resample(df[df['Type'] == 'Auditory'], replace=False, n_samples=min_samples, random_state=42)
df_kinesthetic = resample(df[df['Type'] == 'Kinesthetic'], replace=False, n_samples=min_samples, random_state=42)
df_visual = resample(df[df['Type'] == 'Visual'], replace=False, n_samples=min_samples, random_state=42)
df_balanced = pd.concat([df_auditory, df_kinesthetic, df_visual])

# Check the distribution of styles after balancing
print("\nDistribution of styles after balancing:")
print(df_balanced['Type'].value_counts())

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize stopwords
stop_words = set(stopwords.words('english'))
# Add common words that do not indicate a specific style
custom_stop_words = stop_words.union({
    'learn', 'learning', 'understand', 'understanding', 'better', 'best',
    'love', 'like', 'enjoy', 'prefer', 'feel', 'think', 'know', 'help',
    'make', 'need', 'want', 'use', 'way', 'good', 'great'
})

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in custom_stop_words]
    return ' '.join(words)

# Apply preprocessing
df_balanced['Processed_Sentence'] = df_balanced['Sentence'].apply(preprocess_text)

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Convert sentences to BERT embeddings
X = np.array([get_bert_embedding(sent) for sent in df_balanced['Processed_Sentence']])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_balanced['Type'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the XGBoost model
xgb_model = XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=1)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Style')
plt.ylabel('True Style')
plt.show()

# Save the model
joblib.dump(xgb_model, 'xgb_bert_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Test
input_text = "I learn by practicing tasks physically."
processed_text = preprocess_text(input_text)
input_vec = get_bert_embedding(processed_text)
pred = xgb_model.predict([input_vec])
probs = xgb_model.predict_proba([input_vec])[0]
predicted_style = label_encoder.inverse_transform(pred)[0]
print(f"\nInput: {input_text}")
print(f"Dominant Style: {predicted_style}")
print("Prediction Confidence:")
for style, prob in zip(label_encoder.classes_, probs):
    print(f"{style}: {prob*100:.2f}%")
