import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


df = pd.read_csv('/home/maali/code/3Maali/VAK_ML/raw_data/dataset.csv')


df_types = df['Type'].unique()
print(df_types)

df['Type'].value_counts()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


df['Processed_Sentence'] = df['Sentence'].apply(preprocess_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Processed_Sentence'])

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Type'])
print(f"TF-IDF matrix shape: {X.shape}")
print(f"Encoded labels (first 5): {y[:5]}")
print(f"Original classes: {label_encoder.classes_}")

# Distribution of learning styles
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Type')
plt.title('Distribution of Learning Styles')
plt.xlabel('Learning Style')
plt.ylabel('Number of Sentences')
plt.show()

def plot_wordcloud(text, title):
    font_path = './Roboto-Regular.ttf'
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate word clouds
for learning_style in df['Type'].unique():
    style_text = ' '.join(df[df['Type'] == learning_style]['Processed_Sentence'])
    plot_wordcloud(style_text, f'Word Cloud for {learning_style} Style')


df['Sentence_Length'] = df['Processed_Sentence'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Type', y='Sentence_Length')
plt.title('Sentence Length Distribution by Learning Style')
plt.xlabel('Learning Style')
plt.ylabel('Number of Words in Sentence')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Style')
plt.ylabel('True Style')
plt.show()


joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
