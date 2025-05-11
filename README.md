# VAK Learning Style Classifier

![VAK Logo](image/VAK-removebg.png)

Interactive web app to predict your learning style (Visual, Auditory, Kinesthetic) using a machine learning model. Built with Streamlit.

Live Demo: [VAK Learning Style Classifier](https://3maali-w5cunvfrkzykedebxtpsyz.streamlit.app/)

## About
Predicts learning style from sentences using a Logistic Regression model (95.15% accuracy). Features include:
- Predict dominant learning style
- Confidence scores
- Word cloud visualization
- Personalized learning tips
- Sample sentences
- Model details

## Technologies
- Python 3.8+
- Streamlit
- NLTK
- scikit-learn
- WordCloud, Matplotlib
- Joblib
- Pillow

---
---
## How to Use
1. Visit [live app](https://3maali-w5cunvfrkzykedebxtpsyz.streamlit.app/)
2. Enter sentences (e.g., "I learn best by watching videos")
3. Click "Predict" for results and tips
4. Check "Model Details" for ML info

## Example Sentences
- "I learn best by watching videos and diagrams." (Visual)
- "I prefer listening to lectures or podcasts." (Auditory)
- "I enjoy hands-on activities and experiments." (Kinesthetic)

## Model
- **Type**: Logistic Regression
- **Data**: Labeled sentences (`dataset.csv`)
- **Preprocessing**: Lowercase, remove punctuation, stopwords, lemmatize (NLTK)
- **Features**: TF-IDF Vectorizer (1000 features)
- **Accuracy**: 95.15% (3090 test samples)

## Setup
1. Clone repo: `git clone https://github.com/your-username/vak-learning-style-classifier.git`
2. Create virtual env: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Download NLTK data:
   ```python
   import nltk
   nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'])
