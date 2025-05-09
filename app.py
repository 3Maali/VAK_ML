import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

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


model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')


MODEL_ACCURACY = 0.9515


helper_sentences = {
    "Visual": [
        "I learn best by watching videos and diagrams.",
        "I understand concepts better when I see charts.",
        "I prefer reading books with illustrations.",
        "I enjoy visualizing information in my mind.",
        "I find it easier to learn from slideshows."
    ],
    "Auditory": [
        "I prefer listening to lectures or podcasts.",
        "I learn better when I hear explanations.",
        "I enjoy discussing topics to understand them.",
        "I find audiobooks very helpful for learning.",
        "I remember information by repeating it aloud."
    ],
    "Kinesthetic": [
        "I enjoy hands-on activities and experiments.",
        "I like to move around while studying.",
        "I learn best by practicing tasks physically.",
        "I prefer building models to understand concepts.",
        "I find role-playing helpful for learning."
    ]
}


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');

    /* Main app background with Old Lace and a prominent gradient */
    .stApp {
        background-color: #F7F8E5; /* Old Lace as base background */
        background-image: linear-gradient(to bottom right, #F7F8E5, rgba(2, 2, 2, 0.4), rgba(35, 45, 35, 0.6), rgba(109, 106, 97, 0.5)); /* Gradient with Old Lace, Rich Black, Pine Tree, Dim Gray */
        background-size: cover;
        background-attachment: fixed; /* Fixed background for a smooth effect */
        font-family: 'Noto Sans Arabic', sans-serif;
        color: #232D23; /* Pine Tree for text to be dominant */
        margin: 0;
        padding: 0;
        width: 100vw;
        overflow-x: hidden;
        padding-top: 60px; /* Space for fixed nav bar */
    }

    /* Main content container */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
        margin: 0 auto;
    }

    /* Fixed navigation bar for tabs */
    .nav-bar {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        background-color: #232D23; /* Pine Tree for nav bar */
        padding: 0.5rem 1rem;
        z-index: 1000 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }


    /* Style the Streamlit tabs */
    div.stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        gap: 2rem;
        align-items: center;
        background-color: transparent;
        width: 100%;
        max-width: 1200px;
        justify-content: center;
        padding: 0 1rem;
    }

    div.stTabs [data-baseweb="tab"] {
        color: #F7F8E5 !important; /* Old Lace for tab text */
        background-color: transparent !important;
        font-family: 'Noto Sans Arabic', sans-serif;
        font-size: 1.5rem !important;
        padding: 0.5rem 1.5rem !important;
        border: none !important;
        transition: color 0.3s ease, border-bottom 0.3s ease;
        cursor: pointer;
    }

    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #F7F8E5 !important; /* Old Lace for active tab */
        font-weight: bold !important;
        border-bottom: 3px solid #232D23 !important; /* Pine Tree border */
    }

    div.stTabs [data-baseweb="tab"]:hover {
        color: #F7F8E5 !important;
        border-bottom: 3px solid #232D23 !important; /* Pine Tree border on hover */
    }

    /* Ensure tab content is not hidden behind nav bar */
    div.stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }

    /* Larger fonts for headings */
    h1 {
        font-size: 48px !important;
        color: #232D23 !important; /* Pine Tree for headings */
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    h3 {
        font-size: 30px !important;
        color: #232D23 !important; /* Old Lace for h3 */
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    h4 {
        font-size: 28px !important;
        color: #232D23 !important; /* Pine Tree for subheadings */
        margin: 0.5rem 0;
    }

    /* Body text */
    .stMarkdown, .stText, .stWrite {
        font-size: 18px !important;
        color: #232D23 !important; /* Old Lace for body text */
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #232D23; /* Pine Tree for sidebar */
        border-right: 2px solid #6D6A61; /* Dim Gray border */
        width: 250px !important;
    }
    .stSidebar .stButton>button {
        background-color: #F7F8E5; /* Old Lace for buttons */
        color: #232D23; /* Pine Tree for button text */
        font-size: 19px;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        margin-bottom: 10px;
        transition: background-color 0.3s ease;
    }
    .stSidebar .stButton>button:hover {
        background-color: #E5E6D3; /* Slightly darker Old Lace for hover */
    }

    /* Text area */
    .stTextArea textarea {
        font-size: 19px !important;
        border: 2px solid #232D23; /* Pine Tree border */
        border-radius: 8px;
        background-color: rgba(247, 248, 229, 0.9);  /* Old Lace background */
        color: #232D23; /* Pine Tree text */
        width: 100% !important;
    }
    /* Learning Tips Styling */
        .learning-tips {
        font-size: 24px !important; /* Larger font size for tips */
        color: #232D23 !important; /* Pine Tree for text */
        line-height: 1.6; /* Improved readability */
        margin: 0.5rem 0;
    }
     /* Examples  Styling */
        .examples {
        font-size: 24px !important; /* Larger font size for tips */
        color: #232D23 !important; /* Pine Tree for text */
        line-height: 1.6; /* Improved readability */
        margin: 0.5rem 0;
    }

    /* Predict and Reset buttons */
    .stButton>button {
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        margin: 5px;
        padding: 10px 20px;
        transition: transform 0.2s, background-color 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    /* Predict button */
    .stButton>button:first-child {
        background-color: #232D23; /* Pine Tree */
        color: #F7F8E5; /* Old Lace */
    }
    .stButton>button:first-child:hover {
        background-color: #1B231B; /* Darker Pine Tree */
    }
    /* Reset button */
    .stButton>button:nth-child(2) {
        background-color: #6D6A61; /* Dim Gray */
        color: #F7F8E5; /* Old Lace */
    }
    .stButton>button:nth-child(2):hover {
        background-color: #5A584F; /* Darker Dim Gray */
    }

    /* Success message */
    .stSuccess {
        font-size: 28px !important; /* Larger font for better visibility */
        font-weight: bold;
        background-color: rgba(35, 45, 35, 0.2) !important; /* Pine Tree with transparency */
        border: 2px solid #232D23 !important; /* Pine Tree border */
        border-radius: 8px;
        color: #232D23 !important;
        padding: 1.5rem;
        text-align: center;
        width: 100%;
    }

    /* Results container */
    .results-container {
        background-color: rgba(35, 45, 35, 0.1); /* Light Pine Tree background */
        border: 2px solid #232D23; /* Pine Tree border */
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        width: 100%;
        max-width: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Prediction confidence text */
    .confidence-text {
        font-size: 28px !important;
        color: #232D23 !important;
        text-align: Left;
    }

    /* Word Cloud container */
    .wordcloud {
        background-color: #F7F8E5; /* Old Lace */
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #232D23; /* Pine Tree border */
        width: 100%;
        max-width: 800px; /* Larger for laptop screens */
        margin: 0 auto;
        text-align: center;
    }

    /* Image styling */
    img {
        border-radius: 12px;
        margin: 0 auto;
        display: block;
        max-width: 100%;
    }

    /* Responsive design */
    @media (min-width: 768px) {
        .main .block-container {
            max-width: 1200px !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    }

    @media (max-width: 768px) {
        .stApp {
            padding-top: 100px;
        }
        .nav-bar {
            flex-direction: column;
            align-items: center;
            padding: 0.5rem;
        }
        div.stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
            gap: 0.5rem;
            width: 100%;
            margin-bottom: 0.5rem;
            justify-content: center;

        }
        div.stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0;
            text-align: center;
            font-size: 1rem !important;

        }
        .css-1d391kg {
            width: 100% !important;
        }
        .wordcloud {
            max-width: 100%; /* Full width on mobile */
        }
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>VAK Learning Style Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3>Visual, Auditory, Kinesthetic</h3>", unsafe_allow_html=True)

st.image("/home/maali/code/3Maali/VAK_ML/VAK-removebg.png", caption="Explore Your Learning Style with VAK", use_container_width=True)

# Sidebar
st.sidebar.markdown("### Get Sentence Suggestions")
st.sidebar.write("Click a button to get a sample sentence for a learning style:")


if "visual_index" not in st.session_state:
    st.session_state.visual_index = 0
if "auditory_index" not in st.session_state:
    st.session_state.auditory_index = 0
if "kinesthetic_index" not in st.session_state:
    st.session_state.kinesthetic_index = 0


if st.sidebar.button("Visual"):
    st.session_state.visual_index = (st.session_state.visual_index + 1) % len(helper_sentences["Visual"])
    st.sidebar.write(f"**Suggested Sentence**: {helper_sentences['Visual'][st.session_state.visual_index]}")

if st.sidebar.button("Auditory"):
    st.session_state.auditory_index = (st.session_state.auditory_index + 1) % len(helper_sentences["Auditory"])
    st.sidebar.write(f"**Suggested Sentence**: {helper_sentences['Auditory'][st.session_state.auditory_index]}")

if st.sidebar.button("Kinesthetic"):
    st.session_state.kinesthetic_index = (st.session_state.kinesthetic_index + 1) % len(helper_sentences["Kinesthetic"])
    st.sidebar.write(f"**Suggested Sentence**: {helper_sentences['Kinesthetic'][st.session_state.kinesthetic_index]}")


with st.container():
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    tabs = st.tabs(["Predict Learning Style", "Model Details"])
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 1: Predict Learning Style
with tabs[0]:
    st.header("Predict Learning Style")
    about = f"The VAK model categorizes learning styles into Visual (learning by seeing), Auditory (learning by hearing), and Kinesthetic (learning by doing) based on a sentence you provide. This app uses a machine learning model trained on labeled sentences with an accuracy of {MODEL_ACCURACY:.2%} on test data."
    st.markdown(f'<div class="learning-tips"> {about}</div>', unsafe_allow_html=True)


    examples = [
    "- I learn best by watching videos and diagrams. (Visual)",
    "- I prefer listening to lectures or podcasts. (Auditory)",
    "- I enjoy hands-on activities and experiments. (Kinesthetic)",
    "- I understand concepts better when I see charts.",
    "- I like to move around while studying.",
    "- I learn better by teaching others or explaining aloud. (Auditory)"
]

     # to add new line
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("#### Examples of Sentences to Try:")
    for example in examples:
        st.markdown(f'<div class="learning-tips"> {example} </div>', unsafe_allow_html=True)



    # st.markdown("""
    # - I learn best by watching videos and diagrams. (Visual)
    # - I prefer listening to lectures or podcasts. (Auditory)
    # - I enjoy hands-on activities and experiments. (Kinesthetic)
    # - I understand concepts better when I see charts.
    # - I like to move around while studying.
    # """)

    # to add new line
    st.markdown("<br><br>", unsafe_allow_html=True)


    st.markdown("#### Enter Your Sentence(s):")
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Type one or more sentences describing how you learn (separate multiple sentences with new lines):",
                              value=st.session_state.user_input, height=150)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset"):
            st.session_state.user_input = ""
            st.rerun()
    with col2:
        predict_button = st.button("Predict")


    if predict_button:
        if user_input:
            # Split input into sentences
            sentences = [s.strip() for s in user_input.split('\n') if s.strip()]
            if not sentences:
                st.error("Please enter at least one valid sentence.")
            else:
                predictions = []
                probabilities_list = []
                for sentence in sentences:

                    if len(sentence.split()) < 3:
                        st.warning(f"Sentence '{sentence}' is too short (minimum 3 words). Skipping.")
                        continue

                    # Preprocess the sentence
                    processed_text = preprocess_text(sentence)

                    # Transform into TF-IDF vector
                    text_vector = vectorizer.transform([processed_text])

                    # Make prediction and probabilities
                    prediction = model.predict(text_vector)
                    predicted_style = label_encoder.inverse_transform(prediction)[0]
                    probabilities = model.predict_proba(text_vector)[0]
                    predictions.append(predicted_style)
                    probabilities_list.append(probabilities)

                if predictions:
                    dominant_style = max(set(predictions), key=predictions.count)

                    avg_probabilities = sum(probabilities_list) / len(probabilities_list)
                    prob_dict = dict(zip(label_encoder.classes_, avg_probabilities))


                    with st.container():

                        st.success(f"Dominant Learning Style: **{dominant_style}**")
                        st.markdown("### Prediction Confidence:")
                        for style, prob in prob_dict.items():
                            st.markdown(f'<p class="confidence-text">{style}: {prob:.2%}</p>', unsafe_allow_html=True)

                        # Display Word Cloud

                        processed_text_all = ' '.join([preprocess_text(sentence) for sentence in sentences])
                        wordcloud = WordCloud(width=800, height=400,
                                             background_color='white',
                                             contour_color='#232D23', contour_width=2).generate(processed_text_all)
                        plt.figure(figsize=(12, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)



                        style_info = {
                            "Visual": {
                                "description": "Visual learners prefer learning through images, diagrams, videos, and written information.",
                                "tips": [
                                    "Use diagrams, charts, and infographics to organize information.",
                                    "Watch educational videos or tutorials to grasp concepts.",
                                    "Highlight or underline key points in texts with colors.",
                                    "Create mind maps to visualize connections between ideas."
                                ]
                            },
                            "Auditory": {
                                "description": "Auditory learners excel at learning through listening to lectures, discussions, or audio materials.",
                                "tips": [
                                    "Listen to podcasts or audiobooks related to your subject.",
                                    "Record lectures and replay them for review.",
                                    "Discuss topics with peers to reinforce understanding.",
                                    "Use mnemonic devices or rhymes to memorize information."
                                ]
                            },
                            "Kinesthetic": {
                                "description": "Kinesthetic learners learn best through hands-on activities, movement, and physical engagement.",
                                "tips": [
                                    "Engage in hands-on experiments or projects.",
                                    "Take frequent breaks to move around while studying.",
                                    "Use physical objects (e.g., models) to understand concepts.",
                                    "Practice role-playing or simulations to apply knowledge."
                                ]
                            }
                        }

                        st.markdown(f"### About {dominant_style} Learners")
                        st.write(style_info[dominant_style]["description"])

                        st.markdown("#### Tips for Effective Learning:")
                        for tip in style_info[dominant_style]["tips"]:
                            st.markdown(f'<div class="learning-tips">- {tip}</div>', unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No valid sentences were processed. Please ensure each sentence has at least 3 words.")

# Tab 2: Model Details
with tabs[1]:
    st.header("Model Details")
    st.markdown("### Model Details")
    st.write("""
    The VAK Learning Style Classifier uses a **Logistic Regression** model to predict whether a given sentence reflects a Visual, Auditory, or Kinesthetic learning style. Below are the details of the model and its implementation.
    """)

    st.markdown("#### Model Implementation")
    st.write("""
    - **Dataset**: The model was trained on a dataset (`dataset.csv`) containing sentences labeled as Visual, Auditory, or Kinesthetic learning styles.
    - **Text Preprocessing**:
      - Sentences are converted to lowercase.
      - Punctuation is removed.
      - Text is tokenized into words using NLTK's `word_tokenize`.
      - Stopwords (common words like 'the', 'is') are removed using NLTK's English stopwords list.
      - Words are lemmatized using NLTK's `WordNetLemmatizer` to reduce them to their base form (e.g., 'running' to 'run').
    - **Feature Extraction**: The preprocessed text is transformed into numerical features using a **TF-IDF Vectorizer** with a maximum of 1000 features, capturing the importance of words relative to the dataset.
    - **Label Encoding**: The learning styles (Visual, Auditory, Kinesthetic) are encoded into numerical labels using scikit-learn's `LabelEncoder`.
    - **Model Training**:
      - A **Logistic Regression** model was trained with a maximum of 1000 iterations and a random state of 42 for reproducibility.
      - The dataset was split into 80% training and 20% testing sets, with stratification to maintain class balance.
    - **Prediction**: The trained model predicts the learning style based on the TF-IDF features of the input sentence, and probabilities are computed to show confidence in each class.
    """)

    st.markdown("#### Model Performance")
    st.write(f"""
    The model achieved an **accuracy of {MODEL_ACCURACY:.2%}** on the test set, based on 3090 samples. Below is the classification report:
    """)

    st.markdown("""
    | Learning Style | Precision | Recall | F1-Score | Support |
    |----------------|-----------|--------|----------|---------|
    | Auditory       | 0.98      | 0.94   | 0.96     | 961     |
    | Kinesthetic    | 0.97      | 0.93   | 0.95     | 964     |
    | Visual         | 0.92      | 0.98   | 0.95     | 1165    |
    | **Macro Avg**  | 0.96      | 0.95   | 0.95     | 3090    |
    | **Weighted Avg** | 0.95    | 0.95   | 0.95     | 3090    |
    """)

    st.write("""
    - **Precision**: Indicates the proportion of correct predictions for each class.
    - **Recall**: Measures the proportion of actual instances correctly identified.
    - **F1-Score**: Balances precision and recall, providing a single metric for performance.
    - **Support**: Number of test samples for each class.
    The high precision, recall, and F1-scores (all around 0.95-0.96) demonstrate the model's robustness across all learning styles, with Visual having the highest recall (0.98) and Auditory the highest precision (0.98).
    """)
