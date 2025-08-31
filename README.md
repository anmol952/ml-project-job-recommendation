<div align="center">
<h1> Job Recommendation System 🤖</h1>
<p>
A machine learning-based system to classify resumes into relevant job categories using NLP and classification algorithms.
</p>
</div>

🚀 Project Overview
This project is a machine learning-based job recommendation system that classifies resumes into different job categories. The system utilizes Natural Language Processing (NLP) techniques to process and understand the textual data from resumes and employs various classification algorithms to predict the most suitable job category. The primary models used for classification are K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and Random Forest Classifier. The final implementation uses the OneVsRestClassifier with SVC for prediction after finding it to be highly accurate.

📊 Dataset
The project uses the "UpdatedResumeDataSet.csv" dataset, which contains two columns:

Category: The job category for the resume.

Resume: The full text of the resume.

The dataset includes resumes from 25 different job categories such as Data Science, HR, Advocate, Web Designing, Mechanical Engineer, and more.

🛠️ Methodology
The project follows a standard machine learning pipeline:

1. 📂 Data Loading and Exploration
The dataset is loaded using the pandas library.

Initial exploration is done to understand the distribution of resumes across different categories using bar charts and pie charts.

2. 🧹 Data Preprocessing & Cleaning
Handling Imbalance: The initial dataset was imbalanced. To address this, oversampling was performed on the minority classes to match the number of samples in the majority class, creating a balanced dataset for training.

Text Cleaning: A function cleanResume is used to preprocess the resume text. This involves:

Removing URLs, RTs, CCs, hashtags, and mentions.

Eliminating punctuation and special characters.

Removing non-ASCII characters.

Normalizing whitespace.

3. 🔬 Feature Engineering
TF-IDF Vectorization: The cleaned resume text is converted into a numerical format using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This technique helps in highlighting words that are more important to a specific document within the corpus. English stop words are removed during this process.

4. 🧠 Model Training and Evaluation
Label Encoding: The categorical job titles are converted into numerical labels using LabelEncoder.

Train-Test Split: The dataset is split into training (80%) and testing (20%) sets.

Model Selection: The following classification models were implemented and evaluated:

K-Nearest Neighbors (wrapped in OneVsRestClassifier)

Support Vector Classifier (SVC) (wrapped in OneVsRestClassifier)

Random Forest Classifier (wrapped in OneVsRestClassifier)

Evaluation Metrics: The models' performance is evaluated based on:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

✨ Results
All three models performed exceptionally well on the test data after balancing the dataset. The SVC model was ultimately chosen for the final prediction function and saved for deployment.

Model

Accuracy

KNeighborsClassifier

99.54%

Support Vector Classifier (SVC)

99.77%

RandomForestClassifier

99.77%

The high accuracy indicates that the TF-IDF features are very effective for this classification task and that the models are able to correctly classify resumes into their respective job categories with high precision.

⚙️ How to Use
To use this project, you will need to have the saved models and vectorizers.

Dependencies
Python 3.x

pandas

numpy

scikit-learn

matplotlib

seaborn

re

Prediction
Load the saved TF-IDF vectorizer (tfidf.pkl), the label encoder (encoder.pkl), and the trained classifier (clf.pkl).

Use the pred function provided in the notebook to pass a new resume text.

The function will clean the text, transform it using the loaded vectorizer, and predict the job category using the trained SVC model.

import pickle

# Load the saved components
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
svc_model = pickle.load(open('clf.pkl', 'rb'))

# Example resume text
my_resume = """
[Paste a sample resume text here]
"""

# Get the prediction
predicted_category = pred(my_resume)
print(f"The predicted category for the resume is: {predicted_category}")





💾 Saved Files
tfidf.pkl: The saved TF-IDF vectorizer object.

clf.pkl: The trained Support Vector Classifier model.

encoder.pkl: The label encoder for the job categories.

👥 Contributors
<p align="center">
<a href="https://github.com/anmol952" target="_blank">
<img src="https://avatars.githubusercontent.com/u/185101209?v=4" width="100px;" alt="Anmol's Profile Picture" style="border-radius:50%">
<br>
<sub><b>Anmol</b></sub>
</a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/amitsinghrawat777" target="_blank">
<img src="https://avatars.githubusercontent.com/u/101490788?v=4" width="100px;" alt="Amit Rawat's Profile Picture" style="border-radius:50%">
<br>
<sub><b>Amit Rawat</b></sub>
</a>
</p>
