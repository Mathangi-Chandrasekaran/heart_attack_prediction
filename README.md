## Heart Attack prediction
# Problem Description
This project is a classification problem aimed at predicting the likelihood of heart attack occurrences in humans based on various medical factors. Cardiovascular diseases are one of the leading causes of death globally, with an estimated 17.9 million lives lost each year. Around 80% of these deaths are due to heart attacks and strokes.

By predicting individuals at higher risk of heart attacks, timely intervention and treatment can be provided, potentially reducing the global mortality rate. The dataset used in this project contains several medical attributes, and the goal is to forecast the likelihood of a heart attack based on these factors.

# Data for Modeling
The dataset used for this project is stored in the heart.csv file, containing 303 samples and 14 attributes, including one target variable. These attributes represent various medical factors and their respective measurements.

# Attributes Description:
age: Age of the patient
sex: Sex of the patient (0: Female, 1: Male)
cp: Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal angina, 3: Asymptomatic)
trtbps: Resting blood pressure [mm Hg]
chol: Cholesterol level [mg/dl]
fbs: Fasting blood sugar (0: <= 120 mg/dl, 1: > 120 mg/dl)
restecg: Resting electrocardiogram results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricle hypertrophy)
thalachh: Maximum heart rate achieved
exng: Exercise-induced angina (1: Yes, 0: No)
oldpeak: ST depression induced by exercise
slp: Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
caa: Number of major vessels coloured by fluoroscopy (0-3)
thal: Thalassemia results (0: Null, 1: Fixed defect, 2: Normal, 3: Reversible defect)
output: Target variable indicating whether the patient has a higher chance of heart attack (1: Yes, 0: No)

# Data Preprocessing
The dataset was first loaded using Pandas and checked for null values and balanced class distribution. We performed normalization (standardization) to scale the input data and applied transformations to reduce skewness in certain features.

After preprocessing, the dataset was split into training and testing sets using train_test_split from scikit-learn.

# Algorithms Used
This project applies three different machine learning algorithms for classification to predict heart attack likelihood:

Support Vector Machine (SVM):
A supervised learning algorithm used for classification tasks. SVM finds the best hyperplane to separate the classes, maximizing the margin between them. It works well for small to medium-sized datasets with high-dimensional spaces.

Naive Bayes:
A probabilistic classifier based on Bayes' theorem. It assumes independence between features, which simplifies the classification process. The Gaussian Naive Bayes model is used in this project, which assumes that the input features follow a normal distribution.

Decision Tree:
A decision tree is a flowchart-like model that recursively splits the dataset based on the best attribute. Each internal node represents an attribute, and each leaf node represents the classification outcome. This algorithm is fast and easy to interpret.

# Installation & Setup
To run this project, ensure you have the following installed:
Python 3.x
Jupyter Notebook or Google Colaboratory 

Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn

You can install the required libraries using pip:
pip install pandas numpy scikit-learn matplotlib seaborn

Running the Code
Clone this repository to your local machine:
git clone https://github.com/your-username/heart-attack-prediction.git

Navigate to the project directory:
cd heart-attack-prediction

Open the Jupyter Notebook (or use Google Colaboratory) to run the code in the HeartAttackPrediction.ipynb file.
jupyter notebook heart_attack_prediction.ipynb

Run all the cells to see the results.

# Results & Evaluation
Each of the models (SVM, Naive Bayes, Decision Tree) was evaluated using accuracy, precision, recall, and F1-score. These metrics help us assess how well the models perform in predicting heart attack likelihood. The results are displayed in the notebook, with visualizations for comparison.

# Conclusion
This project demonstrates the use of machine learning algorithms to predict the likelihood of a heart attack based on medical factors. The predictions can assist healthcare providers in making early interventions for at-risk patients, which could reduce the global mortality rate from cardiovascular diseases.
