# 🎵 Music Recommendation with Machine Learning (Python)

This project is a beginner-level machine learning task where a model recommends music genres based on a user's age and gender using the **Decision Tree Classifier** algorithm. Along with that, this README includes practical notes from a short ML course.

---

## 🛠 Tools & Setup

- Python (Recommended via [Anaconda](https://www.anaconda.com/))
- Jupyter Notebook
- `pandas`, `scikit-learn`, `joblib`
- Dataset from [Kaggle](https://www.kaggle.com/)

---

## 📓 How to Start Jupyter Notebook

1. Open **Anaconda Prompt**
2. Run:
   ```bash
   jupyter notebook
A browser window will open at localhost

Create a new notebook

⬇️ Getting the Dataset
Visit: https://bit.ly/music-csv

Download music.csv and place it in your working directory

🔢 Jupyter Notebook Shortcuts
Action	Shortcut
Delete cell	Press D twice
New cell below	B
New cell above	A
Show all shortcuts	Ctrl + Shift + H
Method suggestions	TAB after dot
Show docstring	Shift + TAB on method

🧪 Step-by-Step Machine Learning Project
1. 📥 Import the Data

import pandas as pd

music_data = pd.read_csv('music.csv')
music_data
2. 🧹 Clean the Data
For this dataset, there’s no missing or empty field — so no cleaning needed.

3. 📂 Split Data into Input & Output

X = music_data.drop(columns=['genre'])  # Input (age, gender)
y = music_data['genre']                 # Output (genre)
4. 🌲 Create the Model

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
5. 🏋️‍♂️ Train the Model

model.fit(X, y)
6. 🔮 Make Predictions

predictions = model.predict([[21, 1], [22, 0]])
print(predictions)  # Output: ['HipHop' 'Dance']
📏 Measure Accuracy of Model
To split data into training and testing sets:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)
test_size=0.2 means 20% data used for testing

accuracy_score() shows how well model performs

💾 Save & Load Trained Model
To persist the model (so you don’t have to retrain every time):

Save the model:

import joblib

joblib.dump(model, 'music-recommender.joblib')
Load and use it later:

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1], [22, 0]])
print(predictions)
🌳 Visualize the Decision Tree
Export tree structure to a .dot file:

from sklearn import tree

tree.export_graphviz(model,
                     out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
Then you can convert the .dot file to an image using tools like Graphviz.

📈 What's Next?
Here’s what you can do after this project:

✅ Try other algorithms like KNeighborsClassifier or RandomForestClassifier

✅ Learn about data preprocessing and handling null/missing values

✅ Use real-world datasets (from Kaggle or UCI)

✅ Explore evaluation metrics: precision, recall, F1-score

✅ Learn about model deployment (e.g. with Flask or Streamlit)

✅ Understand overfitting and cross-validation

📚 Resources for You
Sklearn Documentation

Pandas Documentation

Kaggle Datasets

Graphviz for Visualizing Models

🙋 About
This project and notes were created as part of a short ML with Python course — ideal for beginners looking to get hands-on with real ML workflows.
