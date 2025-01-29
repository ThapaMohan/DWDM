import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load the dataset (Pima Indians Diabetes dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
           'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Prepare the input and output
X = data.iloc[:, :-1].values  # Features (all except 'Outcome')
Y = data.iloc[:, -1].values   # Target ('Outcome')

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, Y_train)

# Model accuracy
result = svm_classifier.score(X_test, Y_test)
print("The SVM model has given Accuracy of: %.3f%%" % (result * 100.0))

# Make predictions on the test set
Y_pred = svm_classifier.predict(X_test)
print("Predictions:", Y_pred)

# Confusion Matrix
cf_mtrx = metrics.confusion_matrix(Y_test, Y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cf_mtrx, display_labels=['Negative', 'Positive'])
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(Y_test, Y_pred, target_names=['Negative', 'Positive']))
