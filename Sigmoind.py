import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Loading the dataset as a pandas frame
dataFrame = pd.read_csv("preprocessed_ddos.csv")

# Printing 1st five rows
print(dataFrame.head().to_string())

# Separating the features and the labels
X = dataFrame.drop("Label", axis=1)
Y = dataFrame["Label"]

# Splitting the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Specifying the kernel to use ('sigmoid' in this case)
kernel = 'sigmoid'

# Creating and fitting the SVM model
model = SVC(kernel=kernel,C=100,gamma=0.01)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculating the performance metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
confusionMatrix = metrics.confusion_matrix(y_test, y_pred)

# Converting the Confusion Matrix into a Graph
cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1])
cmDisplay.plot()

# Setting the title
plt.title(f'Confusion Matrix for Kernel = {kernel}')
plt.show()

# Storing the performance metrics in a dictionary
results = [{
    'Kernel': kernel,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'True Negative': confusionMatrix[0][0],
    'True Positive': confusionMatrix[1][1],
    'False Positive': confusionMatrix[0][1],
    'False Negative': confusionMatrix[1][0],
}]

# Converting the results list into a DataFrame
resultsDataFrame = pd.DataFrame(results)

print("-----------------------------------------------------------------------------")
# Printing the DataFrame containing the results by converting into a string
print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format, header=True))

# Setting the x values for the accuracy bar
xValues = [kernel]
yValues = [accuracy]

# Plotting the accuracy bar
plt.bar(xValues, yValues)

# Setting up x and y labels
plt.xlabel("kernel")
plt.ylabel("Accuracy")

# Setting up the title
plt.title(f'Accuracy for Kernel = {kernel}')
plt.show()
