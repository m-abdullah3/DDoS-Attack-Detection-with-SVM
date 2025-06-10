import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_ddos.csv")

#Printing 1st five rows
print(dataFrame.head().to_string())

#seperating the features and the labels
X=dataFrame.drop("Label",axis=1)
Y=dataFrame["Label"]
#Splitting the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

#list stroing kernels
kernel= ["poly","linear","sigmoid"]
# a list to store the performance metrics
results=[]
for i in kernel:

    #Creating the SVM Model with a certain Kernel
    model = SVC(kernel=i)
    #fitting the svm model
    model.fit(X_train, y_train)
    #Making predication based on training data
    y_pred = model.predict(X_test)

    # Calculaing the performance metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # zero_divsion = 0 to handle divisions with zero as the denominator
    precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)

    # Converting the Confusion Matrix into a Graph
    cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0,1])

    # Calculting the True Positives etc. using the Confusion Matrix
    trueNegative, falsePositive, falseNegative, truePositive = confusionMatrix.ravel()

    # Adding the performnace metrics to the results list in the form of a dictionary
    results.append({
        'Kernel': i,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'True Negative': trueNegative,
        'True Positive': truePositive,
        'False Positive': falsePositive,
        'False Negative': falseNegative,
    })

    # Plotting the confsion matrix
    cmDisplay.plot()

    # setting the title
    plt.title(f'Confusion Matrix for Kernel = {i}')
    plt.show()

# converting the results list into a dataframe
resultsDataFrame = pd.DataFrame(results)

print("-----------------------------------------------------------------------------")
# Printing the dataframe containing the results by converting into a string
# The parameters set allow the dataframe to be fully printed
print(
    resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format,
                               header=True))

# setting the x values for the accuracy bar
xValues = kernel

yValues = []

# getting the accuracies from the results list to act as y values
for i in results:
    yValues.append(i.get("Accuracy"))

# PLotting the accuracy bar
plt.bar(xValues, yValues)

# setting up x and y labels
plt.xlabel("Kernel")
plt.ylabel("Accuracy")

# setting up the title
plt.title("Accuracy Comparsion Between different Kernels")
plt.show()