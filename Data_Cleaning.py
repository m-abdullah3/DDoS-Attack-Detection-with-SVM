import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Loading the dataset as a pandas frame
dataFrame= pd.read_csv("ddos_dataset.csv")
print ("shape = ",dataFrame.shape)

#Printing 1st five rows using head()
print("Head: ")
print(dataFrame.head().to_string())
print("------------------------------------------------")

#Printing last five rows using tail()
print("Tail")
print(dataFrame.tail().to_string())
print("------------------------------------------------")



# Convert 'Timestamp' column to datetime format, allowing pandas to infer the format and using dayfirst=True
dataFrame['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'], dayfirst=True, errors='coerce')
# Extract the hour of the day
dataFrame['hour_of_day'] = dataFrame['Timestamp'].dt.hour
print("------------------------------------------------")
dataFrame.drop(["Timestamp"], axis=1,inplace=True)

print(dataFrame.head().to_string())


#Finding Missing Values
print("Missing Values")
print(dataFrame.isnull().sum())
print("------------------------------------------------")

#Finding Duplicate values
print("number of Duplicates = ",dataFrame.duplicated().sum())

#Removing duplicates
dataFrame.drop_duplicates(inplace=True)
print("Removing Duplictes..........")
print("number of Duplicates = ", dataFrame.duplicated().sum())

print("------------------------------------------------")

#getting statistics about the data using describe
print("Describe()")
print(dataFrame.describe().to_string())

print("------------------------------------------------")

#Getting information about the dataframe using info()
print("Info()")
print(dataFrame.info())

print("------------------------------------------------")

#Checking count of unique labels
print("Count of Labels")
labelCount=dataFrame["Label"].value_counts()

#printing the result
print(labelCount)


print(dataFrame.head().to_string())

labelEncoder=preprocessing.LabelEncoder()
dataFrame["Label"]=labelEncoder.fit_transform(dataFrame["Label"])
dataFrame["Timestamp"]=labelEncoder.fit_transform(dataFrame["Timestamp"])
plt.hist(dataFrame, bins=20, edgecolor='black')
plt.title("Histogram")
plt.show()


featuresScaler=preprocessing.MinMaxScaler()
features=dataFrame.drop("Label", axis=1).columns

scaledFeatures = featuresScaler.fit_transform(dataFrame[features])
scaledDataframe = pd.DataFrame(scaledFeatures, columns=features)

# Ensure the label column is correctly aligned
scaledDataframe["Label"] = dataFrame["Label"].values

print("------------------------------------------------")
print('Cleaned Dataset')
print(scaledDataframe.head().to_string())
print(scaledDataframe.isnull().sum())
print("dups = ",scaledDataframe.duplicated().sum())



#cleanedDataset ="cleaned_ddos.csv"
#scaledDataframe.to_csv(cleanedDataset,index=False)

