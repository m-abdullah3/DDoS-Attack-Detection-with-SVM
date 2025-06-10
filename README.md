# DDoS-Attack-Detection-with-SVM
This project focuses on detecting Distributed Denial of Service (DDoS) attacks using Support Vector Machine (SVM) models with different kernel functions (Linear, Polynomial, and Sigmoid). The dataset is preprocessed to remove inconsistencies, and the models are evaluated using performance metrics such as accuracy, precision, recall, F1-score, and confusion matrices to compare their effectiveness.

Features
Data Preprocessing: Cleans and prepares the DDoS dataset by handling missing values, removing duplicates, encoding categorical variables, and scaling features using Min-Max scaling.
SVM Model Training: Implements SVM models with Linear, Polynomial, and Sigmoid kernels, including hyperparameter tuning (e.g., polynomial degree, gamma, and C values).
Performance Evaluation: Analyzes model performance using confusion matrices, accuracy, precision, recall, and F1-score metrics, with visualizations for clear insights.
Visualization: Includes histograms of dataset features and confusion matrix plots to aid in understanding model performance.
Project Structure

DDoS-Attack-Detection-SVM/

├── Data_Cleaning.py          # Initial data preprocessing and exploration

├── PreprocessingData.py      # Final preprocessing script to generate a cleaned dataset

├── Tunning.py                # Hyperparamter Tuning

├── linear.py                 # SVM model with Linear kernel

├── Poly.py                   # SVM model with Polynomial kernel

├── Sigmoind.py               # SVM model with Sigmoid kernel

├── Training_SVM.py           # All three models combined for comparison

├── preprocessed_ddos.csv     # Output dataset after preprocessing

├── README.md                 # Project documentation

Installation & Setup Instructions

Set Up a Python Environment:
Ensure Python 3.8+ is installed.
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
Install required libraries using pip:

pip install pandas scikit-learn matplotlib
Prepare the Dataset:
Place the ddos_dataset.csv file in the project directory.
Run the preprocessing script to generate the cleaned dataset:

python PreprocessingData.py
Run the SVM Models:
Execute each model script to train and evaluate:

python linear.py
python Poly.py
python Sigmoind.py
Tech Stack / Libraries Used
Python: Core programming language.
Pandas: Data manipulation and preprocessing.
Scikit-learn: SVM model implementation and performance metrics.
Matplotlib: Visualization of histograms and confusion matrices.
License
This project is licensed under the MIT License - see the LICENSE file for details.
