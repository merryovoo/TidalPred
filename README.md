# **Project Introduction**

This project comprises three core scripts along with their corresponding test datasets, with each component serving the following functions:
1.LSTM-withBO.py: This script employs Bayesian Optimization to automatically search for the optimal hyperparameter combination of the LSTM model based on the current training set. After determining the best hyperparameters, it constructs and trains the model, then proceeds to predict subsequent data. The accompanying test data, LSTM-withBO-testData.csv, is used to validate the completeness and effectiveness of this pipeline.
2.DALSTM-findBestTrain.py: Designed to optimize the input length during the model training phase. This script systematically evaluates model prediction performance (using metrics such as RMSE and MAE) and computational efficiency across different training set lengths, automatically determining the optimal training window size to achieve a balance between accuracy and efficiency. The accompanying test data is DALSTM-findBestTrain-testData.
3.DALSTM-findBestTest.py: Addressing the length optimization in the prediction phase, this script introduces a sliding window mechanism combined with Bayesian Optimization to dynamically search for the optimal prediction step length under the current trained model. This strategy helps enhance the model's adaptability and prediction stability in practical applications. The accompanying test data is DALSTM-findBestTest-testData.csv.
These three scripts respectively correspond to three critical aspects: model hyperparameter optimization, training length selection, and prediction length optimization. Together, they form a comprehensive dynamic adaptive LSTM modeling and prediction framework.

# **Model Usage Guide**

This document provides a detailed introduction to the usage of three core Python scripts:
LSTM-withBO.py - an LSTM model based on Bayesian optimization; DALSTM-findBestTest.py - a DALSTM model (exploring the optimal prediction length); DALSTM-findBestTrain.py - a DALSTM model (analyzing the optimal training set length).
It is recommended to execute the scripts in the following order: First, run LSTM-withBO.py; Then, execute DALSTM-findBestTest.py; Finally, use DALSTM-findBestTrain.py.
Through this guide, you will learn how to properly configure parameters, run the programs, and interpret the experimental results.

**1.LSTM-withBO.py Usage Instructions**

**1.1 Parameter Configuration Instructions*s*
**1.1.1 Data Path Configuration (Required)**
Modification Location: Line 40
Description: Replace the default data path with the path to your actual dataset.
·Using Test Data: You can directly set it as dataFilePath = 'LSTM-withBO-testData.csv'. Ensure this test data file is placed in the same directory as the program file for correct reading.
·Using Custom Data: Ensure the path is accurate (supports both absolute and relative paths).
**1.1.2 Training Set Length Setting (Adjust as Needed)**
Modification Location: Line 48 and Line 327 (These two parameters must be consistent).
Description: Each run supports testing only a single training set length. To test multiple lengths, run the program in separate batches:
·First run: Set both parameters to 7 (7-day training set).
·Second run: Reopen the file and set both parameters to 14.
·Third run: Set both parameters to 28.
Note: The Bayesian optimization process is computationally intensive and does not currently support batch execution for multiple training set lengths.
**1.1.3 Test Data Range Adjustment (Adjust as Needed)**
Modification Location: Line 325
Format: range(start_day, end_day + 1, step)
Examples:
·Predict days 1-28: range(1, 29, 1)
·Predict days 3-10: range(3, 11, 1)
·Predict every 2 days: range(1, 29, 2)
**1.1.4 Output Path and Filename Configuration (Adjust as Needed)**
Results Folder Path: Line 64 and Line 340 (Ensure consistency).
Metrics Results Filename: Line 468
Example:folder_path = f'E:/PythonJieGuo/yaoan/{x}-{v}-{y}';
df1.to_csv('metics.csv', mode='a', header=False, index=False)
**1.2 Execution Process**
After completing the parameter configuration, run the script directly. The program will automatically execute the following steps:
Data Splitting → Data Normalization → Bayesian Hyperparameter Optimization → Model Training → Test Set Prediction → Result Saving
**1.3 Output Results Description**
**1.3.1 Full Process Results Folder**
Content: Complete data for the test set corresponding to each training set.
Includes: Raw data, normalized data, model prediction results, etc.
Purpose: Supports plotting analysis, data backtracking, and in-depth validation.
**1.3.2 Metrics Summary CSV File**
Content: Key evaluation metrics (MAE, RMSE, R², etc.).
Purpose: Enables quick comparison of model performance and filtering of optimal parameters.
**2.DALSTM-findBestTest.py Usage Instructions
2.1 Model Background**
To overcome the limitations of traditional LSTM models in hyperparameter optimization and data adaptation, this script implements the DALSTM model, which features optimal prediction length exploration and sliding window iterative prediction capabilities.
**2.2 Parameter Configuration
2.2.1 Data File Path Configuration (Required)**
Modification Location: Line 40
Description: Set the dataFilePath parameter to the actual data path.
·Test Data: dataFilePath = 'DALSTM-findBestTest-testData.csv'
·Custom Data: Ensure the path is accurate.
**2.2.2 Results Folder Path Configuration (Adjust as Needed)**
Modification Location: Line 49
Description: Set the results save path via the folder_path parameter.
Example: folder_path = 'E:/PythonJieGuo/DALSTM_Results'
**2.2.3 Initial Training Segment Setting (Adjust as Needed)**
Modification Location: Line 411
Description: Set the number of days for the initial training set via the initial_train_days parameter.
·Default Value: 7 days
·User Guidance: Users should adjust this based on the research objective (e.g., short-term/long-term prediction) and data characteristics (e.g., data time span, sampling frequency). For instance, for monthly predictions, the initial training segment could be set to 30 days.
**2.3 Execution Process**
Data Loading & Preprocessing → Initial Training Set Construction → Optimal Prediction Length Exploration → Sliding Window Iterative Prediction → Result Generation
**2.4 Output Results Description
2.4.1 Hyperparameter Record TXT File**
Content: Optimal hyperparameter combinations for each training session.
Includes: Learning rate, number of hidden units, iteration count, etc.
Purpose: Parameter tracking and reference for model tuning.
**2.4.2 Metrics Summary CSV File**
Content: Key evaluation metrics (MSE, RMSE, MAE, R²).
Purpose: Quantifies model performance and assesses stability and applicable scenarios.
**3.DALSTM-findBestTrain.py Usage Instructions**
**3.1 Parameter Configuration**
**3.1.1 Data File Path Configuration (Required)**
Modification Location: Line 186
Description: Set csv_file_path to the CSV file containing experimental results.
·Test Data: csv_file_path = 'DALSTM-findBestTrain-testData.csv' 
·Data Source: Generated after running LSTM-withBO.py.
**3.1.2 Target Prediction Length Setting (Required)**
Modification Location: Line 194
Description: Set the analysis target via the target_pred_length parameter.
Example: To find the optimal training set length for a 28-day prediction set → target_pred_length = 28.
**3.2 Execution Process**
Run the program directly after configuration. It will automatically analyze the data and output recommended results.
**3.3 Output Results**
Console Output: Recommended optimal training set length.
Example: "The optimal training set length for the 28-day prediction set is: 14 days."
**4.General Considerations**
①File Paths: Avoid using Chinese characters or special symbols (e.g., spaces, #, &) in all paths (data paths, output paths) to prevent encoding-related errors.
②Dependencies: Ensure the Python environment has all required libraries installed (e.g., TensorFlow, scikit-learn, numpy, pandas). It is recommended to install dependencies in advance using requirements.txt or command-line tools.
③Runtime Considerations:Bayesian optimization and optimal prediction length exploration are computationally intensive.Execution time depends on data volume, parameter ranges, and hardware performance.Run these processes during idle periods to avoid interruptions.
④Reproducibility: For repeated experiments, maintain consistency in parameter configurations and filenames to facilitate comparison of results across different batches and ensure stability.
