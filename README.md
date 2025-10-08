This project comprises three core scripts along with their corresponding test datasets, with each component serving the following functions:

1.LSTM-withBO.py
This script employs Bayesian Optimization to automatically search for the optimal hyperparameter combination of the LSTM model based on the current training set. After determining the best hyperparameters, it constructs and trains the model, then proceeds to predict subsequent data. The accompanying test data, LSTM-withBO-testData.csv, is used to validate the completeness and effectiveness of this pipeline.

2.DALSTM-findBestTrain.py
Designed to optimize the input length during the model training phase. This script systematically evaluates model prediction performance (using metrics such as RMSE and MAE) and computational efficiency across different training set lengths, automatically determining the optimal training window size to achieve a balance between accuracy and efficiency. The accompanying test data is DALSTM-findBestTrain-testData.

3.DALSTM-findBestTest.py
Addressing the length optimization in the prediction phase, this script introduces a sliding window mechanism combined with Bayesian Optimization to dynamically search for the optimal prediction step length under the current trained model. This strategy helps enhance the model's adaptability and prediction stability in practical applications. The accompanying test data is DALSTM-findBestTest-testData.csv.

These three scripts respectively correspond to three critical aspects: model hyperparameter optimization, training length selection, and prediction length optimization. Together, they form a comprehensive dynamic adaptive LSTM modeling and prediction framework.
