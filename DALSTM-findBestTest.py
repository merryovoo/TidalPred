from datetime import datetime, timedelta, time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import os
import time
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, pyll
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping,Callback
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 检查 GPU 是否可用
print("TensorFlow 版本：", tf.__version__)
print("是否支持 GPU：", tf.config.list_physical_devices('GPU'))

# 设置 TensorFlow 使用 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("使用 GPU：", gpus[0])
    except RuntimeError as e:
        print(e)


# 数据展示
def custom_parser(x):
    return pd.to_datetime(x, format='%Y%m%d%H', errors='coerce')


dataFilePath = 'DALSTM-findBestTest-testData.csv'
data = pd.read_csv(dataFilePath, index_col=0, parse_dates=['Date'])
if data is None:
    raise ValueError(f"数据读取失败，请检查文件路径：{dataFilePath}")
print(data.head())
print('数据长度是：', len(data))
print(type(data))

# 创建文件夹路径
folder_path = 'E:/PythonJieGuo/DALSTM_Results'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 数据归一化
def normalize_dataframe(df):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return normalized_data, scaler


# 准备数据
def prepare_data(data, win_size, target_feature_idx):
    num_features = data.shape[1]
    X = []
    y = []
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size, :]
        temp_y = data[i + win_size, target_feature_idx]
        X.append(temp_x)
        y.append(temp_y)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


# 建立模型
def build_model(params, input_shape):
    model = Sequential()
    layers_config = params['layers_config']
    num_layers = layers_config['num_layers']
    lstm_units = layers_config['lstm_units']

    assert len(lstm_units) == num_layers, "lstm_units长度必须等于num_layers"

    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(
                units=lstm_units[i],
                activation=params['activation'],
                input_shape=input_shape,
                kernel_regularizer=l2(params['l2_reg']),
                return_sequences=(i < num_layers - 1)
            ))
        else:
            model.add(LSTM(
                units=lstm_units[i],
                activation=params['activation'],
                kernel_regularizer=l2(params['l2_reg']),
                return_sequences=(i < num_layers - 1)
            ))
    model.add(Dense(1))

    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model


# 早停回调函数
class CustomEarlyStopping(Callback):
    def __init__(self, patience=0, pq_alpha_threshold=10, progress_period=20, calc_period=5):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.pq_alpha_threshold = pq_alpha_threshold
        self.progress_period = progress_period
        self.calc_period = calc_period
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
        self.train_losses = []
        self.val_losses = []
        self.actual_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_train_loss = logs.get('loss')
        self.train_losses.append(current_train_loss)
        self.val_losses.append(current_val_loss)

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if epoch % self.calc_period == 0 and epoch >= self.progress_period:
            P_k_t = 1000 * (sum(self.train_losses[epoch - self.progress_period:epoch]) / (
                    self.progress_period * min(self.train_losses[epoch - self.progress_period:epoch])) - 1)
            GL_t = 100 * (current_val_loss / min(self.val_losses[:epoch + 1]) - 1)
            PQ_alpha_t = GL_t / P_k_t if P_k_t != 0 else float('inf')
            print(f"Epoch {epoch}: P_k_t = {P_k_t}, GL_t = {GL_t}, PQ_alpha_t = {PQ_alpha_t}")
            if PQ_alpha_t > self.pq_alpha_threshold:
                print(f"Early stopping at epoch {epoch} due to PQ_alpha > {self.pq_alpha_threshold}")
                self.model.stop_training = True
                self.actual_epochs = epoch + 1
                return

        if self.no_improvement_count >= self.patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in validation loss")
            self.model.stop_training = True
            self.actual_epochs = epoch + 1


# 评价标准
def evaluation_criteria(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mse, rmse, mae, r2


# 贝叶斯优化目标函数
def objective_function(params, train_x, train_y, val_x, val_y, win_size, target_feature_idx):
    try:
        model = build_model(params, (train_x.shape[1], train_x.shape[2]))
        custom_early_stopping = CustomEarlyStopping(patience=10, pq_alpha_threshold=10, progress_period=20)

        history = model.fit(
            train_x, train_y,
            epochs=300,
            batch_size=int(params['batch_size']),
            verbose=0,
            validation_data=(val_x, val_y),
            callbacks=[custom_early_stopping]
        )

        val_loss = model.evaluate(val_x, val_y, verbose=0)[0]
        del model
        return {'loss': val_loss, 'status': STATUS_OK}
    except Exception as e:
        print(f"模型训练失败: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK}


# 确定最优预测长度
def determine_optimal_prediction_length(initial_train_data, complete_dataset, max_n, val_ratio, best_hparams, win_size,
                                        target_feature_idx):
    min_error = float('inf')
    best_n = 1

    for n in range(1, max_n + 1):
        print(f"测试预测长度: {n}")

        # 准备训练数据
        train_data_normalized, scaler = normalize_dataframe(initial_train_data)
        train_x, train_y = prepare_data(train_data_normalized.values, win_size, target_feature_idx)

        # 构建并训练模型
        model = build_model(best_hparams, (train_x.shape[1], train_x.shape[2]))
        model.fit(train_x, train_y, epochs=30, batch_size=best_hparams['batch_size'], verbose=0)

        # 预测接下来的n天数据
        predictions = []
        current_input = train_x[-1:].copy()

        for _ in range(n):
            pred = model.predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)

            # 更新输入序列
            new_input = np.roll(current_input[0], -1, axis=0)
            new_input[-1, :] = 0  # 用预测值更新最后一个时间步
            new_input[-1, target_feature_idx] = pred
            current_input = new_input.reshape(1, win_size, train_x.shape[2])

        # 获取真实值
        start_idx = len(initial_train_data)
        end_idx = start_idx + n
        if end_idx > len(complete_dataset):
            break

        true_values = complete_dataset.iloc[start_idx:end_idx, target_feature_idx].values

        # 反归一化进行比较
        if len(predictions) == len(true_values):
            pred_array = np.array(predictions).reshape(-1, 1)
            true_array = true_values.reshape(-1, 1)

            # 计算误差
            mse, rmse, mae, r2 = evaluation_criteria(true_array, pred_array)
            error = mse  # 使用MSE作为误差标准

            print(f"预测长度 {n}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            if error < min_error:
                min_error = error
                best_n = n

    print(f"最优预测长度: {best_n}, 最小误差: {min_error:.4f}")
    return best_n


# 滑动窗口迭代预测
def sliding_window_prediction(complete_dataset, initial_train_data, window_size, val_ratio, param_space, max_iterations,
                              win_size, target_feature_idx):
    current_window_data = initial_train_data.copy()
    all_predictions = []
    iteration_count = 0

    while len(current_window_data) + window_size <= len(complete_dataset):
        iteration_count += 1
        print(f"\n=== 滑动窗口迭代 {iteration_count} ===")
        print(f"当前窗口数据长度: {len(current_window_data)}")

        # 归一化当前窗口数据
        current_data_normalized, scaler = normalize_dataframe(current_window_data)

        # 准备训练和验证数据
        X, y = prepare_data(current_data_normalized.values, win_size, target_feature_idx)

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - val_ratio))
        train_x, train_y = X[:split_idx], y[:split_idx]
        val_x, val_y = X[split_idx:], y[split_idx:]

        if len(train_x) == 0 or len(val_x) == 0:
            print("数据不足，停止迭代")
            break

        # 贝叶斯优化超参数
        print("进行贝叶斯优化...")
        trials = Trials()

        def current_objective(params):
            return objective_function(params, train_x, train_y, val_x, val_y, win_size, target_feature_idx)

        best = fmin(
            fn=current_objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=min(max_iterations, 10),  # 限制评估次数以提高效率
            trials=trials,
            verbose=0
        )

        best_hparams_current = get_best_params(best)
        print(f"当前窗口最优超参数: {best_hparams_current}")

        # 使用最优超参数训练模型
        current_model = build_model(best_hparams_current, (train_x.shape[1], train_x.shape[2]))
        current_model.fit(
            train_x, train_y,
            epochs=300,
            batch_size=best_hparams_current['batch_size'],
            validation_data=(val_x, val_y),
            verbose=0
        )

        # 预测下一个窗口
        future_predictions = []
        current_input = X[-1:].copy()

        for _ in range(window_size):
            pred = current_model.predict(current_input, verbose=0)[0, 0]
            future_predictions.append(pred)

            # 更新输入序列
            new_input = np.roll(current_input[0], -1, axis=0)
            new_input[-1, :] = 0
            new_input[-1, target_feature_idx] = pred
            current_input = new_input.reshape(1, win_size, X.shape[2])

        # 获取真实值
        start_idx = len(current_window_data)
        end_idx = start_idx + window_size
        true_values = complete_dataset.iloc[start_idx:end_idx, target_feature_idx].values

        # 计算评价指标
        pred_array = np.array(future_predictions).reshape(-1, 1)
        true_array = true_values.reshape(-1, 1)

        mse, rmse, mae, r2 = evaluation_criteria(true_array, pred_array)

        print(f"预测结果 - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # 保存结果
        result = {
            'iteration': iteration_count,
            'predictions': future_predictions,
            'true_values': true_values.tolist(),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'window_size': len(current_window_data)
        }
        all_predictions.append(result)

        # 检查是否满足更新条件
        if mse < 0.1 and rmse < 0.1 and mae < 0.1 and r2 > 0.9:
            print("满足更新条件，更新滑动窗口...")
            # 移除最旧的window_size个数据，添加新的window_size个数据
            current_window_data = current_window_data.iloc[window_size:]
            new_data = complete_dataset.iloc[start_idx:end_idx]
            current_window_data = pd.concat([current_window_data, new_data])
        else:
            print("不满足更新条件，停止迭代")
            break

    return all_predictions


# 获取最佳参数
def get_best_params(best):
    layers_choice_idx = best['layers_config']

    if layers_choice_idx == 0:  # 2层
        units = [
            [16, 32, 64, 128, 256][best['units_layer21']],
            [16, 32, 64, 128, 256][best['units_layer22']]
        ]
        num_layers = 2
    elif layers_choice_idx == 1:  # 3层
        units = [
            [16, 32, 64, 128, 256][best['units_layer31']],
            [16, 32, 64, 128, 256][best['units_layer32']],
            [16, 32, 64, 128, 256][best['units_layer33']]
        ]
        num_layers = 3
    else:  # 1层
        units = [
            [16, 32, 64, 128, 256][best['units_layer11']],
        ]
        num_layers = 1

    return {
        'layers_config': {
            'num_layers': num_layers,
            'lstm_units': units
        },
        'learning_rate': np.exp(best['learning_rate']),
        'batch_size': [32, 64, 128][best['batch_size']],
        'l2_reg': np.exp(best['l2_reg']),
        'activation': ['sigmoid', 'tanh', 'relu'][best['activation']],
        'optimizer': ['adam', 'sgd', 'rmsprop'][best['optimizer']]
    }


# 主函数 - DALSTM模型实现
def DALSTM_model(complete_dataset, max_n=7, val_ratio=0.2, max_iterations=10, win_size=24, target_feature_idx=0):
    """
    DALSTM模型主函数

    参数:
    complete_dataset: 完整数据集
    max_n: 最大候选预测天数
    val_ratio: 验证集比例
    max_iterations: 最大迭代次数
    win_size: 窗口大小
    target_feature_idx: 目标特征索引
    """

    print("=== DALSTM模型开始运行 ===")

    # 1. 选择初始训练段
    initial_train_days = 7  # 初始7天数据
    initial_train_size = initial_train_days * 1440  # 每天1440个数据点
    initial_train_data = complete_dataset.iloc[:initial_train_size]

    print(f"初始训练集长度: {len(initial_train_data)} (相当于{initial_train_days}天)")

    # 2. 定义超参数空间
    param_space = {
        'layers_config': hp.choice('layers_config', [
            {
                'num_layers': 2,
                'lstm_units': [
                    hp.choice('units_layer21', [16, 32, 64, 128, 256]),
                    hp.choice('units_layer22', [16, 32, 64, 128, 256])
                ]
            },
            {
                'num_layers': 3,
                'lstm_units': [
                    hp.choice('units_layer31', [16, 32, 64, 128, 256]),
                    hp.choice('units_layer32', [16, 32, 64, 128, 256]),
                    hp.choice('units_layer33', [16, 32, 64, 128, 256])
                ]
            },
            {
                'num_layers': 1,
                'lstm_units': [
                    hp.choice('units_layer11', [16, 32, 64, 128, 256]),
                ]
            },
        ]),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
        'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'activation': hp.choice('activation', ['sigmoid', 'tanh', 'relu']),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    }

    # 3. 初始贝叶斯优化寻找最优超参数
    print("=== 初始贝叶斯优化 ===")
    initial_data_normalized, scaler = normalize_dataframe(initial_train_data)
    train_x, train_y = prepare_data(initial_data_normalized.values, win_size, target_feature_idx)

    # 划分训练集和验证集
    split_idx = int(len(train_x) * (1 - val_ratio))
    train_x_split, train_y_split = train_x[:split_idx], train_y[:split_idx]
    val_x_split, val_y_split = train_x[split_idx:], train_y[split_idx:]

    trials = Trials()

    def initial_objective(params):
        return objective_function(params, train_x_split, train_y_split, val_x_split, val_y_split, win_size,
                                  target_feature_idx)

    best_initial = fmin(
        fn=initial_objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=min(max_iterations, 100),
        trials=trials,
        verbose=0
    )

    best_hparams = get_best_params(best_initial)
    print(f"初始最优超参数: {best_hparams}")

    # 4. 确定最优预测长度
    print("=== 确定最优预测长度 ===")
    optimal_n = determine_optimal_prediction_length(
        initial_train_data, complete_dataset, max_n, val_ratio,
        best_hparams, win_size, target_feature_idx
    )

    window_size = optimal_n
    print(f"最终确定的滑动窗口尺寸: {window_size}")

    # 5. 滑动窗口迭代预测
    print("=== 开始滑动窗口迭代预测 ===")
    results = sliding_window_prediction(
        complete_dataset, initial_train_data, window_size, val_ratio,
        param_space, max_iterations, win_size, target_feature_idx
    )

    print("=== DALSTM模型运行完成 ===")
    return results, best_hparams, window_size


# 运行DALSTM模型
if __name__ == "__main__":
    # 参数设置
    max_n = 7  # 最大候选预测天数
    val_ratio = 0.2  # 验证集比例
    max_iterations = 10  # 最大迭代次数
    win_size = 24  # 窗口大小
    target_feature_idx = 0  # 目标特征索引

    # 运行模型
    results, best_params, optimal_window_size = DALSTM_model(
        data, max_n, val_ratio, max_iterations, win_size, target_feature_idx
    )

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(folder_path, 'DALSTM_results.csv'), index=False)

    # 保存最佳参数
    with open(os.path.join(folder_path, 'best_parameters.txt'), 'w') as f:
        f.write("最佳超参数:\n")
        f.write(str(best_params) + "\n\n")
        f.write(f"最优窗口大小: {optimal_window_size}\n")
        f.write(f"总迭代次数: {len(results)}\n")

    print(f"结果已保存到: {folder_path}")
    print(f"总迭代次数: {len(results)}")
    print(f"最优窗口大小: {optimal_window_size}")