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
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

# 检查 GPU 是否可用
print("TensorFlow 版本：", tf.__version__)
print("是否支持 GPU：", tf.config.list_physical_devices('GPU'))

# 设置 TensorFlow 使用 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 TensorFlow 使用第一个 GPU，并允许动态分配内存
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("使用 GPU：", gpus[0])
    except RuntimeError as e:
        print(e)


# 数据展示
def custom_parser(x):
    return pd.to_datetime(x, format='%Y%m%d%H', errors='coerce')


dataFilePath = '姚安水位201207060000预处理.csv'
data = pd.read_csv(dataFilePath, index_col=0, parse_dates=['Date'])
if data is None:
    raise ValueError(f"数据读取失败，请检查文件路径：{dataFilePath}")
print(data.head())
print('数据长度是：', len(data))
print(type(data))

trainNum = 28 * 1440
valNum = trainNum + round(trainNum / 4)
testNum = valNum + 1 * 1440

df_train = data.iloc[:trainNum]
df_val = data.iloc[trainNum:valNum]
df_test = data.iloc[valNum:testNum]

x = int(len(df_train) / 1440)
print(f'训练集的长度是：{len(df_train)},相当于{x}天')

v = np.round(len(df_val) / 1440, 1)
print(f'验证集的长度是：{len(df_val)},相当于{v}天')

y = int(len(df_test) / 1440)
print(f'测试集的长度是：{len(df_test)},相当于{y}天')

# 创建文件夹路径
folder_path = f'E:/PythonJieGuo/姚安28天/{x}天-{v}天-{y}天'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 数据归一化
def normalize_dataframe(train_df, test_df, val_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_data = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns, index=df_train.index)
    test_data = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=df_test.index)
    val_data = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=df_val.index)
    return train_data, test_data, val_data,scaler


data_train, data_test, data_val,scaler = normalize_dataframe(df_train, df_test, df_val)
if data_train is None or data_test is None or data_val is None:
    raise ValueError("数据归一化失败，返回值为 None")

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


win_size = 24
target_feature_idx = 0
train_x, train_y = prepare_data(data_train.values, win_size, target_feature_idx)
val_x, val_y = prepare_data(data_val.values, win_size, target_feature_idx)
test_x, test_y = prepare_data(data_test.values, win_size, target_feature_idx)

print("训练集形状:", train_x.shape, train_y.shape)
print("验证集形状:", val_x.shape, val_y.shape)
print("测试集形状:", test_x.shape, test_y.shape)

StopPatience = round(len(val_x) / 20)
print('早停次数:', StopPatience)

#建立模型
def build_model1(params):
    model = Sequential()
    layers_config = params['layers_config']
    # 提取层数和单元数配置
    num_layers = layers_config['num_layers']
    lstm_units = layers_config['lstm_units']
    # 确保参数合法性
    assert len(lstm_units) == num_layers, "lstm_units长度必须等于num_layers"
    # # 逐层添加LSTM
    # for i in range(num_layers):
    #     return_seq = (i < num_layers - 1)
    #     model.add(LSTM(
    #         units=lstm_units[i],
    #         activation='relu',
    #         input_shape=(train_x.shape[1], train_x.shape[2]) if i == 0 else None,
    #         # input_shape=(train_x.shape[1], train_x.shape[2]) if i == 0 else None,
    #         kernel_regularizer=l2(params['l2_reg']),
    #         return_sequences=return_seq
    #     ))
    for i in range(num_layers):
        if i == 0:
            # 第一层指定input_shape
            model.add(LSTM(
                units=lstm_units[i],
                activation=params['activation'],
                input_shape=(train_x.shape[1], train_x.shape[2]),
                kernel_regularizer=l2(params['l2_reg']),
                return_sequences=(i < num_layers - 1)
            ))
        else:
            # 后续层不设置input_shape
            model.add(LSTM(
                units=lstm_units[i],
                activation=params['activation'],
                kernel_regularizer=l2(params['l2_reg']),
                return_sequences=(i < num_layers - 1)
            ))
    model.add(Dense(1))
    # model.compile(loss = 'mse', optimizer =Adam(learning_rate=params['learning_rate']))
    # 选择优化器
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    # 编译模型
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    return model
# 在每次超参数优化尝试中，动态监控优化标准，并在达到标准时提前停止训练。
from keras.callbacks import Callback
class CustomEarlyStopping(Callback):
    # ①自定义回调函数：创建了一个自定义回调函数CustomEarlyStopping，它结合了patience和PQ_alpha的逻辑。在每个epoch结束时，动态计算PQ_alpha和验证集损失的改善情况。如果满足任一早停条件，则停止训练。
    #②在训练过程中动态监控优化标准，而不是在训练结束后进行评估。
    #③避免了每个超参数组合都运行完整的 500 个 epoch，提高了超参数优化的效率。
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
# 在最大迭代次数（500次）范围内，不断优化超参数，并在达到某个优化标准时停止训练模型。
def objective(space):
    model = build_model1(space)
    print(f"Trying parameters: {space}")
    if model is None:
        raise ValueError("模型构建失败，返回值为 None")
       # 定义自定义早停回调
    custom_early_stopping = CustomEarlyStopping(patience=StopPatience, pq_alpha_threshold=10, progress_period=20)
    #PQ_alpha 的阈值 α 被设置为 10。这个值是一个经验值，适用于大多数情况
    # 训练模型
    history = model.fit(train_x, train_y, epochs=300, batch_size=int(space['batch_size']),
                        verbose=1, validation_data=(val_x, val_y), callbacks=[custom_early_stopping])
    # 获取最终的测试集预测结果
    predictions = model.predict(test_x)
    mse = np.mean((predictions[:, 0] - test_y) ** 2)
    # 获取实际的迭代次数
    actual_epochs = custom_early_stopping.actual_epochs  # 从回调对象中获取实际迭代次数
    # 释放模型:及时释放不再使用的变量，避免内存占用过高
    del model
    return {'loss': mse, 'status': STATUS_OK, 'actual_epochs': actual_epochs}

#定义超参数空间
space = {
# 层数和对应单元数的组合选项
    'layers_config': hp.choice('layers_config', [
        {  # 选项1：2层LSTM
            'num_layers': 2,
            'lstm_units': [
                hp.choice('units_layer21', [16, 32, 64, 128,256]),
                hp.choice('units_layer22', [16, 32, 64, 128,256])
            ]
        },
        {  # 选项2：3层LSTM
            'num_layers': 3,
            'lstm_units':[
                hp.choice('units_layer31', [16, 32, 64, 128,256]),
                hp.choice('units_layer32', [16, 32, 64, 128,256]),
                hp.choice('units_layer33', [16, 32, 64, 128,256])
            ]
        },
        {  # 选项3：1层LSTM
            'num_layers': 1,
            'lstm_units': [
                hp.choice('units_layer11', [16, 32, 64, 128,256]),
            ]
        },
    ]),
    # 'lstm_units': hp.choice('lstm_units', [16, 32, 64, 128, 256]),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),  # 上限调低
    'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-2)),  # 减小正则化
    'batch_size': hp.choice('batch_size', [16,32, 64,128] ),  # 增大批次
    'activation':hp.choice('activation', ['sigmoid', 'tanh', 'relu']),
    'optimizer':hp.choice('optimizer', ['adam', 'sgd', 'rmsprop'])
}
#运行贝叶斯优化
trials = Trials()
best = fmin(fn=objective,
    space=space,

    algo=tpe.suggest,
    max_evals=100,
    trials=trials,


    )
# 获取最佳尝试的索引
best_trial_id = trials.best_trial['tid']
# 获取最佳尝试的实际迭代次数
best_actual_epochs = trials.trials[best_trial_id]['result']['actual_epochs']
print("Best epochs: ", best_actual_epochs)
def get_best_params(best):
    # 获取选择的layers_config选项索引
    layers_choice_idx = best['layers_config']
    # 根据索引确定是2层还是3层配置
    if layers_choice_idx == 0:  # 2层
        units = [
            [16,32, 64, 128,256][best['units_layer21']],
            [16,32, 64, 128,256][best['units_layer22']]
        ]
        num_layers = 2
    elif layers_choice_idx == 1:  # 3层
        units = [
            [16,32, 64, 128,256][best['units_layer31']],
            [16,32, 64, 128,256][best['units_layer32']],
            [16,32, 64, 128,256][best['units_layer33']]
        ]
        num_layers = 3
    else:  # 1层
        units = [
            [16,32, 64, 128,256][best['units_layer11']],
        ]
        num_layers =1
    return {
        'layers_config': {
            'num_layers': num_layers,
            'lstm_units': units
        },
        'learning_rate': np.exp(best['learning_rate']),
        'batch_size': [32, 64,128][best['batch_size']],
        'l2_reg': np.exp(best['l2_reg']),
        'activation':['sigmoid', 'tanh', 'relu'][best['activation']],
        'optimizer': ['adam', 'sgd', 'rmsprop'][best['optimizer']]
    }


best_params = get_best_params(best)
print("最佳超参数:", best_params)
# best_actual_epochs=350
# best_params={'layers_config': {'num_layers': 1, 'lstm_units': [128]}, 'learning_rate': np.float64(0.0001985828801432), 'batch_size': 64, 'l2_reg': np.float64(0.0000103206089863), 'activation': 'tanh', 'optimizer': 'adam'}
保存最佳模型和参数
with open("best_params_and_epochs.txt", "w") as f:
    f.write("# 贝叶斯优化结果\n\n")
    f.write("## 最佳超参数\n\n")
    f.write("```json\n")
    f.write(str(best_params).replace("'", '"') + "\n")
    f.write("```\n\n")
    f.write("## 最佳迭代次数\n\n")
    f.write(f"最佳迭代次数为：{best_actual_epochs}\n")

best_model = build_model1(best_params)
history = best_model.fit(train_x, train_y, epochs=best_actual_epochs, validation_data=(val_x, val_y),
                         batch_size=best_params['batch_size'], verbose=1)
# model_filename = f'best_model_{int(time.time())}.h5'
# best_model.save(model_filename)
# print(f"最佳模型已保存到 {model_filename}")

# 绘制损失曲线图并保存
plt.figure()
plt.plot(history.history['loss'], c='b', label='loss')
plt.plot(history.history['val_loss'], c='g', label='val_loss')
plt.legend()
plt.savefig(os.path.join(folder_path, f'image_{int(time.time())}.png'))
plt.show()

# 测试阶段
for i in range(1, 57, 1):
    data = pd.read_csv(dataFilePath, index_col=0, parse_dates=['Date'])
    trainNum =28 * 1440
    valNum = trainNum + round(trainNum / 4)
    testNum = valNum + i * 1440

    df_train = data.iloc[:trainNum]
    df_val = data.iloc[trainNum:valNum]
    df_test = data.iloc[valNum:testNum]

    x = int(len(df_train) / 1440)
    v = np.round(len(df_val) / 1440, 1)
    y = int(len(df_test) / 1440)

    folder_path = f'E:/PythonJieGuo/姚安28天/{x}天-{v}天-{y}天'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df_train.to_csv(os.path.join(folder_path, '1train_data.csv'), index=True)
    df_test.to_csv(os.path.join(folder_path, '1test_data.csv'), index=True)
    df_val.to_csv(os.path.join(folder_path, '1val_data.csv'), index=True)

    data_train, data_test, data_val, scaler = normalize_dataframe(df_train, df_test, df_val)

    data_train.to_csv(os.path.join(folder_path, '2GuiYi_train_data.csv'), index=True)
    data_test.to_csv(os.path.join(folder_path, '2GuiYi_test_data.csv'), index=True)
    data_val.to_csv(os.path.join(folder_path, '2GuiYi_val_data.csv'), index=True)

    train_x, train_y = prepare_data(data_train.values, win_size, target_feature_idx)
    val_x, val_y = prepare_data(data_val.values, win_size, target_feature_idx)
    test_x, test_y = prepare_data(data_test.values, win_size, target_feature_idx)


    # 评价标准
    def evaluation_Criteria(test_y, y_pred):
        from sklearn import metrics
        from sklearn.metrics import r2_score, explained_variance_score
        mse = metrics.mean_squared_error(test_y, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)
        explained_variance = explained_variance_score(test_y, y_pred)
        mape = np.mean(np.abs((test_y - y_pred) / test_y)) * 100
        return mse, rmse, mae, r2, explained_variance, mape


    train_pred = best_model.predict(train_x)
    val_pred = best_model.predict(val_x)
    y_pred = best_model.predict(test_x)

    mse, rmse, mae, r2, explained_variance, mape = evaluation_Criteria(test_y, y_pred)
    mse_val, rmse_val, mae_val, r2_val, explained_variance_val, mape_val = evaluation_Criteria(val_y, val_pred)
    mse_train, rmse_train, mae_train, r2_train, explained_variance_train, mape_train = evaluation_Criteria(train_y,
                                                                                                           train_pred)

    # 保存预测结果
    array_1d = y_pred.flatten()
    series = pd.Series(array_1d)
    series.to_csv(os.path.join(folder_path, '4y_pred.csv'), index=False, header=False)


    # 逆归一化
    def inverse_normalize_dataframe(normalized_df, scaler):
        if len(normalized_df.shape) == 1:
            normalized_df = normalized_df.reshape(-1, 1)
        original_df = pd.DataFrame(scaler.inverse_transform(normalized_df))
        return original_df


    original_train_y = inverse_normalize_dataframe(train_y.reshape(-1, 1), scaler)
    original_train_yPred = inverse_normalize_dataframe(train_pred.reshape(-1, 1), scaler)
    original_test_y = inverse_normalize_dataframe(test_y.reshape(-1, 1), scaler)
    original_y_pred = inverse_normalize_dataframe(y_pred, scaler)

    # 验证集逆归一化（新增）
    original_val_y = inverse_normalize_dataframe(val_y.reshape(-1, 1), scaler)
    original_val_yPred = inverse_normalize_dataframe(val_pred.reshape(-1, 1), scaler)

    # 保存逆归一化后的数据
    series1 = pd.Series(original_train_y.values.flatten())
    series1.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_trainY.csv'), index=False, header=False)

    series2 = pd.Series(original_train_yPred.values.flatten())
    series2.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_trainYPred.csv'), index=False, header=False)

    series3 = pd.Series(original_test_y.values.flatten())
    series3.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_testY.csv'), index=False, header=False)

    series4 = pd.Series(original_y_pred.values.flatten())
    series4.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_Ypred.csv'), index=False, header=False)

    series5 = pd.Series(original_val_y.values.flatten())
    series5.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_valY.csv'), index=False, header=False)

    series6 = pd.Series(original_val_yPred.values.flatten())
    series6.to_csv(os.path.join(folder_path, '5Scaler_niGuiYiHua_valYPred.csv'), index=False, header=False)

    # 绘制预测结果
    plt.figure(figsize=(15, 8), dpi=300)
    plt.subplot(2, 1, 1)

    # 确保索引是一维的
    train_index = pd.date_range(start=df_train.index[24], end=df_train.index[-1], freq='T')
    val_index = pd.date_range(start=df_val.index[24], end=df_val.index[-1], freq='T')
    test_index = pd.date_range(start=df_test.index[24], end=df_test.index[-1], freq='T')

    plt.plot(train_index, original_train_y.values.flatten(), color='c', linewidth=1)
    plt.plot(val_index, inverse_normalize_dataframe(val_y.reshape(-1, 1), scaler).values.flatten(), color='b',
             linewidth=1)
    plt.plot(val_index, inverse_normalize_dataframe(val_pred, scaler).values.flatten(), color='g', linewidth=1)
    plt.plot(test_index, original_test_y.values.flatten(), color='r', linewidth=1)
    plt.plot(test_index, original_y_pred.values.flatten(), color='y', linewidth=1, linestyle='--')
    plt.title('JSW')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('m')
    plt.legend(['训练集', '验证集真实值', '验证集预测值', '测试集真实值', '测试集预测值'], fontsize='6', loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(test_index, original_test_y.values.flatten(), color='r', linewidth=1, label='测试集JSW')
    plt.plot(test_index, original_y_pred.values.flatten(), color='y', linewidth=1, linestyle='--',
             label='测试集预测JSW')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('m')
    plt.legend(fontsize='6', loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'{x}-{y}--{int(time.time())}.png'), bbox_inches='tight')
    plt.close()

    # 保存评价指标
    data1 = {
        '训练集': [x], '验证集': [v], '测试集': [y], '拟合程度': [r2], "均方误差 (MSE)": [mse], "均方根误差": [rmse],
        "平均绝对误差 (MAE)": [mae], "解释方差分数": [explained_variance], "平均绝对百分比误差 (MAPE)": [mape],
        '拟合程度_val': [r2_val], "均方误差 (MSE)_val": [mse_val], "均方根误差_val": [rmse_val],
        "平均绝对误差 (MAE)_val": [mae_val], "解释方差分数_val": [explained_variance_val],
        "平均绝对百分比误差 (MAPE)_val": [mape_val],
        '拟合程度_train': [r2_train], "均方误差 (MSE)_train": [mse_train], "均方根误差_train": [rmse_train],
        "平均绝对误差 (MAE)_train": [mae_train], "解释方差分数_train": [explained_variance_train],
        "平均绝对百分比误差 (MAPE)_train": [mape_train],
    }
    df1 = pd.DataFrame(data1)
    df1.to_csv('metics姚安28天.csv', mode='a', header=False, index=False)
    print("写入成功")