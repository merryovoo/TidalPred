import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

class TrainLengthOptimizer:
    def __init__(self, csv_file_path, results_folder=None, candidate_train_lengths=None):
       #初始化训练集长度优化器
        self.csv_file_path = csv_file_path
        self.results_folder = results_folder
        if candidate_train_lengths is None:
            self.candidate_train_lengths = [7, 14, 21, 28]
        else:
            self.candidate_train_lengths = candidate_train_lengths
        self.results_df = None
        self.optimal_train_length = None
        self.accuracy_thresholds = {
            'mse': 0.01,
            'rmse': 0.1,
            'mae': 0.1,
            'r2': 0.9
        }

    def load_and_process_results(self):
        """直接从CSV文件加载并处理所有结果"""
        print("开始加载和处理CSV结果文件...")
        try:
            # 读取CSV文件
            if not os.path.exists(self.csv_file_path):
                raise FileNotFoundError(f"CSV文件不存在: {self.csv_file_path}")
            # 读取数据
            data_df = pd.read_csv(self.csv_file_path)
            print(f"成功读取CSV文件，共 {len(data_df)} 行数据")
            print(f"数据列名: {list(data_df.columns)}")
            # 处理数据 - 根据你的CSV文件结构调整这部分
            all_data = []
            for _, row in data_df.iterrows():
                try:
                    metrics_data = {
                        'train_length': int(row.get('train_length', row.get('train_days', 0))),
                        'pred_length': int(row.get('pred_length', row.get('pred_days', 0))),
                        'mse': float(row.get('mse', 0)),
                        'rmse': float(row.get('rmse', 0)),
                        'mae': float(row.get('mae', 0)),
                        'r2': float(row.get('r2', row.get('r2_score', 0))),
                        'explained_variance': float(row.get('explained_variance', 0)),
                        'mape': float(row.get('mape', 0)),
                        'model_type': row.get('model_type', 'unknown'),
                        'experiment_id': row.get('experiment_id', 'unknown')
                    }
                    # 只保留候选训练集长度的数据
                    if metrics_data['train_length'] in self.candidate_train_lengths:
                        all_data.append(metrics_data)
                except Exception as e:
                    print(f"处理行数据失败: {e}")
                    continue
            if all_data:
                self.results_df = pd.DataFrame(all_data)
                print(f"\n成功处理 {len(self.results_df)} 条有效预测结果")
                print(f"训练集长度分布: {sorted(self.results_df['train_length'].unique())}")
                print(f"预测集长度分布: {sorted(self.results_df['pred_length'].unique())}")
                # 显示数据统计
                print("\n数据统计:")
                print(f"- MSE范围: [{self.results_df['mse'].min():.6f}, {self.results_df['mse'].max():.6f}]")
                print(f"- R²范围: [{self.results_df['r2'].min():.4f}, {self.results_df['r2'].max():.4f}]")
            else:
                raise ValueError("未找到任何有效的实验结果数据")
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            raise
    def _auto_detect_columns(self, columns):
        """自动检测列名映射"""
        column_mapping = {}
        columns_lower = [str(col).lower() for col in columns]
        # 训练集长度相关列
        train_keys = ['train_length', 'train_days', 'train_len', 'training_length']
        for key in train_keys:
            for i, col in enumerate(columns_lower):
                if key in col:
                    column_mapping['train_length'] = columns[i]
                    break
        # 预测集长度相关列
        pred_keys = ['pred_length', 'pred_days', 'pred_len', 'prediction_length']
        for key in pred_keys:
            for i, col in enumerate(columns_lower):
                if key in col:
                    column_mapping['pred_length'] = columns[i]
                    break
        # 指标列
        metric_mappings = {
            'mse': ['mse'],
            'rmse': ['rmse'],
            'mae': ['mae'],
            'r2': ['r2', 'r2_score', 'r_squared'],
            'explained_variance': ['explained_variance', 'exp_var'],
            'mape': ['mape']
        }
        for target, sources in metric_mappings.items():
            for source in sources:
                for i, col in enumerate(columns_lower):
                    if source in col:
                        column_mapping[target] = columns[i]
                        break

        return column_mapping
    def estimate_training_time(self, train_length):
        """估计训练时间"""
        # 简单的线性模型：基础时间 + 每增加一天的时间
        base_time = 0.5  # 小时
        time_per_day = 0.1  # 小时/天
        return base_time + time_per_day * train_length

    def find_optimal_train_length(self, target_pred_length=7):
        """寻找最优训练集长度"""
        print(f"\n开始寻找最优训练集长度 (预测集长度: {target_pred_length}天)")
        print("=" * 60)

        # 步骤1: 筛选出预测集长度等于目标长度的所有行
        target_pred_data = self.results_df[
            self.results_df['pred_length'] == target_pred_length
            ].copy()
        if len(target_pred_data) == 0:
            raise ValueError(f"没有找到预测集长度为 {target_pred_length} 天的数据")
        # 步骤2: 按照训练集长度排序（从短到长）
        target_pred_data = target_pred_data.sort_values('train_length')
        print("候选训练集长度分析:")
        print("-" * 50)
        # 步骤3-6: 遍历每个训练集长度，检查精度阈值
        optimal_candidate = None
        all_candidates = []
        for _, row in target_pred_data.iterrows():
            train_length = row['train_length']
            metrics = {
                'mse': row['mse'],
                'rmse': row['rmse'],
                'mae': row['mae'],
                'r2': row['r2']
            }
            # 估计训练时间
            estimated_time = self.estimate_training_time(train_length)
            # 检查是否满足精度阈值
            meets_criteria = (
                    metrics['mse'] < self.accuracy_thresholds['mse'] and
                    metrics['rmse'] < self.accuracy_thresholds['rmse'] and
                    metrics['mae'] < self.accuracy_thresholds['mae'] and
                    metrics['r2'] > self.accuracy_thresholds['r2']
            )
            candidate_info = {
                'train_length': train_length,
                'metrics': metrics,
                'estimated_time': estimated_time,
                'meets_criteria': meets_criteria
            }
            all_candidates.append(candidate_info)

            status = "✓ 满足精度要求" if meets_criteria else "✗ 不满足精度要求"
            print(f"训练集 {train_length:2d} 天: "
                  f"MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}, "
                  f"预计时间={estimated_time:.2f}h - {status}")
            # 步骤4-5: 如果满足条件且是第一个满足的，选择它
            if meets_criteria and optimal_candidate is None:
                optimal_candidate = candidate_info
                print(f"  → 找到满足条件的候选: {train_length}天")
        # 步骤7: 如果没有满足条件的，选择MSE最小的
        if optimal_candidate is None:
            print(f"\n无满足精度要求的训练集长度，选择MSE最小的")
            optimal_candidate = min(all_candidates, key=lambda x: x['metrics']['mse'])
        self.optimal_train_length = optimal_candidate['train_length']

        # 输出最终选择
        print(f"\n" + "=" * 60)
        print(f"最终选择的训练集长度: {self.optimal_train_length} 天")
        print(f"性能指标:")
        print(f"   - MSE: {optimal_candidate['metrics']['mse']:.6f}")
        print(f"   - RMSE: {optimal_candidate['metrics']['rmse']:.6f}")
        print(f"   - MAE: {optimal_candidate['metrics']['mae']:.6f}")
        print(f"   - R²: {optimal_candidate['metrics']['r2']:.4f}")
        print(f"预计训练时间: {optimal_candidate['estimated_time']:.2f} 小时")
        return self.optimal_train_length

# 运行测试
def main():
    # 初始化优化器
    csv_file_path = 'DALSTM-findBestTrain-testData.csv'  # 请替换为你的CSV文件路径
    optimizer = TrainLengthOptimizer(
        csv_file_path=csv_file_path,
    )
    try:
        # 加载和处理结果
        optimizer.load_and_process_results()
        # 寻找最优训练集长度
        optimal_length = optimizer.find_optimal_train_length(target_pred_length=28)
        print(f"\n优化完成！最终推荐的最佳训练集长度为: {optimal_length} 天")
        return optimizer, optimal_length
    except Exception as e:
        print(f"运行出错: {e}")
        return None, None

if __name__ == "__main__":
    optimizer, optimal_length = main()