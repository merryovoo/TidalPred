import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class TrainLengthOptimizer:
    def __init__(self, results_folder, candidate_train_lengths=None):
        """
        初始化训练集长度优化器
        """
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
        """加载并处理所有结果文件"""
        print("开始加载和处理结果文件...")

        all_data = []

        for train_length in self.candidate_train_lengths:
            train_folder = f"姚安{train_length}天"
            train_folder_path = os.path.join(self.results_folder, train_folder)

            if not os.path.exists(train_folder_path):
                print(f"警告: 训练集长度 {train_length} 天的文件夹不存在: {train_folder_path}")
                continue

            print(f"\n处理训练集长度: {train_length}天")

            for folder_name in os.listdir(train_folder_path):
                folder_path = os.path.join(train_folder_path, folder_name)

                if not os.path.isdir(folder_path):
                    continue

                try:
                    parts = folder_name.split('-')
                    if len(parts) == 3:
                        train_days = int(parts[0].replace('天', ''))
                        val_days = float(parts[1].replace('天', ''))
                        pred_days = int(parts[2].replace('天', ''))

                        if train_days == train_length:
                            metrics_data = self._extract_metrics_from_folder(
                                folder_path, train_length, pred_days
                            )
                            if metrics_data:
                                all_data.append(metrics_data)

                except Exception as e:
                    print(f"解析文件夹 {folder_name} 失败: {e}")
                    continue

        if all_data:
            self.results_df = pd.DataFrame(all_data)
            print(f"\n成功加载 {len(self.results_df)} 条预测结果")
            print(f"训练集长度分布: {sorted(self.results_df['train_length'].unique())}")
            print(f"预测集长度分布: {sorted(self.results_df['pred_length'].unique())}")
        else:
            raise ValueError("未找到任何有效的结果数据")

    def _extract_metrics_from_folder(self, folder_path, train_length, pred_days):
        """从文件夹中提取指标数据"""
        metrics_file = os.path.join(folder_path, 'metics姚安7天.csv')

        if not os.path.exists(metrics_file):
            return None

        try:
            metrics_df = pd.read_csv(metrics_file, header=None)

            if len(metrics_df) == 0:
                return None

            # 获取最新的一条记录（最后一行）
            latest_metrics = metrics_df.iloc[-1]

            # 根据CSV文件结构解析指标
            # 注意：这里需要根据实际的CSV列顺序调整索引
            metrics_data = {
                'train_length': train_length,
                'pred_length': pred_days,
                'mse': float(latest_metrics[4]),  # 第5列是MSE
                'rmse': float(latest_metrics[5]),  # 第6列是RMSE
                'mae': float(latest_metrics[6]),  # 第7列是MAE
                'r2': float(latest_metrics[3]),  # 第4列是R²
                'explained_variance': float(latest_metrics[7]),  # 第8列
                'mape': float(latest_metrics[8]),  # 第9列
                'folder_path': folder_path
            }

            print(f"  预测长度 {pred_days}天: "
                  f"MSE={metrics_data['mse']:.6f}, R²={metrics_data['r2']:.4f}")

            return metrics_data

        except Exception as e:
            print(f"处理文件夹 {folder_path} 失败: {e}")
            return None

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

    def visualize_analysis(self, target_pred_length=7, save_path=None):
        """可视化分析结果"""
        if self.results_df is None:
            print("请先加载数据")
            return

        target_data = self.results_df[
            self.results_df['pred_length'] == target_pred_length
            ].copy()

        if len(target_data) == 0:
            print(f"没有找到预测集长度为 {target_pred_length} 天的数据")
            return

        target_data['estimated_time'] = target_data['train_length'].apply(
            self.estimate_training_time
        )

        # 创建可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. MSE vs 训练集长度
        ax1.plot(target_data['train_length'], target_data['mse'], 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=self.accuracy_thresholds['mse'], color='r', linestyle='--', label='MSE阈值')
        ax1.set_xlabel('训练集长度 (天)')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE vs 训练集长度')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. R² vs 训练集长度
        ax2.plot(target_data['train_length'], target_data['r2'], 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=self.accuracy_thresholds['r2'], color='r', linestyle='--', label='R²阈值')
        ax2.set_xlabel('训练集长度 (天)')
        ax2.set_ylabel('R²')
        ax2.set_title('R² vs 训练集长度')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. 预计时间 vs 训练集长度
        ax3.plot(target_data['train_length'], target_data['estimated_time'], 's-',
                 linewidth=2, markersize=8, color='orange')
        ax3.set_xlabel('训练集长度 (天)')
        ax3.set_ylabel('预计训练时间 (小时)')
        ax3.set_title('训练时间 vs 训练集长度')
        ax3.grid(True, alpha=0.3)

        # 4. 标记最优选择
        ax4.bar(target_data['train_length'], target_data['r2'], alpha=0.7)
        if self.optimal_train_length is not None:
            optimal_data = target_data[target_data['train_length'] == self.optimal_train_length]
            if len(optimal_data) > 0:
                ax4.bar(optimal_data['train_length'], optimal_data['r2'],
                        color='red', alpha=0.9, label='最优选择')
        ax4.set_xlabel('训练集长度 (天)')
        ax4.set_ylabel('R²')
        ax4.set_title('最优训练集长度选择')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分析图表已保存至: {save_path}")

        plt.show()


# 运行测试
def main():
    # 初始化优化器
    results_folder = 'DALSTM-findBestTrain-testData/'
    optimizer = TrainLengthOptimizer(results_folder)

    try:
        # 加载和处理结果
        optimizer.load_and_process_results()

        # 寻找最优训练集长度（针对7天预测）
        optimal_length = optimizer.find_optimal_train_length(target_pred_length=7)

        # 可视化分析
        optimizer.visualize_analysis(
            target_pred_length=7,
            save_path='training_length_optimization_analysis.png'
        )

        print(f"\n 优化完成！最终推荐的最佳训练集长度为: {optimal_length} 天")

        return optimizer, optimal_length

    except Exception as e:
        print(f" 运行出错: {e}")
        return None, None


if __name__ == "__main__":
    optimizer, optimal_length = main()