'''
批量测试多个数据集
'''

import os
import sys
import numpy as np
import time
import glob
import pandas as pd

# 添加项目根目录到系统路径以便导入模块
sys.path.append('/home/zgb/llm4ad')

from load_data import load_fjsp_data
from test_best_algorithm import schedule_fjsp_instance

def run_batch_test(data_dir):
    """对指定目录中的所有FJSP数据集运行批量测试"""
    results = []
    
    # 查找所有数据文件
    data_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    for filepath in data_files:
        filename = os.path.basename(filepath)
        print(f"\n正在处理数据集: {filename}")
        
        try:
            # 加载数据
            processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
            
            # 使用最佳算法调度
            start_time = time.time()
            makespan, _ = schedule_fjsp_instance(processing_times, n_jobs, n_machines)
            compute_time = time.time() - start_time
            
            # 记录结果
            results.append({
                'Dataset': filename,
                'Jobs': n_jobs,
                'Machines': n_machines,
                'Makespan': makespan,
                'Compute_Time(s)': compute_time
            })
            
            print(f"作业数: {n_jobs}, 机器数: {n_machines}")
            print(f"完工时间(Makespan): {makespan}")
            print(f"计算时间: {compute_time:.4f} 秒")
            
        except Exception as e:
            print(f"处理数据集 {filename} 时出错: {str(e)}")
    
    # 将结果保存为CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('fjsp_batch_results.csv', index=False)
        print(f"\n所有结果已保存到 fjsp_batch_results.csv")
        
        # 打印汇总统计
        print("\n结果汇总:")
        print(results_df)

if __name__ == "__main__":
    # 指定数据目录
    data_dir = '/home/zgb/llm4ad/GcodeTest/data'
    
    if not os.path.exists(data_dir):
        print(f"数据目录 {data_dir} 不存在，请先创建并添加数据文件")
        sys.exit(1)
    
    run_batch_test(data_dir)