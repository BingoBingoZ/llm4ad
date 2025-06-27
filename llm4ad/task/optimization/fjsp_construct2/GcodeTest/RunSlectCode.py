import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径以便导入模块
sys.path.append('/home/zgb/llm4ad')

# from load_data import load_fjsp_data, print_fjsp_instance
from load_data import load_fjsp_data
from llm4ad.task.optimization.fjsp_construct2.evaluation import FJSPEvaluation
# from llm4ad.task.optimization.fjsp_construct2 import FJSPEvaluation

# 导入EOH生成的最佳代码
sys.path.append('/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2')
from best_code import determine_next_operation

def schedule_fjsp_instance(processing_times, n_jobs, n_machines):
    """
    使用最佳算法调度单个FJSP实例
    
    返回:
        makespan: 完工时间
        schedule: 调度方案
    """
    # 初始化FJSP求解器状态
    machine_status = [0] * n_machines  # 每台机器的可用时间
    job_status = [0] * n_jobs          # 每个作业的可用时间
    operation_sequence = [[] for _ in range(n_jobs)]  # 各作业的操作序列
    
    # 初始化作业的下一工序索引
    job_next_op = [0] * n_jobs
    
    # 计算总工序数
    n_ops = [len(job_ops) for job_ops in processing_times]
    total_ops = sum(n_ops)
    scheduled_ops = 0
    
    # 调度所有工序
    while scheduled_ops < total_ops:
        # 确定可行操作
        feasible_operations = []
        for job_id in range(n_jobs):
            op_idx = job_next_op[job_id]
            if op_idx < n_ops[job_id]:
                machine_id_list, processing_time_list = processing_times[job_id][op_idx]
                # 检查是否有机器可用
                for machine_id, processing_time in zip(machine_id_list, processing_time_list):
                    if job_status[job_id] <= machine_status[machine_id]:
                        feasible_operations.append((job_id, machine_id_list, processing_time_list))
                        break
        
        if not feasible_operations:
            break
        
        # 使用最佳算法确定下一步调度的操作
        job_id, best_machine, best_processing_time = determine_next_operation(
            {'machine_status': machine_status, 'job_status': job_status}, 
            feasible_operations
        )
        op_idx = job_next_op[job_id]
        
        # 在选定的机器上安排操作
        start_time = max(job_status[job_id], machine_status[best_machine])
        end_time = start_time + best_processing_time
        machine_status[best_machine] = end_time
        job_status[job_id] = end_time
        operation_sequence[job_id].append((best_machine, start_time, end_time))
        
        # 更新作业的下一工序索引
        job_next_op[job_id] += 1
        scheduled_ops += 1

    # 计算完工时间
    makespan = max(job_status)
    return makespan, operation_sequence

def plot_gantt_chart(schedule, n_jobs, n_machines):
    """绘制甘特图展示调度结果"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建颜色映射
    colors = plt.cm.get_cmap('tab10', n_jobs)
    
    # 绘制每个作业的操作
    for job_idx, operations in enumerate(schedule):
        for operation in operations:
            machine, start_time, end_time = operation
            # 绘制水平条形图
            ax.barh(machine, end_time - start_time, left=start_time,
                    color=colors(job_idx), label=f'作业 {job_idx}')
    
    # 自定义图表
    ax.set_xlabel('时间')
    ax.set_ylabel('机器')
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'机器 {i}' for i in range(n_machines)])
    ax.set_title('FJSP调度甘特图')
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 移除重复标签
    ax.legend(by_label.values(), by_label.keys(), title="作业", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # plt.savefig(f'gantt_chart_{n_jobs}jobs_{n_machines}machines.png')
    plt.show()

def main():
    filepath='/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2/GcodeTest/data_test/Public/v_la01.fjs'
    processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
    
    # 打印问题实例
    # print_fjsp_instance(processing_times, n_jobs, n_machines)
    
    # 使用最佳算法调度
    start_time = time.time()
    makespan, schedule = schedule_fjsp_instance(processing_times, n_jobs, n_machines)
    end_time = time.time()
    
    print(f"\n调度结果:")
    print(f"完工时间(Makespan): {makespan}")
    print(f"计算时间: {end_time - start_time:.4f} 秒")
    
    # 绘制甘特图
    plot_gantt_chart(schedule, n_jobs, n_machines)
    
    # print("\n调度详情:")
    # for job_id, operations in enumerate(schedule):
    #     print(f"作业 {job_id}:")
    #     for op_id, (machine, start, end) in enumerate(operations):
    #         print(f"  工序 {op_id}: 机器 {machine}, 开始时间 {start}, 结束时间 {end}, 处理时间 {end-start}")

if __name__ == "__main__":
    main()