import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import random

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
                # 直接添加到可行操作，不再进行提前筛选
                feasible_operations.append((job_id, machine_id_list, processing_time_list))
        
        if not feasible_operations:
            # 这种情况不应该发生，因为我们已经检查了total_ops
            print("警告: 无可行操作但仍有未调度的工序")
            break
        
        # 使用最佳算法确定下一步调度的操作
        job_id, best_machine, best_processing_time = determine_next_operation(
            {'machine_status': machine_status, 'job_status': job_status}, 
            feasible_operations
        )
        
        if job_id is None:
            # 如果算法无法确定下一个操作，找到最早可用的时间点
            next_time = float('inf')
            for m in range(n_machines):
                if machine_status[m] < next_time:
                    next_time = machine_status[m]
            
            for j in range(n_jobs):
                if job_next_op[j] < n_ops[j] and job_status[j] < next_time:
                    next_time = job_status[j]
            
            # 前进到下一个时间点
            for m in range(n_machines):
                if machine_status[m] < next_time:
                    machine_status[m] = next_time
            
            for j in range(n_jobs):
                if job_status[j] < next_time:
                    job_status[j] = next_time
            
            continue
        
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
        
        # 调试信息
        # print(f"调度作业 {job_id} 的工序 {op_idx} 到机器 {best_machine}, 开始:{start_time}, 结束:{end_time}")
        # print(f"已调度: {scheduled_ops}/{total_ops} 工序")

    # 计算完工时间
    makespan = max(job_status)
    
    # 验证所有工序是否都已调度
    for job_id in range(n_jobs):
        if len(operation_sequence[job_id]) != n_ops[job_id]:
            print(f"警告: 作业 {job_id} 只调度了 {len(operation_sequence[job_id])}/{n_ops[job_id]} 个工序")
    
    return makespan, operation_sequence

# 画甘特图
def drawGantt(schedule, n_jobs, n_machines):
    # 将按作业组织的数据转换为按机器组织的数据
    machine_schedules = [[] for _ in range(n_machines)]
    
    # 初始化每个机器的时间表
    for i in range(n_machines):
        machine_schedules[i] = [i]  # 第一个元素是机器ID
    
    # 填充机器时间表
    for job_idx, operations in enumerate(schedule):
        for op_idx, operation in enumerate(operations):
            machine, start_time, end_time = operation
            # 添加到对应机器的时间表 [开始时间, 作业ID, 工序ID, 结束时间]
            machine_schedules[machine].append([start_time, job_idx, op_idx, end_time])
    
    # 按照开始时间排序每个机器的操作
    for machine_idx in range(n_machines):
        machine_schedules[machine_idx][1:] = sorted(machine_schedules[machine_idx][1:], key=lambda x: x[0])
    
    # 创建一个新的图形
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色映射字典，为每个工件分配一个唯一的颜色
    color_map = {}
    for machine_schedule in machine_schedules:
        for task_data in machine_schedule[1:]:
            job_idx = task_data[1]
            if job_idx not in color_map:
                # 为新工件分配一个随机颜色
                color_map[job_idx] = (random.random(), random.random(), random.random())

    # 遍历机器
    for machine_idx, machine_schedule in enumerate(machine_schedules):
        for task_data in machine_schedule[1:]:
            start_time, job_idx, operation_idx, end_time = task_data
            color = color_map[job_idx]  # 获取工件的颜色

            # 绘制甘特图条形，使用工件的颜色
            ax.barh(machine_idx, end_time - start_time, left=start_time, height=0.4, color=color)

            # 在色块内部标注工件-工序
            label = f'{job_idx + 1}-{operation_idx + 1}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='white',
                    fontsize=10)

    # 设置Y轴标签为机器名称
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i + 1}' for i in range(n_machines)])

    # 设置X轴标签
    plt.xlabel("Time")

    # 添加标题
    plt.title("FJSP Gantt Chart")

    # 显示图形
    plt.show()

def main():
    filepath='/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2/GcodeTest/data_test/Public/v_la01.fjs' # Mk01 v_la01
    processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
    
    # 打印问题信息
    print(f"问题规模: {n_jobs}个作业, {n_machines}台机器")
    print(f"各作业工序数: {[len(job_ops) for job_ops in processing_times]}")
    total_ops = sum(len(job_ops) for job_ops in processing_times)
    print(f"总工序数: {total_ops}")
    
    # 使用最佳算法调度
    start_time = time.time()
    makespan, schedule = schedule_fjsp_instance(processing_times, n_jobs, n_machines)
    end_time = time.time()
    
    print(f"\n调度结果:")
    print(f"完工时间(Makespan): {makespan}")
    print(f"计算时间: {end_time - start_time:.4f} 秒")
    
    # 检查每个作业的工序数
    for job_id, operations in enumerate(schedule):
        print(f"作业 {job_id}: {len(operations)}/{len(processing_times[job_id])} 个工序")
    
    # 详细输出调度结果
    # print("\n调度详情:")
    # for job_id, operations in enumerate(schedule):
    #     print(f"作业 {job_id}:")
    #     for op_id, (machine, start, end) in enumerate(operations):
    #         print(f"  工序 {op_id}: 机器 {machine}, 开始时间 {start}, 结束时间 {end}, 处理时间 {end-start}")
    
    # 绘制甘特图 - 使用新的drawGantt函数
    drawGantt(schedule, n_jobs, n_machines)

if __name__ == "__main__":
    main()




