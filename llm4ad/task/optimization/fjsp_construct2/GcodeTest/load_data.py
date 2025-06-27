import numpy as np
import os
import re

def load_fjsp_data(filepath):
    """
    加载FJSP标准数据集并转换为代码所需的格式
    
    参数:
        filepath: 数据文件路径
        
    返回:
        (processing_times, n_jobs, n_machines): 转换后的处理时间矩阵和问题规模
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 移除注释行和空行
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    # 解析问题规模
    first_line = re.findall(r'\d+', lines[0])
    n_jobs = int(first_line[0])
    n_machines = int(first_line[1])
    
    # 解析工序信息
    data_line = ' '.join(lines[1:])  # 合并所有剩余行
    data_values = list(map(int, re.findall(r'\d+', data_line)))
    
    # 初始化处理时间矩阵
    processing_times = []
    
    index = 0
    for job_id in range(n_jobs):
        job_operations = []
        # 获取当前作业的工序数
        n_operations = data_values[index]
        index += 1
        
        for _ in range(n_operations):
            # 获取当前工序的可选机器数
            n_machines_for_op = data_values[index]
            index += 1
            
            # 提取机器ID和处理时间
            machine_ids = []
            proc_times = []
            for _ in range(n_machines_for_op):
                machine_id = data_values[index] - 1  # 通常数据集从1开始编号，转为0开始
                index += 1
                proc_time = data_values[index]
                index += 1
                
                machine_ids.append(machine_id)
                proc_times.append(proc_time)
            
            job_operations.append((machine_ids, proc_times))
        
        processing_times.append(job_operations)
    
    return processing_times, n_jobs, n_machines

def print_fjsp_instance(processing_times, n_jobs, n_machines):
    """打印FJSP问题实例的可读形式"""
    print(f"问题规模: {n_jobs}个作业, {n_machines}台机器")
    
    for job_id, job_ops in enumerate(processing_times):
        print(f"\n作业 {job_id} (共{len(job_ops)}道工序):")
        
        for op_id, (machine_ids, proc_times) in enumerate(job_ops):
            print(f"  工序 {op_id}:")
            for m_idx, (m_id, p_time) in enumerate(zip(machine_ids, proc_times)):
                print(f"    机器 {m_id}: {p_time}时间单位")

# if __name__ == "__main__":
#     filepath='/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2/GcodeTest/data_test/Public/Mk01.fjs'
#     processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
#     print_fjsp_instance(processing_times, n_jobs, n_machines)