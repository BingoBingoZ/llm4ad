# template_program = '''
# import numpy as np

# def determine_next_operation(current_status, feasible_operations):
#     """
#     Determine the next operation to schedule for FJSP.
    
#     Args:
#         current_status: A dictionary with machine_status and job_status
#         feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples
        
#     Returns:
#         A tuple (job_id, best_machine, best_processing_time)
#     """
#     machine_status = current_status['machine_status']
#     job_status = current_status['job_status']
    
#     best_job_id = None
#     best_machine = None
#     best_processing_time = float('inf')
    
#     # Simple greedy heuristic: choose the operation with shortest processing time
#     for job_id, machine_id_list, processing_time_list in feasible_operations:
#         for i, (machine_id, processing_time) in enumerate(zip(machine_id_list, processing_time_list)):
#             if processing_time < best_processing_time:
#                 best_processing_time = processing_time
#                 best_machine = machine_id
#                 best_job_id = job_id
    
#     return best_job_id, best_machine, best_processing_time
# '''

# task_description = '''
# Given jobs and machines (each operation can be processed on multiple machines with different processing times), 
# schedule jobs on machines to minimize the total makespan. Design an algorithm to select the next operation and machine in each step.
# '''


template_program = '''
import numpy as np
import random
from collections import defaultdict

def determine_next_operation(current_status, feasible_operations):
    """
    Determine the next operation to schedule for FJSP.
    
    Args:
        current_status: A dictionary with machine_status and job_status
        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples
        
    Returns:
        A tuple (job_id, best_machine, best_processing_time)
    """
    # An example of a constructive algorithm for solving the FJSP problem.
    machine_status = current_status['machine_status']
    job_status = current_status['job_status']
    
    # 计算当前makespan和平均负载
    current_makespan = max(machine_status.values())
    
    # 识别瓶颈机器
    machine_loads = {m_id: load/current_makespan for m_id, load in machine_status.items()}
    bottleneck_machines = {m_id for m_id, load in machine_loads.items() if load > 0.7}
    
    # 计算作业关键度
    job_urgency = {}
    for job_id in job_status:
        remaining_ops = sum(1 for op in feasible_operations if op[0] == job_id)
        # 关键作业 = 剩余操作多 + 已等待时间长
        job_urgency[job_id] = remaining_ops * 0.7 + job_status[job_id] * 0.3
    
    # 评分列表
    scores = []
    
    for operation in feasible_operations:
        job_id, machine_ids, processing_times = operation
        
        for machine_id, proc_time in zip(machine_ids, processing_times):
            # 计算关键指标
            start_time = max(machine_status[machine_id], job_status[job_id])
            completion_time = start_time + proc_time
            
            # 1. 基础完成时间得分
            time_score = completion_time * 1.5
            
            # 2. 作业紧急度得分 (紧急的作业优先)
            urgency_score = -job_urgency.get(job_id, 0) * 0.8
            
            # 3. 瓶颈机器惩罚
            bottleneck_penalty = 20 if machine_id in bottleneck_machines else 0
            
            # 4. 负载均衡奖励 (使用低负载机器优先)
            load_score = machine_status[machine_id] * 0.5
            
            # 5. 处理时间奖励 (短任务优先)
            time_reward = proc_time * 0.7
            
            # 6. 空闲时间惩罚 (减少机器空闲)
            idle_penalty = max(0, job_status[job_id] - machine_status[machine_id]) * 0.6
            
            # 综合得分 (越低越好)
            total_score = time_score + urgency_score + bottleneck_penalty + load_score + time_reward + idle_penalty
            
            scores.append((job_id, machine_id, total_score, proc_time))
    
    # 按得分排序 (低分优先)
    scores.sort(key=lambda x: x[2])
    
    # 选择最佳操作
    if not scores:
        return None, None, None
        
    best_job_id, best_machine, _, best_processing_time = scores[0]
    return best_job_id, best_machine, best_processing_time
'''

task_description = '''
Flexible Job Shop Scheduling Problem (FJSP) Description:
- '#' means code comment.
- We have a set of jobs, each composed of a sequence of operations that must be performed in order.
- We have a set of machines, and each operation can be processed on one of several alternative machines.
- Each operation has different processing times depending on which machine is used.
- A machine can process only one operation at a time.
- Once an operation starts on a machine, it cannot be interrupted.
- Operations of the same job must be processed in their specified sequence.

Your task is to design an algorithm that decides which operation to schedule next and on which machine, with the goal of minimizing the total makespan (completion time of the last job).

Key considerations to incorporate in your algorithm:
1. Machine load balancing - Avoid creating bottlenecks by distributing work evenly
2. Processing time optimization - Consider both immediate and long-term effects of choosing fast/slow machines
3. Critical path identification - Prioritize operations that lie on the critical path
4. Queue management - Consider waiting times and potential machine idle times
5. Look-ahead capability - Evaluate how current decisions impact future scheduling options

Your algorithm should be adaptive, considering the current state of all machines and jobs when making decisions, rather than following a fixed rule. The goal is to outperform simple greedy heuristics.

Input:
- current_status: Contains machine_status (current available time of each machine) and job_status (current available time of each job)
- feasible_operations: List of operations that can be scheduled next, each with a job_id, list of possible machines, and corresponding processing times

Output:
- The job_id, machine_id, and processing_time of the selected operation
'''
