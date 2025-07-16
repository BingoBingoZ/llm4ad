# filepath: /home/zgb/llm4ad/llm4ad/task/optimization/fjsp_ls/gls/ls_neighborhood.py
from typing import List, Tuple, Dict, Any
import random
import copy

def apply_swap_neighborhood(solution):
    """交换同一机器上的相邻操作"""
    # 获取当前解的调度方案
    schedule = solution.schedule
    
    # 按机器分组
    machine_ops = {}
    for op in schedule:
        machine_id = op['machine_id']
        if machine_id not in machine_ops:
            machine_ops[machine_id] = []
        machine_ops[machine_id].append(op)
    
    # 为每个机器上的操作排序
    for machine_id in machine_ops:
        machine_ops[machine_id].sort(key=lambda x: x['start_time'])
    
    # 随机选择一个机器和两个相邻操作
    if machine_ops:
        machine_id = random.choice(list(machine_ops.keys()))
        ops = machine_ops[machine_id]
        
        if len(ops) >= 2:
            # 随机选择一个索引
            idx = random.randint(0, len(ops) - 2)
            
            # 获取对应的操作序列位置
            op_seq = solution.get_operation_sequence()
            pos1 = op_seq.index(ops[idx]['job_id'] * 100 + ops[idx]['op_idx'])
            pos2 = op_seq.index(ops[idx + 1]['job_id'] * 100 + ops[idx + 1]['op_idx'])
            
            # 执行交换
            new_solution = solution.copy()
            new_solution.swap_operations(pos1, pos2)
            
            return new_solution
    
    return solution

def apply_reassign_neighborhood(solution):
    """将操作重新分配到不同的机器"""
    # 随机选择一个操作
    machine_assign = solution.get_machine_assignment()
    
    if machine_assign:
        keys = list(machine_assign.keys())
        job_id, op_idx = random.choice(keys)
        
        # 获取当前机器和可用机器列表
        current_machine = machine_assign[(job_id, op_idx)]
        available_machines = solution.get_available_machines(job_id, op_idx)
        
        # 排除当前机器
        if current_machine in available_machines and len(available_machines) > 1:
            available_machines.remove(current_machine)
            
            # 随机选择一个新机器
            new_machine = random.choice(available_machines)
            
            # 执行重新分配
            new_solution = solution.copy()
            new_solution.reassign_machine(job_id, op_idx, new_machine)
            
            return new_solution
    
    return solution

def apply_critical_path_neighborhood(solution):
    """改进关键路径上的操作"""
    # 获取makespan
    makespan = solution.evaluate()
    
    # 找出结束时间接近makespan的操作（关键操作）
    critical_threshold = makespan * 0.9
    critical_ops = []
    
    for op in solution.schedule:
        if op['end_time'] >= critical_threshold:
            critical_ops.append((op['job_id'], op['op_idx']))
    
    # 随机选择一个关键操作
    if critical_ops:
        job_id, op_idx = random.choice(critical_ops)
        
        # 尝试重新分配机器
        current_machine = solution.get_machine_assignment()[(job_id, op_idx)]
        available_machines = solution.get_available_machines(job_id, op_idx)
        
        if current_machine in available_machines and len(available_machines) > 1:
            available_machines.remove(current_machine)
            
            # 随机选择一个新机器
            new_machine = random.choice(available_machines)
            
            # 执行重新分配
            new_solution = solution.copy()
            new_solution.reassign_machine(job_id, op_idx, new_machine)
            
            return new_solution
    
    return solution

def apply_neighborhood(solution, processing_times, n_jobs, n_machines, max_iterations=100):
    """组合多种邻域搜索策略"""
    best_solution = solution.copy()
    best_makespan = solution.evaluate()
    
    # 邻域操作列表
    neighborhoods = [
        apply_swap_neighborhood,
        apply_reassign_neighborhood,
        apply_critical_path_neighborhood
    ]
    
    # 邻域搜索
    for i in range(max_iterations):
        # 随机选择一个邻域操作
        neighborhood_op = random.choice(neighborhoods)
        
        # 应用邻域操作
        neighbor = neighborhood_op(best_solution)
        
        # 评估邻居解
        neighbor_makespan = neighbor.evaluate()
        
        # 接受更好的解
        if neighbor_makespan < best_makespan:
            best_solution = neighbor
            best_makespan = neighbor_makespan
    
    return best_solution