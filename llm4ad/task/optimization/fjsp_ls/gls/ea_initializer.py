# filepath: /home/zgb/llm4ad/llm4ad/task/optimization/fjsp_ls/gls/ea_initializer.py
from typing import List, Tuple, Dict, Any
import random
import copy
from llm4ad.task.optimization.fjsp_ls.gls.solution import FJSPSolution
from llm4ad.task.optimization.fjsp_construct2.best_code import determine_next_operation

def construct_solution(processing_times, n_jobs, n_machines):
    """使用构造式方法生成初始解"""
    # 初始化状态
    machine_status = [0] * n_machines  # 机器可用时间
    job_status = [0] * n_jobs          # 作业可用时间
    job_next_op = [0] * n_jobs         # 每个作业的下一工序
    n_ops = [len(job_ops) for job_ops in processing_times]
    
    # 存储最终调度方案
    schedule = []
    
    # 调度所有工序
    while sum(job_next_op) < sum(n_ops):
        # 确定可行操作
        feasible_operations = []
        for job_id in range(n_jobs):
            op_idx = job_next_op[job_id]
            if op_idx < n_ops[job_id]:
                machine_id_list, proc_time_list = processing_times[job_id][op_idx]
                feasible_operations.append((job_id, machine_id_list, proc_time_list))
        
        if not feasible_operations:
            break
            
        # 使用启发式规则选择下一个操作
        current_status = {'machine_status': machine_status, 'job_status': job_status}
        job_id, machine_id, proc_time = determine_next_operation(current_status, feasible_operations)
        
        # 安排选择的操作
        start_time = max(job_status[job_id], machine_status[machine_id])
        end_time = start_time + proc_time
        
        # 更新状态
        machine_status[machine_id] = end_time
        job_status[job_id] = end_time
        
        # 记录调度决策
        schedule.append({
            'job_id': job_id,
            'op_idx': job_next_op[job_id],
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': end_time
        })
        
        job_next_op[job_id] += 1
    
    # 转换为FJSPSolution对象
    solution = FJSPSolution(schedule, n_jobs, n_machines, processing_times)
    return solution

def generate_initial_solution(processing_times, n_jobs, n_machines, population_size=20, generations=10):
    """使用简单进化算法生成优质初始解"""
    # 初始种群
    population = []
    for _ in range(population_size):
        # 添加随机权重的构造解
        urgency_weight = random.uniform(1.5, 2.5)
        time_weight = random.uniform(0.8, 1.2)
        solution = construct_with_weights(processing_times, n_jobs, n_machines, urgency_weight, time_weight)
        population.append(solution)
    
    # 进化过程
    for generation in range(generations):
        # 评估
        population.sort(key=lambda x: x.evaluate())
        
        # 选择前半部分
        elite_size = population_size // 2
        elite = population[:elite_size]
        
        # 生成新个体
        new_population = list(elite)  # 保留精英
        
        while len(new_population) < population_size:
            # 交叉
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = crossover(parent1, parent2, processing_times)
            
            # 变异
            if random.random() < 0.2:
                mutate(child)
                
            new_population.append(child)
        
        population = new_population
    
    # 返回最佳解
    population.sort(key=lambda x: x.evaluate())
    return population[0]

def construct_with_weights(processing_times, n_jobs, n_machines, urgency_weight, time_weight):
    """使用自定义权重的构造式方法"""
    # 初始化状态
    machine_status = [0] * n_machines
    job_status = [0] * n_jobs
    job_next_op = [0] * n_jobs
    n_ops = [len(job_ops) for job_ops in processing_times]
    
    schedule = []
    
    while sum(job_next_op) < sum(n_ops):
        feasible_operations = []
        for job_id in range(n_jobs):
            op_idx = job_next_op[job_id]
            if op_idx < n_ops[job_id]:
                machine_id_list, proc_time_list = processing_times[job_id][op_idx]
                feasible_operations.append((job_id, machine_id_list, proc_time_list))
        
        if not feasible_operations:
            break
        
        # 自定义评分函数
        best_score = float('inf')
        best_job_id = None
        best_machine_id = None
        best_proc_time = None
        
        for job_id, machine_id_list, proc_time_list in feasible_operations:
            job_pending_operations = n_ops[job_id] - job_next_op[job_id]
            
            for machine_id, proc_time in zip(machine_id_list, proc_time_list):
                start_time = max(job_status[job_id], machine_status[machine_id])
                finish_time = start_time + proc_time
                
                # 自定义评分
                urgency_penalty = job_pending_operations * urgency_weight
                time_score = finish_time * time_weight
                score = time_score + urgency_penalty
                
                if score < best_score:
                    best_score = score
                    best_job_id = job_id
                    best_machine_id = machine_id
                    best_proc_time = proc_time
        
        # 安排选择的操作
        job_id = best_job_id
        machine_id = best_machine_id
        proc_time = best_proc_time
        
        start_time = max(job_status[job_id], machine_status[machine_id])
        end_time = start_time + proc_time
        
        machine_status[machine_id] = end_time
        job_status[job_id] = end_time
        
        schedule.append({
            'job_id': job_id,
            'op_idx': job_next_op[job_id],
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': end_time
        })
        
        job_next_op[job_id] += 1
    
    solution = FJSPSolution(schedule, n_jobs, n_machines, processing_times)
    return solution

def crossover(parent1, parent2, processing_times):
    """两个解的交叉操作"""
    p1_seq = parent1.get_operation_sequence()
    p2_seq = parent2.get_operation_sequence()
    p1_assign = parent1.get_machine_assignment()
    p2_assign = parent2.get_machine_assignment()
    
    # 交叉点
    crossover_point = random.randint(1, len(p1_seq) - 1)
    
    # 新的操作序列
    new_seq = p1_seq[:crossover_point] + p2_seq[crossover_point:]
    
    # 新的机器分配
    new_assign = {}
    for key in set(p1_assign.keys()).union(p2_assign.keys()):
        if key in p1_assign and key in p2_assign:
            new_assign[key] = p1_assign[key] if random.random() < 0.5 else p2_assign[key]
        elif key in p1_assign:
            new_assign[key] = p1_assign[key]
        else:
            new_assign[key] = p2_assign[key]
    
    # 创建新解
    child = FJSPSolution.from_encoding(new_seq, new_assign, parent1.n_jobs, parent1.n_machines, processing_times)
    return child

def mutate(solution):
    """变异操作"""
    # 随机选择一种变异
    mutation_type = random.choice(['swap', 'reassign'])
    
    if mutation_type == 'swap':
        # 交换两个操作的顺序
        seq_len = len(solution.get_operation_sequence())
        if seq_len >= 2:
            pos1 = random.randint(0, seq_len - 1)
            pos2 = random.randint(0, seq_len - 1)
            solution.swap_operations(pos1, pos2)
    else:
        # 重新分配机器
        machine_assign = solution.get_machine_assignment()
        if machine_assign:
            job_id, op_idx = random.choice(list(machine_assign.keys()))
            available_machines = solution.get_available_machines(job_id, op_idx)
            if len(available_machines) > 1:
                current_machine = machine_assign[(job_id, op_idx)]
                available_machines.remove(current_machine)
                new_machine = random.choice(available_machines)
                solution.reassign_machine(job_id, op_idx, new_machine)
    
    return solution