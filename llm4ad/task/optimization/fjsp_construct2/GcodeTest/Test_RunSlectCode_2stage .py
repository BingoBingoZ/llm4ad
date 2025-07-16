import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
import math
from collections import defaultdict

# 添加项目根目录到系统路径以便导入模块
sys.path.append('/home/zgb/llm4ad')

from load_data import load_fjsp_data
from llm4ad.task.optimization.fjsp_construct2.evaluation import FJSPEvaluation

# 导入EOH生成的最佳代码
sys.path.append('/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2')
from best_code import determine_next_operation

def schedule_fjsp_instance(processing_times, n_jobs, n_machines):
    """使用构造式算法调度FJSP实例"""
    # 初始化状态
    machine_status = [0] * n_machines
    job_status = [0] * n_jobs
    operation_sequence = [[] for _ in range(n_jobs)]
    job_next_op = [0] * n_jobs
    
    # 计算总工序数
    n_ops = [len(job_ops) for job_ops in processing_times]
    total_ops = sum(n_ops)
    scheduled_ops = 0
    
    # 调度所有工序
    while scheduled_ops < total_ops:
        # 收集可行操作
        feasible_operations = []
        for job_id in range(n_jobs):
            op_idx = job_next_op[job_id]
            if op_idx < n_ops[job_id]:
                machine_id_list, processing_time_list = processing_times[job_id][op_idx]
                feasible_operations.append((job_id, machine_id_list, processing_time_list))
        
        if not feasible_operations:
            print("警告: 无可行操作但仍有未调度的工序")
            break
        
        # 确定下一步调度的操作
        job_id, best_machine, best_processing_time = determine_next_operation(
            {'machine_status': machine_status, 'job_status': job_status}, 
            feasible_operations
        )
        
        if job_id is None:
            # 如果算法无法确定下一个操作，前进到最早可用时间点
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
        
        # 安排操作
        start_time = max(job_status[job_id], machine_status[best_machine])
        end_time = start_time + best_processing_time
        machine_status[best_machine] = end_time
        job_status[job_id] = end_time
        operation_sequence[job_id].append((best_machine, start_time, end_time))
        
        # 更新下一工序索引
        job_next_op[job_id] += 1
        scheduled_ops += 1

    # 计算完工时间
    makespan = max(job_status)
    
    return makespan, operation_sequence

def calculate_makespan(schedule):
    """计算调度方案的完工时间"""
    max_end_time = 0
    for job in schedule:
        if job:  # 检查作业列表非空
            max_job_end = max(op[2] for op in job)
            max_end_time = max(max_end_time, max_job_end)
    return max_end_time

def is_valid_schedule(schedule, n_jobs, n_machines):
    """检查调度是否有效"""
    # 1. 检查作业内部工序顺序约束
    for job_id in range(n_jobs):
        for op_idx in range(len(schedule[job_id]) - 1):
            if schedule[job_id][op_idx][2] > schedule[job_id][op_idx+1][1]:
                return False
    
    # 2. 检查机器资源冲突
    for machine_id in range(n_machines):
        ops_on_machine = []
        for job_id in range(n_jobs):
            for op_idx, (m_id, start, end) in enumerate(schedule[job_id]):
                if m_id == machine_id:
                    ops_on_machine.append((start, end, job_id, op_idx))
        
        # 按开始时间排序
        ops_on_machine.sort()
        
        # 检查重叠
        for i in range(len(ops_on_machine) - 1):
            if ops_on_machine[i][1] > ops_on_machine[i+1][0]:
                return False  # 发现重叠
    
    return True

def build_operation_graph(schedule, n_jobs, n_machines):
    """
    构建操作依赖图，用于关键路径分析
    返回节点列表和邻接表
    """
    # 为每个操作创建节点，格式为(job_id, op_idx)
    nodes = []
    for job_id in range(n_jobs):
        for op_idx in range(len(schedule[job_id])):
            nodes.append((job_id, op_idx))
    
    # 虚拟起点和终点
    START_NODE = (-1, -1)
    END_NODE = (-2, -2)
    nodes.append(START_NODE)
    nodes.append(END_NODE)
    
    # 创建邻接表表示有向图
    graph = {node: [] for node in nodes}
    
    # 添加工序约束边 (同一作业的操作顺序)
    for job_id in range(n_jobs):
        if not schedule[job_id]:
            continue
            
        # 连接虚拟起点到每个作业的第一个操作
        first_node = (job_id, 0)
        graph[START_NODE].append((first_node, 0))
        
        for op_idx in range(len(schedule[job_id]) - 1):
            current_node = (job_id, op_idx)
            next_node = (job_id, op_idx + 1)
            # 边的权重是当前操作的处理时间
            weight = schedule[job_id][op_idx][2] - schedule[job_id][op_idx][1]
            graph[current_node].append((next_node, weight))
        
        # 连接每个作业的最后一个操作到虚拟终点
        last_op_idx = len(schedule[job_id]) - 1
        last_node = (job_id, last_op_idx)
        last_weight = schedule[job_id][last_op_idx][2] - schedule[job_id][last_op_idx][1]
        graph[last_node].append((END_NODE, last_weight))
    
    # 添加机器约束边 (同一机器上的操作顺序)
    for machine_id in range(n_machines):
        # 收集该机器上的所有操作
        ops_on_machine = []
        for job_id in range(n_jobs):
            for op_idx, (m_id, start, end) in enumerate(schedule[job_id]):
                if m_id == machine_id:
                    ops_on_machine.append((job_id, op_idx, start, end))
        
        # 按开始时间排序
        ops_on_machine.sort(key=lambda x: x[2])
        
        # 添加相邻操作间的边
        for i in range(len(ops_on_machine) - 1):
            job_i, op_i, _, end_i = ops_on_machine[i]
            job_j, op_j, start_j, _ = ops_on_machine[i+1]
            
            from_node = (job_i, op_i)
            to_node = (job_j, op_j)
            
            # 边的权重是前一个操作到后一个操作的间隔时间
            weight = start_j - end_i
            
            # 避免添加重复边 (不同于工序约束)
            if (job_i != job_j) or (op_i + 1 != op_j):
                graph[from_node].append((to_node, weight))
    
    return nodes, graph, START_NODE, END_NODE

def identify_critical_path(schedule, n_jobs, n_machines):
    """
    精确识别关键路径
    返回关键路径操作列表及其关键程度
    """
    nodes, graph, start_node, end_node = build_operation_graph(schedule, n_jobs, n_machines)
    
    # 计算每个节点的最早完成时间 (forward pass)
    earliest_finish = {node: float('-inf') for node in nodes}
    earliest_finish[start_node] = 0
    
    # 拓扑排序
    topo_order = []
    visited = set()
    
    def dfs(node):
        visited.add(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        topo_order.append(node)
    
    # 运行DFS获取拓扑排序
    dfs(start_node)
    topo_order.reverse()
    
    # 计算最早完成时间
    for node in topo_order:
        if node == start_node:
            earliest_finish[node] = 0
        else:
            for pred, weight in [(n, w) for n in nodes for neighbor, w in graph[n] if neighbor == node]:
                earliest_finish[node] = max(earliest_finish[node], earliest_finish[pred] + weight)
    
    # 最晚完成时间 (backward pass)
    latest_finish = {node: float('inf') for node in nodes}
    latest_finish[end_node] = earliest_finish[end_node]
    
    # 反向拓扑排序
    for node in reversed(topo_order):
        if node == end_node:
            continue
        for neighbor, weight in graph[node]:
            latest_finish[node] = min(latest_finish[node], latest_finish[neighbor] - weight)
    
    # 计算每个操作的松弛时间和关键程度
    # 松弛时间为0的操作在关键路径上
    critical_ops = []
    criticality = {}
    
    for node in nodes:
        if node == start_node or node == end_node:
            continue
            
        job_id, op_idx = node
        slack = latest_finish[node] - earliest_finish[node]
        criticality[node] = 1.0 / (1.0 + slack) if slack > 0 else 1.0
        
        if abs(slack) < 0.001:  # 松弛时间约等于0的操作
            critical_ops.append(node)
    
    return critical_ops, criticality

def rebuild_schedule(schedule, n_jobs, n_machines):
    """增强版调度修复算法"""
    temp_schedule = copy.deepcopy(schedule)
    
    # 第一阶段：修复作业内部顺序约束
    for job_id in range(n_jobs):
        for op_idx in range(1, len(temp_schedule[job_id])):
            prev_end = temp_schedule[job_id][op_idx-1][2]
            machine_id, start, end = temp_schedule[job_id][op_idx]
            duration = end - start
            
            if start < prev_end:
                # 更新开始和结束时间
                temp_schedule[job_id][op_idx] = (machine_id, prev_end, prev_end + duration)
    
    # 第二阶段：修复机器冲突 - 使用更智能的方法处理
    # 为每台机器创建时间线
    machine_timelines = [[] for _ in range(n_machines)]
    
    # 收集所有操作并按作业优先级排序
    all_operations = []
    for job_id in range(n_jobs):
        for op_idx, (machine, start, end) in enumerate(temp_schedule[job_id]):
            all_operations.append((job_id, op_idx, machine, start, end))
    
    # 按开始时间排序操作
    all_operations.sort(key=lambda x: x[3])
    
    # 清空调度，重新构建
    rebuilt_schedule = [[] for _ in range(n_jobs)]
    machine_end_times = [0] * n_machines
    job_end_times = [0] * n_jobs
    
    # 重新插入每个操作
    for job_id, op_idx, machine, _, duration in all_operations:
        # 获取实际处理时间
        if isinstance(duration, tuple):  # 如果保存的是(开始时间,结束时间)
            proc_time = duration[1] - duration[0]
        else:  # 如果保存的是结束时间
            proc_time = duration - temp_schedule[job_id][op_idx][1]
        
        # 计算最早可行的开始时间
        earliest_start = max(job_end_times[job_id], machine_end_times[machine])
        
        # 安排操作
        end_time = earliest_start + proc_time
        rebuilt_schedule[job_id].append((machine, earliest_start, end_time))
        
        # 更新时间状态
        machine_end_times[machine] = end_time
        job_end_times[job_id] = end_time
    
    # 确保重建的调度包含所有原始操作
    for job_id in range(n_jobs):
        if len(rebuilt_schedule[job_id]) != len(temp_schedule[job_id]):
            # 如果数量不匹配，放弃重建，使用简单修复
            return simple_repair_schedule(schedule, n_jobs, n_machines)
    
    return rebuilt_schedule

def simple_repair_schedule(schedule, n_jobs, n_machines):
    """简单的调度修复算法，作为后备"""
    temp_schedule = copy.deepcopy(schedule)
    
    # 1. 修复作业内部顺序约束
    for job_id in range(n_jobs):
        for op_idx in range(1, len(temp_schedule[job_id])):
            prev_end = temp_schedule[job_id][op_idx-1][2]
            machine_id, start, end = temp_schedule[job_id][op_idx]
            duration = end - start
            
            if start < prev_end:
                # 更新开始和结束时间
                temp_schedule[job_id][op_idx] = (machine_id, prev_end, prev_end + duration)
    
    # 2. 修复机器冲突
    for machine_id in range(n_machines):
        ops_on_machine = []
        for job_id in range(n_jobs):
            for op_idx, (m_id, start, end) in enumerate(temp_schedule[job_id]):
                if m_id == machine_id:
                    ops_on_machine.append((start, end, job_id, op_idx))
        
        if not ops_on_machine:
            continue
            
        # 按开始时间排序
        ops_on_machine.sort()
        
        # 解决重叠
        for i in range(1, len(ops_on_machine)):
            _, prev_end, _, _ = ops_on_machine[i-1]
            curr_start, curr_end, curr_job, curr_op = ops_on_machine[i]
            
            if curr_start < prev_end:
                # 存在重叠，移动当前操作
                duration = curr_end - curr_start
                new_start = prev_end
                new_end = new_start + duration
                
                # 更新操作时间
                temp_schedule[curr_job][curr_op] = (machine_id, new_start, new_end)
                ops_on_machine[i] = (new_start, new_end, curr_job, curr_op)
                
                # 更新后续操作
                for next_op in range(curr_op + 1, len(temp_schedule[curr_job])):
                    next_m, next_s, next_e = temp_schedule[curr_job][next_op]
                    next_duration = next_e - next_s
                    next_new_start = max(next_s, new_end)
                    temp_schedule[curr_job][next_op] = (next_m, next_new_start, next_new_start + next_duration)
    
    return temp_schedule

def enhanced_local_search(schedule, processing_times, n_jobs, n_machines, max_iterations=1000, time_limit=60):
    """增强版局部搜索优化器"""
    start_time = time.time()
    
    current_schedule = copy.deepcopy(schedule)
    best_schedule = copy.deepcopy(schedule)
    
    current_makespan = calculate_makespan(current_schedule)
    best_makespan = current_makespan
    
    print(f"开始增强型局部搜索优化，初始makespan: {current_makespan}")
    
    # 设置模拟退火参数
    initial_temp = 100.0  # 初始温度
    final_temp = 0.1      # 最终温度
    alpha = 0.98        # 冷却系数
    
    # 统计数据
    total_improved = 0
    total_accepted = 0
    no_improve_count = 0
    iterations_per_temp = 20  # 每个温度的迭代次数
    
    # 当前温度
    temp = initial_temp
    
    # 禁忌搜索相关
    tabu_list = {}
    tabu_tenure = 7  # 禁忌期限
    
    # 机器负载统计
    machine_loads = [0] * n_machines
    for job_id in range(n_jobs):
        for machine, start, end in current_schedule[job_id]:
            machine_loads[machine] += (end - start)
    
    # 瓶颈机器识别 - 修复为Python列表
    bottleneck_machines = np.argsort(machine_loads)[-3:].tolist()  # 取负载最高的三台机器
    
    # 局部搜索历史记录
    move_history = {
        'machine_reassign': {'attempts': 0, 'successes': 0},
        'swap_operations': {'attempts': 0, 'successes': 0},
        'critical_path': {'attempts': 0, 'successes': 0},
        'load_balancing': {'attempts': 0, 'successes': 0},
        'execution_time_perturbation': {'attempts': 0, 'successes': 0}  # 新增扰动类型
    }
    
    # 多样化邻域结构
    def generate_neighborhood(schedule, move_type):
        """生成多样化的邻域结构"""
        if move_type == 'machine_reassign':
            return machine_reassignment_move(schedule)
        elif move_type == 'swap_operations':
            return swap_operations_move(schedule)
        elif move_type == 'critical_path':
            return critical_path_move(schedule)
        elif move_type == 'load_balancing':
            return load_balancing_move(schedule)
        elif move_type == 'execution_time_perturbation':
            return execution_time_perturbation(schedule)
        return None
    def find_critical_path_fjsp(schedule, n_jobs, n_machines):
        """找出关键路径和关键块"""
        critical_ops, criticality = identify_critical_path(schedule, n_jobs, n_machines)
        machine_operations = [[] for _ in range(n_machines)]
        for job_id in range(n_jobs):
            for op_idx, (machine, start, end) in enumerate(schedule[job_id]):
                machine_operations[machine].append((job_id, op_idx, start, end))
        for m in range(n_machines):
            machine_operations[m].sort(key=lambda x: x[2])
        critical_blocks = []
        for machine_id in range(n_machines):
            critical_on_machine = []
            for job_id, op_idx, start, end in machine_operations[machine_id]:
                if (job_id, op_idx) in critical_ops:
                    critical_on_machine.append((job_id, op_idx, start, end))
            if len(critical_on_machine) > 0:
                current_block = [critical_on_machine[0][:2]]
                for i in range(1, len(critical_on_machine)):
                    current = critical_on_machine[i]
                    previous = critical_on_machine[i-1]
                    if abs(current[2] - previous[3]) < 0.001:
                        current_block.append(current[:2])
                    else:
                        if len(current_block) > 0:
                            critical_blocks.append({'machine_id': machine_id, 'B': current_block})
                        current_block = [current[:2]]
                if len(current_block) > 0:
                    critical_blocks.append({'machine_id': machine_id, 'B': current_block})
        return critical_ops, critical_blocks, len(critical_blocks)
    
    def machine_reassignment_move(schedule):
        """机器重分配移动"""
        move_history['machine_reassign']['attempts'] += 1
        
        # 识别关键路径
        critical_ops, criticality = identify_critical_path(schedule, n_jobs, n_machines)
        
        # 选择操作 - 70%概率选择关键操作，30%随机选择
        if critical_ops and random.random() < 0.7:
            job_id, op_idx = random.choice(critical_ops)
        else:
            # 完全随机选择
            job_id = random.randrange(n_jobs)
            if not schedule[job_id]:
                return None
            op_idx = random.randrange(len(schedule[job_id]))
        
        # 获取当前机器和可选机器
        current_machine, _, _ = schedule[job_id][op_idx]
        machine_id_list, proc_time_list = processing_times[job_id][op_idx]
        
        # 寻找替代机器 - 优先考虑负载较低的机器
        alt_machines = [m for m in machine_id_list if m != current_machine]
        if not alt_machines:
            return None
            
        # 按照机器负载排序替代机器（优先选择负载低的）
        alt_machines_with_load = [(m, machine_loads[m], proc_time_list[machine_id_list.index(m)]) 
                                 for m in alt_machines]
        # 综合考虑负载和处理时间
        alt_machines_with_load.sort(key=lambda x: x[1] * x[2])
        
        # 选择前70%的机器中随机一个，平衡探索和利用
        selection_size = max(1, int(len(alt_machines_with_load) * 0.7))
        m_idx = random.randrange(selection_size)
        new_machine, _, new_proc_time = alt_machines_with_load[m_idx]
        
        # 创建邻居解
        neighbor = copy.deepcopy(schedule)
        
        # 计算新的开始时间
        earliest_start = 0
        if op_idx > 0:
            earliest_start = neighbor[job_id][op_idx-1][2]
        
        # 重新分配
        neighbor[job_id][op_idx] = (new_machine, earliest_start, earliest_start + new_proc_time)
        
        # 更新后续操作
        for next_idx in range(op_idx + 1, len(neighbor[job_id])):
            m, s, e = neighbor[job_id][next_idx]
            duration = e - s
            new_s = max(s, neighbor[job_id][next_idx-1][2])
            neighbor[job_id][next_idx] = (m, new_s, new_s + duration)
        
        return neighbor
    
    def swap_operations_move(schedule):
        """操作交换移动 - 增强版：可交换非相邻操作"""
        move_history['swap_operations']['attempts'] += 1
        
        """跨作业交换：随机选两个作业的操作交换机器和时间"""
        move_history['two_job_swap'] = move_history.get('two_job_swap', {'attempts': 0, 'successes': 0})
        move_history['two_job_swap']['attempts'] += 1

        neighbor = copy.deepcopy(schedule)
        job_ids = random.sample(range(n_jobs), 2)
        job1, job2 = job_ids
        if not neighbor[job1] or not neighbor[job2]:
            return None
        op1 = random.randrange(len(neighbor[job1]))
        op2 = random.randrange(len(neighbor[job2]))
        # 交换机器和时间
        m1, s1, e1 = neighbor[job1][op1]
        m2, s2, e2 = neighbor[job2][op2]
        neighbor[job1][op1] = (m2, s1, s1 + (e2 - s2))
        neighbor[job2][op2] = (m1, s2, s2 + (e1 - s1))
        # 修正后续操作时间
        for idx in range(op1 + 1, len(neighbor[job1])):
            m, s, e = neighbor[job1][idx]
            duration = e - s
            new_start = neighbor[job1][idx-1][2]
            neighbor[job1][idx] = (m, new_start, new_start + duration)
        for idx in range(op2 + 1, len(neighbor[job2])):
            m, s, e = neighbor[job2][idx]
            duration = e - s
            new_start = neighbor[job2][idx-1][2]
            neighbor[job2][idx] = (m, new_start, new_start + duration)
        return neighbor
    
    def critical_path_move(schedule):
        """关键路径优化移动"""
        move_history['critical_path']['attempts'] += 1
        
        # 精确识别关键路径
        critical_ops, criticality = identify_critical_path(schedule, n_jobs, n_machines)
        if not critical_ops:
            return None
            
        # 按关键程度排序，选择更关键的操作
        critical_ops_with_score = [(job_id, op_idx, criticality.get((job_id, op_idx), 0)) 
                                  for job_id, op_idx in critical_ops]
        critical_ops_with_score.sort(key=lambda x: x[2], reverse=True)
        
        # 选择前30%的关键操作中的随机一个
        selection_size = max(1, int(len(critical_ops_with_score) * 0.3))
        selected_idx = random.randrange(selection_size)
        job_id, op_idx, _ = critical_ops_with_score[selected_idx]
        
        current_machine, _, _ = schedule[job_id][op_idx]
        
        # 获取可用机器
        machine_id_list, proc_time_list = processing_times[job_id][op_idx]
        alt_machines = [m for m in machine_id_list if m != current_machine]
        
        if not alt_machines:
            return None
            
        # 构建机器评分 - 综合考虑处理时间和负载
        machine_scores = []
        for m in alt_machines:
            idx = machine_id_list.index(m)
            proc_time = proc_time_list[idx]
            load = machine_loads[m]
            
            # 平衡考虑处理时间和负载
            score = proc_time * (1 + load / sum(machine_loads))
            machine_scores.append((m, score, proc_time))
        
        # 选择评分最低的机器
        machine_scores.sort(key=lambda x: x[1])
        best_machine, _, best_time = machine_scores[0]
        
        # 创建邻居解
        neighbor = copy.deepcopy(schedule)
        
        # 计算新的开始时间
        earliest_start = 0
        if op_idx > 0:
            earliest_start = neighbor[job_id][op_idx-1][2]
        
        # 重新分配到最佳机器
        neighbor[job_id][op_idx] = (best_machine, earliest_start, earliest_start + best_time)
        
        # 更新后续操作
        for next_idx in range(op_idx + 1, len(neighbor[job_id])):
            m, s, e = neighbor[job_id][next_idx]
            duration = e - s
            new_s = max(s, neighbor[job_id][next_idx-1][2])
            neighbor[job_id][next_idx] = (m, new_s, new_s + duration)
        
        return neighbor
    
    
    def load_balancing_move(schedule):
        """负载平衡移动 - 从高负载机器移动操作到低负载机器"""
        move_history['load_balancing']['attempts'] += 1
        
        # 计算每台机器的当前负载
        current_loads = [0] * n_machines
        for job_id in range(n_jobs):
            for machine, start, end in schedule[job_id]:
                current_loads[machine] += (end - start)
        
        # 找出负载最高和最低的机器
        high_load_idx = np.argmax(current_loads)
        low_load_idx = np.argmin(current_loads)
        
        # 如果负载已经很平衡，返回None
        if high_load_idx == low_load_idx or current_loads[high_load_idx] < current_loads[low_load_idx] * 1.2:
            return None
            
        # 从高负载机器上找出可转移的操作
        transferable_ops = []
        for job_id in range(n_jobs):
            for op_idx, (machine, start, end) in enumerate(schedule[job_id]):
                if machine == high_load_idx:
                    # 检查是否可以在低负载机器上执行
                    machine_id_list, _ = processing_times[job_id][op_idx]
                    if low_load_idx in machine_id_list:
                        transferable_ops.append((job_id, op_idx, end - start))
        
        if not transferable_ops:
            return None
            
        # 按处理时间排序，优先选择处理时间较长的操作
        transferable_ops.sort(key=lambda x: x[2], reverse=True)
        
        # 选择前30%中的随机一个
        selection_size = max(1, int(len(transferable_ops) * 0.3))
        selected_idx = random.randrange(selection_size)
        job_id, op_idx, _ = transferable_ops[selected_idx]
        
        # 创建邻居解
        neighbor = copy.deepcopy(schedule)
        
        # 获取在低负载机器上的处理时间
        machine_id_list, proc_time_list = processing_times[job_id][op_idx]
        target_idx = machine_id_list.index(low_load_idx)
        target_proc_time = proc_time_list[target_idx]
        
        # 计算新的开始时间
        earliest_start = 0
        if op_idx > 0:
            earliest_start = neighbor[job_id][op_idx-1][2]
        
        # 重新分配
        neighbor[job_id][op_idx] = (low_load_idx, earliest_start, earliest_start + target_proc_time)
        
        # 更新后续操作
        for next_idx in range(op_idx + 1, len(neighbor[job_id])):
            m, s, e = neighbor[job_id][next_idx]
            duration = e - s
            new_s = max(s, neighbor[job_id][next_idx-1][2])
            neighbor[job_id][next_idx] = (m, new_s, new_s + duration)
        
        return neighbor
    def execution_time_perturbation(schedule):
        """基于执行时间的扰动策略
        
        选择执行时间较长的作业，增加其处理时间，重新生成调度方案
        """
        move_history['execution_time_perturbation']['attempts'] += 1
        
        # 计算每个作业的总执行时间
        job_execution_times = []
        for job_id in range(n_jobs):
            total_time = sum(end - start for _, start, end in schedule[job_id])
            job_execution_times.append((job_id, total_time))
        
        # 按执行时间排序，选择执行时间较长的前50%作业
        job_execution_times.sort(key=lambda x: x[1], reverse=True)
        top_jobs = job_execution_times[:n_jobs//2]
        
        # 从这些作业中随机选择25%进行扰动
        num_jobs_to_perturb = max(1, n_jobs//4)
        if len(top_jobs) > num_jobs_to_perturb:
            perturb_candidates = random.sample(top_jobs, num_jobs_to_perturb)
            perturbed_jobs = [job_id for job_id, _ in perturb_candidates]
        else:
            perturbed_jobs = [job_id for job_id, _ in top_jobs]
        
        # 创建处理时间的深拷贝
        perturbed_processing_times = copy.deepcopy(processing_times)
        
        # 扰动百分比 - 增加10%的处理时间
        perturb_percentage = 0.1
        
        # 对选定作业的处理时间增加固定百分比
        for job_id in perturbed_jobs:
            for op_idx in range(len(perturbed_processing_times[job_id])):
                machine_id_list, proc_time_list = perturbed_processing_times[job_id][op_idx]
                # 对每个可行机器的处理时间增加固定百分比
                for i in range(len(proc_time_list)):
                    # 将浮点数四舍五入为整数
                    proc_time_list[i] = round(proc_time_list[i] * (1 + perturb_percentage))
        
        # 使用构造式算法基于扰动后的处理时间重新生成调度
        _, perturbed_schedule = schedule_fjsp_instance(perturbed_processing_times, n_jobs, n_machines)
        
        # 输出扰动信息
        # print(f"执行时间扰动: 扰动了{len(perturbed_jobs)}个作业，增加了{perturb_percentage*100}%的处理时间")
        
        return perturbed_schedule
    # 主循环
    iteration = 0
    while iteration < max_iterations:
        # 检查时间限制
        current_time = time.time()
        if current_time - start_time > time_limit:
            print(f"时间限制到达，停止优化。已完成 {iteration} 次迭代。")
            break
            
        improved_in_temp = False
        
        # 对每个温度执行多次迭代
        for temp_iter in range(iterations_per_temp):
            iteration += 1
            if iteration > max_iterations:
                break
            
            # 根据当前搜索阶段调整移动类型概率
            if temp > initial_temp * 0.7:  # 初始阶段 - 更多探索
                move_probs = [
                    ('machine_reassign', 0.2), 
                    ('swap_operations', 0.2), 
                    ('critical_path', 0.2), 
                    ('load_balancing', 0.2),
                    ('execution_time_perturbation', 0.2)
                ]
            elif temp > initial_temp * 0.3:  # 中期阶段 - 平衡探索和利用
                # 根据历史成功率动态调整
                total_successes = sum(m['successes'] for m in move_history.values())
                if total_successes > 0:
                    move_probs = []
                    for move_name, stats in move_history.items():
                        success_rate = stats['successes'] / max(1, stats['attempts'])
                        # 平衡探索和利用 - 提高成功率高的移动类型的概率
                        prob = 0.1 + 0.4 * success_rate  # 基础概率0.1，最高可增加0.4
                        move_probs.append((move_name, prob))
                    # 归一化概率
                    total_prob = sum(p for _, p in move_probs)
                    move_probs = [(name, p/total_prob) for name, p in move_probs]
                else:
                    move_probs = [
                        ('machine_reassign', 0.22), 
                        ('swap_operations', 0.22), 
                        ('critical_path', 0.22), 
                        ('load_balancing', 0.22),
                        ('execution_time_perturbation', 0.22)
                    ]
            else:  # 后期阶段 - 更多利用
                # 聚焦于历史上最成功的移动类型
                best_moves = sorted(move_history.items(), 
                                key=lambda x: x[1]['successes'] / max(1, x[1]['attempts']), 
                                reverse=True)
                # 给最好的两种移动类型更高的概率
                move_probs = [(name, 0.3 if i < 2 else 0.1 if name == 'execution_time_perturbation' else 0.15) 
                            for i, (name, _) in enumerate(best_moves)]
                # 归一化概率
                total_prob = sum(p for _, p in move_probs)
                move_probs = [(name, p/total_prob) for name, p in move_probs]
            
            # 随机选择移动类型
            rand_val = random.random()
            cumulative_prob = 0
            selected_move = move_probs[0][0]  # 默认第一个
            
            for move_name, prob in move_probs:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_move = move_name
                    break
            
            # 生成邻居解
            neighbor = generate_neighborhood(current_schedule, selected_move)
            
            if neighbor is None:
                continue
                
            # 修复调度以确保有效性
            try:
                neighbor = rebuild_schedule(neighbor, n_jobs, n_machines)
                
                if not is_valid_schedule(neighbor, n_jobs, n_machines):
                    # 如果高级修复失败，尝试简单修复
                    neighbor = simple_repair_schedule(neighbor, n_jobs, n_machines)
                    
                    if not is_valid_schedule(neighbor, n_jobs, n_machines):
                        continue  # 跳过无效的邻居
            except Exception as e:
                print(f"修复调度出错: {str(e)}")
                continue
            
            # 计算新的makespan
            neighbor_makespan = calculate_makespan(neighbor)
            
            # 计算能量差
            delta = neighbor_makespan - current_makespan
            
            # 接受准则 - 模拟退火
            accept = False
            if delta < 0:  # 更好的解，直接接受
                accept = True
                total_improved += 1
                move_history[selected_move]['successes'] += 1
                
                if neighbor_makespan < best_makespan:
                    best_makespan = neighbor_makespan
                    best_schedule = copy.deepcopy(neighbor)
                    print(f"迭代 {iteration}: 找到更好的解，makespan = {best_makespan}，移动类型 = {selected_move}")
                    improved_in_temp = True
                    no_improve_count = 0
            else:  # 较差的解，以一定概率接受
                acceptance_prob = math.exp(-delta / temp)
                if random.random() < acceptance_prob:
                    accept = True
            
            # 更新当前解
            if accept:
                current_schedule = copy.deepcopy(neighbor)
                current_makespan = neighbor_makespan
                total_accepted += 1
                
                # 更新机器负载
                machine_loads = [0] * n_machines
                for job_id in range(n_jobs):
                    for machine, start, end in current_schedule[job_id]:
                        machine_loads[machine] += (end - start)
                
                # 更新瓶颈机器 - 修复为Python列表
                bottleneck_machines = np.argsort(machine_loads)[-3:].tolist()
            else:
                no_improve_count += 1
            
            # 降低禁忌期限
            tabu_keys = list(tabu_list.keys())
            for key in tabu_keys:
                tabu_list[key] -= 1
                if tabu_list[key] <= 0:
                    del tabu_list[key]
            
            # 自适应禁忌期限
            if total_accepted > 0 and iteration % 50 == 0:
                accept_rate = total_accepted / iteration
                if accept_rate > 0.6:  # 接受率过高，增加禁忌期限
                    tabu_tenure = min(15, tabu_tenure + 1)
                elif accept_rate < 0.3:  # 接受率过低，减少禁忌期限
                    tabu_tenure = max(3, tabu_tenure - 1)
        
        # 每个温度结束后，降温
        temp *= alpha
        
        # 如果连续多次无改进，执行多样化重启
        if no_improve_count >= 30:
            # print(f"迭代 {iteration}: 执行多样化重启")
            
            # 1. 识别当前最佳解中的关键操作
            critical_ops, _ = identify_critical_path(best_schedule, n_jobs, n_machines)
            
            # 2. 创建扰动解 - 使用最佳解并随机重分配30%的关键操作
            restart_schedule = copy.deepcopy(best_schedule)
            
            # 如果没有识别到关键操作，随机选择操作
            if not critical_ops:
                all_ops = []
                for j_id in range(n_jobs):
                    for o_idx in range(len(restart_schedule[j_id])):
                        all_ops.append((j_id, o_idx))
                
                n_ops_to_perturb = max(1, int(len(all_ops) * 0.3))
                ops_to_perturb = random.sample(all_ops, n_ops_to_perturb)
            else:
                # 随机选择30%的关键操作
                n_ops_to_perturb = max(1, int(len(critical_ops) * 0.3))
                ops_to_perturb = random.sample(critical_ops, n_ops_to_perturb)
            
            # 3. 强力扰动 - 不仅重分配机器，还随机调整顺序
            for j_id, o_idx in ops_to_perturb:
                current_m, _, _ = restart_schedule[j_id][o_idx]
                machine_list, time_list = processing_times[j_id][o_idx]
                
                # 尝试选择不同的机器
                alt_machines = [m for m in machine_list if m != current_m]
                if alt_machines:
                    new_m = random.choice(alt_machines)
                    new_time = time_list[machine_list.index(new_m)]
                    
                    # 随机调整开始时间 - 引入一些随机性
                    base_start = 0
                    if o_idx > 0:
                        base_start = restart_schedule[j_id][o_idx-1][2]
                    
                    # 添加一些随机扰动，但确保不会早于前序操作
                    random_offset = random.randint(0, 20)
                    earliest_s = base_start + random_offset
                    
                    # 重新分配
                    restart_schedule[j_id][o_idx] = (new_m, earliest_s, earliest_s + new_time)
                    
                    # 更新后续操作
                    for next_idx in range(o_idx + 1, len(restart_schedule[j_id])):
                        m, s, e = restart_schedule[j_id][next_idx]
                        duration = e - s
                        new_s = max(s, restart_schedule[j_id][next_idx-1][2])
                        restart_schedule[j_id][next_idx] = (m, new_s, new_s + duration)
            
            # 4. 修复调度
            restart_schedule = rebuild_schedule(restart_schedule, n_jobs, n_machines)
            
            if is_valid_schedule(restart_schedule, n_jobs, n_machines):
                current_schedule = restart_schedule
                current_makespan = calculate_makespan(current_schedule)
                
                # 如果扰动产生了更好的解，更新最佳解
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_schedule = copy.deepcopy(current_schedule)
                    print(f"迭代 {iteration}: 重启找到更好的解，makespan = {best_makespan}")
                
                # 重置无改进计数和温度
                no_improve_count = 0
                temp = initial_temp * 0.5  # 降低重启温度，加速收敛
                
                # 更新机器负载
                machine_loads = [0] * n_machines
                for job_id in range(n_jobs):
                    for machine, start, end in current_schedule[job_id]:
                        machine_loads[machine] += (end - start)
                
                # 更新瓶颈机器 - 修复为Python列表
                bottleneck_machines = np.argsort(machine_loads)[-3:].tolist()
        
        # 每100次迭代输出一次状态
        if iteration % 100 == 0:
            accept_rate = total_accepted / iteration * 100 if iteration > 0 else 0
            improve_rate = total_improved / iteration * 100 if iteration > 0 else 0
            
            # 计算各种移动类型的成功率
            move_stats = []
            for move_name, stats in move_history.items():
                success_rate = stats['successes'] / max(1, stats['attempts']) * 100
                move_stats.append(f"{move_name}: {success_rate:.1f}%")
            
            # print(f"迭代 {iteration}/{max_iterations}: 当前温度 = {temp:.2f}, 当前最佳makespan = {best_makespan}")
            # print(f"  接受率: {accept_rate:.1f}%, 改进率: {improve_rate:.1f}%")
            # print(f"  移动成功率: {', '.join(move_stats)}")
       
    # 最终检查最佳解的有效性
    if not is_valid_schedule(best_schedule, n_jobs, n_machines):
        print("警告: 最终解无效，进行修复...")
        best_schedule = rebuild_schedule(best_schedule, n_jobs, n_machines)
        
        if not is_valid_schedule(best_schedule, n_jobs, n_machines):
            best_schedule = simple_repair_schedule(best_schedule, n_jobs, n_machines)
            
        best_makespan = calculate_makespan(best_schedule)
    
    print(f"\n局部搜索完成，总迭代次数: {iteration}")
    print(f"初始makespan: {calculate_makespan(schedule)}, 优化后makespan: {best_makespan}")
    
    initial_makespan = calculate_makespan(schedule)
    if initial_makespan > best_makespan:
        improvement = (initial_makespan - best_makespan) / initial_makespan * 100
        print(f"改进率: {improvement:.2f}%")
    else:
        print("未找到更好的解决方案")
    
    # 输出各类移动的统计信息
    print("\n移动类型统计:")
    for move_name, stats in move_history.items():
        attempts = stats['attempts']
        successes = stats['successes']
        success_rate = successes / max(1, attempts) * 100
        print(f"  {move_name}: 尝试次数 = {attempts}, 成功次数 = {successes}, 成功率 = {success_rate:.2f}%")
    
    return best_schedule, best_makespan

def drawGantt(schedule, n_jobs, n_machines):
    """绘制甘特图"""
    # 按机器组织数据
    machine_schedules = [[] for _ in range(n_machines)]
    
    for i in range(n_machines):
        machine_schedules[i] = [i]  # 第一个元素是机器ID
    
    for job_idx, operations in enumerate(schedule):
        for op_idx, operation in enumerate(operations):
            machine, start_time, end_time = operation
            machine_schedules[machine].append([start_time, job_idx, op_idx, end_time])
    
    # 排序
    for machine_idx in range(n_machines):
        machine_schedules[machine_idx][1:] = sorted(machine_schedules[machine_idx][1:], key=lambda x: x[0])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 为每个作业分配颜色
    color_map = {}
    for job_idx in range(n_jobs):
        color_map[job_idx] = (random.random(), random.random(), random.random())

    # 绘制甘特图
    for machine_idx, machine_schedule in enumerate(machine_schedules):
        for task_data in machine_schedule[1:]:
            start_time, job_idx, operation_idx, end_time = task_data
            color = color_map[job_idx]

            ax.barh(machine_idx, end_time - start_time, left=start_time, height=0.5, color=color)
            label = f'{job_idx+1}-{operation_idx+1}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='white', fontsize=10)

    # 设置标签
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(n_machines)])
    plt.xlabel("Time")
    plt.title("FJSP Gantt Chart")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 添加图例
    legend_handles = []
    for job_idx in range(n_jobs):
        legend_handles.append(plt.Rectangle((0,0), 1, 1, color=color_map[job_idx], label=f'Job {job_idx+1}'))
    plt.legend(handles=legend_handles, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    filepath='/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2/GcodeTest/data_test/Public/v_la01.fjs'
    processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
    
    # 打印问题信息
    print(f"问题规模: {n_jobs}个作业, {n_machines}台机器")
    print(f"各作业工序数: {[len(job_ops) for job_ops in processing_times]}")
    total_ops = sum(len(job_ops) for job_ops in processing_times)
    print(f"总工序数: {total_ops}")
    
    # 使用构造式算法生成初始解
    start_time = time.time()
    makespan, schedule = schedule_fjsp_instance(processing_times, n_jobs, n_machines)
    construction_time = time.time() - start_time
    
    print(f"\n构造式调度结果:")
    print(f"完工时间(Makespan): {makespan}")
    print(f"计算时间: {construction_time:.4f} 秒")
    
    # 应用增强版局部搜索优化
    opt_start_time = time.time()
    improved_schedule, improved_makespan = enhanced_local_search(
        schedule, processing_times, n_jobs, n_machines, max_iterations=100000, time_limit=120
    )
    opt_time = time.time() - opt_start_time
    
    print(f"\n增强版局部搜索优化结果:")
    print(f"优化后完工时间(Makespan): {improved_makespan}")
    print(f"优化时间: {opt_time:.4f} 秒")
    print(f"总计算时间: {construction_time + opt_time:.4f} 秒")
    
    if makespan > improved_makespan:
        print(f"改进率: {(makespan - improved_makespan) / makespan * 100:.2f}%")
    else:
        print("优化未找到更好的解决方案")
    
    # 绘制甘特图
    print("\n绘制优化后的甘特图...")
    drawGantt(improved_schedule, n_jobs, n_machines)

if __name__ == "__main__":
    main()