def determine_next_operation(current_status, feasible_operations):
    machine_loads = {}  # 记录每台机器的总负载（可选操作的处理时间之和）
    
    for op in feasible_operations:  # 遍历所有可选操作
        job_id, machine_id, processing_time = op  # 解包操作，获取作业编号、机器编号和处理时间
        if machine_id not in machine_loads:  # 如果该机器还没有记录负载
            machine_loads[machine_id] = 0  # 初始化该机器的负载为0
        machine_loads[machine_id] += processing_time  # 累加该机器的负载
    
    # 选择负载最小的机器编号
    min_load_machine = min(machine_loads, key=machine_loads.get)
    next_operation = min(
        (op for op in feasible_operations if op[1] == min_load_machine),  # 在所有分配到负载最小机器的操作中
        key=lambda op: op[2]  # 选择处理时间最短的操作
    )
    
    return next_operation