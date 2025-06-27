# def determine_next_operation(current_status, feasible_operations):\n    \"\"\"\n    Determine the next operation to schedule for FJSP.\n    \n    Args:\n        current_status: A dictionary with machine_status and job_status\n        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples\n        \n    Returns:\n        A tuple (job_id, best_machine, best_processing_time)\n    \"\"\"\n    machine_status = current_status['machine_status']\n    job_status = current_status['job_status']\n    best_choice = None\n    projected_makespan = float('inf')\n\n    for job_id, machine_id_list, processing_time_list in feasible_operations:\n        for machine_id, processing_time in zip(machine_id_list, processing_time_list):\n            current_load = machine_status[machine_id]\n            future_makespan_impact = current_load + processing_time\n            \n            # Adjust based on a hypothetical future load by considering subsequent jobs' priorities\n            job_priority = job_status[job_id]\n            future_makespan_impact += sum(job_priority for _, _, _ in feasible_operations if job_id != _)\n\n            if future_makespan_impact < projected_makespan:\n                projected_makespan = future_makespan_impact\n                best_choice = (job_id, machine_id, processing_time)\n\n    return best_choice\n\n


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
#     best_choice = None
#     projected_makespan = float('inf')

#     for job_id, machine_id_list, processing_time_list in feasible_operations:
#         for machine_id, processing_time in zip(machine_id_list, processing_time_list):
#             current_load = machine_status[machine_id]
#             future_makespan_impact = current_load + processing_time
            
#             # Adjust based on a hypothetical future load by considering subsequent jobs' priorities
#             job_priority = job_status[job_id]
#             future_makespan_impact += sum(job_priority for _, _, _ in feasible_operations if job_id != _)

#             if future_makespan_impact < projected_makespan:
#                 projected_makespan = future_makespan_impact
#                 best_choice = (job_id, machine_id, processing_time)
#     return best_choice


# def determine_next_operation(current_status, feasible_operations):\n    \"\"\"\n    Determine the next operation to schedule for FJSP.\n    \n    Args:\n        current_status: A dictionary with machine_status and job_status\n        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples\n        \n    Returns:\n        A tuple (job_id, best_machine, best_processing_time)\n    \"\"\"\n    best_job = None\n    best_machine = None\n    best_processing_time = float('inf')\n    \n    machine_loads = current_status['machine_status']\n    job_remaining_work = current_status['job_status']\n\n    for job_id, machine_id_list, processing_time_list in feasible_operations:\n        for machine_id, processing_time in zip(machine_id_list, processing_time_list):\n            # Prioritize machines with the least load and consider processing time\n            effective_load = machine_loads[machine_id] + (processing_time * job_remaining_work[job_id])\n            if effective_load < best_processing_time:\n                best_processing_time = effective_load\n                best_job = job_id\n                best_machine = machine_id\n                \n    return (best_job, best_machine, best_processing_time)\n\n"


def determine_next_operation(current_status, feasible_operations):
    """
    Determine the next operation to schedule for FJSP.
    
    Args:
        current_status: A dictionary with machine_status and job_status
        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples
        
    Returns:
        A tuple (job_id, best_machine, best_processing_time)
    """
    best_job = None
    best_machine = None
    best_processing_time = float('inf')
    
    machine_loads = current_status['machine_status']
    job_remaining_work = current_status['job_status']

    for job_id, machine_id_list, processing_time_list in feasible_operations:
        for machine_id, processing_time in zip(machine_id_list, processing_time_list):
            # Prioritize machines with the least load and consider processing time
            effective_load = machine_loads[machine_id] + (processing_time * job_remaining_work[job_id])
            if effective_load < best_processing_time:
                best_processing_time = effective_load
                best_job = job_id
                best_machine = machine_id
                
    return (best_job, best_machine, best_processing_time)
   
   
   



