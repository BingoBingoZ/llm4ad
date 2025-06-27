# template_program = '''
# import numpy as np

# def determine_next_operation(current_status, feasible_operations):
#     """
#     Determine the next operation to schedule by combining job selection and machine assignment strategies
    
#     Args:
#         current_status: A dictionary containing the current status of machines and jobs
#         feasible_operations: A list of feasible operations that can be scheduled next
        
#     Returns:
#         The next operation to schedule, represented as a tuple (job_id, machine_id, processing_time)
#     """
#     machine_status = current_status['machine_status']
#     job_status = current_status['job_status']
    
#     # Helper function to select the best machine for a given operation
#     def select_best_machine(job_id, machine_id_list, processing_time_list):
#         best_machine_id = None
#         best_score = float('inf')
#         best_processing_time = 0
        
#         for machine_id, processing_time in zip(machine_id_list, processing_time_list):
#             start_time = max(job_status[job_id], machine_status[machine_id])
#             end_time = start_time + processing_time
            
#             idle_time = max(0, start_time - machine_status[machine_id])
#             score = processing_time + 0.5 * idle_time + 0.3 * end_time
            
#             if score < best_score:
#                 best_score = score
#                 best_machine_id = machine_id
#                 best_processing_time = processing_time
        
#         return best_machine_id, best_processing_time, best_score
    
#     # Helper function to select the next job/operation
#     def select_next_job():
#         best_job_idx = 0
#         best_score = float('inf')
        
#         for idx, op in enumerate(feasible_operations):
#             job_id, machine_id_list, processing_time_list = op
            
#             _, _, score = select_best_machine(job_id, machine_id_list, processing_time_list)
            
#             if score < best_score:
#                 best_score = score
#                 best_job_idx = idx
        
#         return best_job_idx
    
#     # Select the next job/operation
#     selected_job_idx = select_next_job()
#     job_id, machine_id_list, processing_time_list = feasible_operations[selected_job_idx]
    
#     # Select the best machine for the chosen job/operation
#     best_machine, best_processing_time, _ = select_best_machine(job_id, machine_id_list, processing_time_list)
    
#     return job_id, best_machine, best_processing_time
# '''

# task_description = '''
# Flexible Job Shop Scheduling Problem (FJSP) requires two key decisions:
# 1. Job Selection: Choose which job/operation to process next from all available ones
# 2. Machine Assignment: Assign the selected operation to the best available machine

# Develop strategies for both decisions to minimize the total makespan. Your code should implement:
# - A strategy to select the best machine for a given operation
# - A strategy to select the next job/operation to process
# - Combine both strategies to make the final scheduling decision
# '''


template_program = '''
import numpy as np

def determine_next_operation(current_status, feasible_operations):
    """
    Determine the next operation to schedule for FJSP.
    
    Args:
        current_status: A dictionary with machine_status and job_status
        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples
        
    Returns:
        A tuple (job_id, best_machine, best_processing_time)
    """
    machine_status = current_status['machine_status']
    job_status = current_status['job_status']
    
    best_job_id = None
    best_machine = None
    best_processing_time = float('inf')
    
    # Simple greedy heuristic: choose the operation with shortest processing time
    for job_id, machine_id_list, processing_time_list in feasible_operations:
        for i, (machine_id, processing_time) in enumerate(zip(machine_id_list, processing_time_list)):
            if processing_time < best_processing_time:
                best_processing_time = processing_time
                best_machine = machine_id
                best_job_id = job_id
    
    return best_job_id, best_machine, best_processing_time
'''