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

task_description = '''
Given jobs and machines (each operation can be processed on multiple machines with different processing times), 
schedule jobs on machines to minimize the total makespan. Design an algorithm to select the next operation and machine in each step.
'''