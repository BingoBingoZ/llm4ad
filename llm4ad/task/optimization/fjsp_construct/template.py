template_program = '''
import numpy as np

def determine_next_operation(current_status, feasible_operations):
    """
    Greedy scheduling algorithm for the Flexible Job-Shop Scheduling Problem (FJSP).
    Each feasible operation contains multiple candidate machines and their corresponding processing times.

    Args:
        current_status: A dictionary representing the current status of each machine and job.
        feasible_operations: A list, each element is (job_id, machine_id_list, processing_time_list)

    Returns:
        The next operation to schedule, represented as a tuple (job_id, machine_id, processing_time)
    """
    # Among all feasible operations and candidate machines, select the job-machine pair with the shortest processing time
    best = None
    min_time = float('inf')
    for op in feasible_operations:
        job_id, machine_id_list, processing_time_list = op
        for machine_id, processing_time in zip(machine_id_list, processing_time_list):
            if processing_time < min_time:
                min_time = processing_time
                best = (job_id, machine_id, processing_time)
    return best
'''

task_description = '''
Given jobs and machines (each operation can be processed on multiple machines with different processing times), schedule jobs on machines to minimize the total makespan. Design an algorithm to select the next operation and machine in each step.
'''
