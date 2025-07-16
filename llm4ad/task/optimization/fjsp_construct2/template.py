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
    
    
#     # Simple greedy heuristic: choose the operation with shortest processing time
#     machine_status = current_status['machine_status']
#     job_status = current_status['job_status']
    
#     best_job_id = None
#     best_machine = None
#     best_processing_time = float('inf')

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



# task_description = '''
# Flexible Job Shop Scheduling Problem (FJSP) Description:
# - '#' means code comment.
# - We have a set of jobs, each composed of a sequence of operations that must be performed in order.
# - We have a set of machines, and each operation can be processed on one of several alternative machines.
# - Each operation has different processing times depending on which machine is used.
# - A machine can process only one operation at a time.
# - Once an operation starts on a machine, it cannot be interrupted.
# - Operations of the same job must be processed in their specified sequence.

# Your task is to design an algorithm that decides which operation to schedule next and on which machine, with the goal of minimizing the total makespan (completion time of the last job).

# Key considerations to incorporate in your algorithm:
# 1. Machine load balancing - Avoid creating bottlenecks by distributing work evenly
# 2. Processing time optimization - Consider both immediate and long-term effects of choosing fast/slow machines
# 3. Critical path identification - Prioritize operations that lie on the critical path
# 4. Queue management - Consider waiting times and potential machine idle times
# 5. Look-ahead capability - Evaluate how current decisions impact future scheduling options

# Your algorithm should be adaptive, considering the current state of all machines and jobs when making decisions, rather than following a fixed rule. The goal is to outperform simple greedy heuristics.

# Input:
# - current_status: Contains machine_status (current available time of each machine) and job_status (current available time of each job)
# - feasible_operations: List of operations that can be scheduled next, each with a job_id, list of possible machines, and corresponding processing times

# Output:
# - The job_id, machine_id, and processing_time of the selected operation
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
    
    # The algorithm selects the next operation by evaluating the potential future makespan impact of each feasible operation on all machines, incorporating both the current machine loads and the anticipated future job priorities to make a more informed scheduling decision.
    best_operation = None
    best_score = float('inf')

    for job_id, machine_id_list, processing_time_list in feasible_operations:
        job_pending_operations = current_status['job_status'][job_id]
        
        for machine_id, processing_time in zip(machine_id_list, processing_time_list):
            current_time = current_status['machine_status'][machine_id]
            finish_time = current_time + processing_time
            
            # New score function: prioritize processing time and job urgency
            # urgency_penalty = job_pending_operations * 2  # greater weight on the number of pending operations
            urgency_penalty = job_pending_operations * 2  # greater weight on the number of pending operations
            score = finish_time + urgency_penalty
            
            if score < best_score:
                best_score = score
                best_operation = (job_id, machine_id, processing_time)

    return best_operation
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