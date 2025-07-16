# template_program = '''
# import random
# import copy

# def generate_neighborhood(schedule, processing_times, n_jobs, n_machines):
#     """
#     Generate a neighborhood solution for FJSP by applying a local search move.
    
#     Args:
#         schedule: Current schedule, represented as a list of operations for each job.
#                  Each operation is a tuple (machine_id, start_time, end_time).
#         processing_times: Processing time data for each job-operation-machine.
#         n_jobs: Number of jobs.
#         n_machines: Number of machines.
        
#     Returns:
#         A new schedule after applying a local search move, or None if no valid move is found.
#     """
#     # A random operation reassignment to an alternative machine while preserving schedule feasibility.
#     neighbor = copy.deepcopy(schedule)
#     job_id = random.randrange(n_jobs)
#     if not neighbor[job_id]:
#         return None
#     op_idx = random.randrange(len(neighbor[job_id]))

#     current_machine, _, _ = neighbor[job_id][op_idx]
#     machine_id_list, proc_time_list = processing_times[job_id][op_idx]
#     alt_machines = [m for m in machine_id_list if m != current_machine]
#     if not alt_machines:
#         return None

#     new_machine = random.choice(alt_machines)
#     new_proc_time = proc_time_list[machine_id_list.index(new_machine)]
#     earliest_start = 0 if op_idx == 0 else neighbor[job_id][op_idx-1][2]

#     machine_end_times = []
#     for j in range(n_jobs):
#         for m, s, e in neighbor[j]:
#             if m == new_machine:
#                 machine_end_times.append((s, e))
#     machine_end_times.sort()

#     start_time = earliest_start
#     for s, e in machine_end_times:
#         if start_time < s and start_time + new_proc_time <= s:
#             break
#         if start_time >= s and start_time < e:
#             start_time = e

#     neighbor[job_id][op_idx] = (new_machine, start_time, start_time + new_proc_time)
#     for next_idx in range(op_idx + 1, len(neighbor[job_id])):
#         m, s, e = neighbor[job_id][next_idx]
#         new_s = max(s, neighbor[job_id][next_idx-1][2])
#         neighbor[job_id][next_idx] = (m, new_s, new_s + (e - s))

#     return neighbor
# '''

# task_description = '''
# Design a neighborhood search strategy for the Flexible Job Shop Scheduling Problem (FJSP).
# In FJSP, each job consists of a sequence of operations, and each operation can be processed on one of several eligible machines with different processing times. Your task is to generate a neighborhood solution by applying a local search move to the current schedule.
# The goal is to minimize the makespan (completion time of all jobs).

# Your neighborhood search function should:
# 1. Take the current schedule, processing times, and problem dimensions as input
# 2. Apply a creative local search move (e.g., operation reordering, machine reassignment, or critical path optimization)
# 3. Return a valid new schedule with potentially improved makespan
# Be creative in designing the search strategy to efficiently explore the solution space.
# '''

template_program = '''
import numpy as np
def update_schedule(processing_times: np.ndarray, current_schedule: np.ndarray, schedule_history: np.ndarray) -> np.ndarray:
    """
    Design a novel algorithm to update the job-machine assignment schedule.

    Args:
    processing_times: A matrix where processing_times[j,m] represents the processing time of job j on machine m.
    current_schedule: A binary matrix where schedule[j,m]=1 if job j is assigned to machine m, 0 otherwise.
    schedule_history: A matrix tracking how many times each job-machine assignment has been used.

    Return:
    updated_schedule: A matrix of the updated job-machine assignments.
    """
    updated_schedule = np.copy(current_schedule)
    
    # Calculate combined importance and frequency factor
    combined_factor = (1 / (schedule_history + 1)) * processing_times
    
    # Find jobs with the highest processing times
    for job in range(processing_times.shape[0]):
        current_machine = np.argmax(current_schedule[job])
        alternative_machine = np.argmin(combined_factor[job])
        
        if alternative_machine != current_machine and processing_times[job, alternative_machine] > 0:
            # Reassign job to alternative machine
            updated_schedule[job, current_machine] = 0
            updated_schedule[job, alternative_machine] = 1
    
    return updated_schedule
'''

task_description = "Given a processing time matrix and a current schedule for a Flexible Job Shop Scheduling Problem, please help me design a strategy to update the job-machine assignments to avoid being trapped in local optima with the final goal of finding a schedule with minimized total processing time. You should create a heuristic for me to update the job assignments."


