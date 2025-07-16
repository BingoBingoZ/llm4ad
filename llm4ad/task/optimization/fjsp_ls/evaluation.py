# from __future__ import annotations
# from typing import Any, List, Tuple, Callable
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import copy

# from llm4ad.base import Evaluation
# from llm4ad.task.optimization.fjsp_ls.get_instance import GetData
# from llm4ad.task.optimization.fjsp_ls.template import template_program, task_description
# from llm4ad.task.optimization.fjsp_ls.initial_solution import generate_initial_solution

# __all__ = ['FJSPEvaluation']

# class FJSPEvaluation(Evaluation):
#     """Evaluator for Flexible Job Shop Scheduling Problem with Local Search."""

#     def __init__(self,
#                  timeout_seconds=30,
#                  n_instance=20,
#                  n_jobs=50,
#                  n_machines=15,
#                  max_iterations=400,
#                  **kwargs):
#         """
#         Initialize the FJSP Local Search Evaluation.
        
#         Args:
#             timeout_seconds: Maximum allowed time for evaluation.
#             n_instance: Number of problem instances to evaluate.
#             n_jobs: Number of jobs in each instance.
#             n_machines: Number of machines in each instance.
#             max_iterations: Maximum iterations for local search.
#         """
#         super().__init__(
#             template_program=template_program,
#             task_description=task_description,
#             use_numba_accelerate=False,
#             timeout_seconds=timeout_seconds
#         )

#         self.n_instance = n_instance
#         self.n_jobs = n_jobs
#         self.n_machines = n_machines
#         self.max_iterations = max_iterations
#         getData = GetData(self.n_instance, self.n_jobs, self.n_machines)
#         self._datasets = getData.generate_instances()
#         self._initial_solutions = {}  # Cache for initial solutions

#     def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
#         """Evaluate the neighborhood strategy implementation."""
#         return self.evaluate(callable_func)

#     def evaluate(self, neighborhood_strategy: Callable) -> float:
#         """
#         Evaluate the neighborhood search strategy on FJSP instances.
        
#         Args:
#             neighborhood_strategy: The neighborhood generation function to evaluate.
            
#         Returns:
#             The average improvement percentage across all instances.
#         """
#         improvements = []

#         for i, instance in enumerate(self._datasets[:self.n_instance]):
#             processing_times, n_jobs, n_machines = instance
            
#             # Generate or retrieve initial solution using genetic algorithm
#             if i not in self._initial_solutions:
#                 print(f"Generating initial solution for instance {i+1}...")
#                 initial_schedule, initial_makespan = generate_initial_solution(
#                     processing_times, n_jobs, n_machines
#                 )
#                 self._initial_solutions[i] = (initial_schedule, initial_makespan)
#             else:
#                 initial_schedule, initial_makespan = self._initial_solutions[i]
                
#             print(f"Instance {i+1}: Initial makespan = {initial_makespan}")
            
#             # Apply local search with the provided neighborhood strategy
#             improved_schedule, improved_makespan = self.local_search(
#                 initial_schedule, 
#                 processing_times, 
#                 n_jobs, 
#                 n_machines, 
#                 neighborhood_strategy
#             )
            
#             # Calculate improvement
#             if initial_makespan > improved_makespan:
#                 improvement = (initial_makespan - improved_makespan) / initial_makespan * 100
#                 print(f"Instance {i+1}: Improved makespan = {improved_makespan}, Improvement = {improvement:.2f}%")
#             else:
#                 improvement = 0
#                 print(f"Instance {i+1}: No improvement")
                
#             improvements.append(improvement)
            
#         average_improvement = np.mean(improvements)
#         print(f"Average improvement: {average_improvement:.2f}%")
#         return average_improvement  # Higher is better

#     def local_search(self, initial_schedule, processing_times, n_jobs, n_machines, neighborhood_strategy):
#         """
#         Perform local search using the provided neighborhood strategy.
        
#         Args:
#             initial_schedule: Initial schedule to start from.
#             processing_times: Processing time data.
#             n_jobs: Number of jobs.
#             n_machines: Number of machines.
#             neighborhood_strategy: Function that generates neighborhood solutions.
            
#         Returns:
#             The best schedule found and its makespan.
#         """
#         start_time = time.time()
#         max_time = self.timeout_seconds
        
#         current_schedule = copy.deepcopy(initial_schedule)
#         current_makespan = self.calculate_makespan(current_schedule)
        
#         best_schedule = copy.deepcopy(current_schedule)
#         best_makespan = current_makespan
        
#         iteration = 0
#         no_improve_count = 0
        
#         while iteration < self.max_iterations and time.time() - start_time < max_time:
#             # Generate neighbor
#             neighbor = neighborhood_strategy(current_schedule, processing_times, n_jobs, n_machines)
            
#             if neighbor is None:
#                 iteration += 1
#                 continue
                
#             # Evaluate neighbor
#             neighbor_makespan = self.calculate_makespan(neighbor)
            
#             # Accept if better
#             if neighbor_makespan < current_makespan:
#                 current_schedule = copy.deepcopy(neighbor)
#                 current_makespan = neighbor_makespan
#                 no_improve_count = 0
                
#                 # Update best
#                 if current_makespan < best_makespan:
#                     best_schedule = copy.deepcopy(current_schedule)
#                     best_makespan = current_makespan
#             else:
#                 no_improve_count += 1
            
#             # Restart if stuck
#             if no_improve_count >= 50:
#                 # Perturb the current solution
#                 perturbed_schedule = self.perturb_solution(current_schedule, processing_times, n_jobs, n_machines)
#                 current_schedule = perturbed_schedule
#                 current_makespan = self.calculate_makespan(current_schedule)
#                 no_improve_count = 0
            
#             iteration += 1
            
#         return best_schedule, best_makespan
    
#     def calculate_makespan(self, schedule):
#         """Calculate the makespan of a schedule."""
#         makespan = 0
#         for job in schedule:
#             for _, _, end_time in job:
#                 makespan = max(makespan, end_time)
#         return makespan
    
#     def perturb_solution(self, schedule, processing_times, n_jobs, n_machines):
#         """
#         Perturb the current solution to escape local optima.
#         Randomly reassigns multiple operations to different machines.
#         """
#         perturbed = copy.deepcopy(schedule)
        
#         # Number of operations to perturb
#         num_perturb = max(1, int(n_jobs * 0.3))
        
#         for _ in range(num_perturb):
#             # Randomly select a job
#             job_id = np.random.randint(0, n_jobs)
#             if not perturbed[job_id]:
#                 continue
                
#             # Randomly select an operation
#             op_idx = np.random.randint(0, len(perturbed[job_id]))
            
#             # Get eligible machines
#             machine_id_list, proc_time_list = processing_times[job_id][op_idx]
            
#             if len(machine_id_list) <= 1:
#                 continue
                
#             # Choose a different machine
#             current_machine, _, _ = perturbed[job_id][op_idx]
#             available_machines = [m for m in machine_id_list if m != current_machine]
            
#             if not available_machines:
#                 continue
                
#             new_machine = np.random.choice(available_machines)
#             new_proc_time = proc_time_list[machine_id_list.index(new_machine)]
            
#             # Calculate new start time
#             earliest_start = 0
#             if op_idx > 0:
#                 earliest_start = perturbed[job_id][op_idx-1][2]
            
#             # Update the operation
#             perturbed[job_id][op_idx] = (new_machine, earliest_start, earliest_start + new_proc_time)
            
#             # Update subsequent operations
#             for next_idx in range(op_idx + 1, len(perturbed[job_id])):
#                 m, s, e = perturbed[job_id][next_idx]
#                 duration = e - s
#                 new_s = max(s, perturbed[job_id][next_idx-1][2])
#                 perturbed[job_id][next_idx] = (m, new_s, new_s + duration)
        
#         return perturbed

#     def plot_solution(self, schedule, n_jobs, n_machines):
#         """Plot the schedule as a Gantt chart."""
#         fig, ax = plt.subplots(figsize=(12, 6))
#         colors = plt.cm.get_cmap('tab20', n_jobs)
        
#         for job_id, operations in enumerate(schedule):
#             for machine_id, start_time, end_time in operations:
#                 ax.barh(
#                     machine_id, 
#                     end_time - start_time, 
#                     left=start_time, 
#                     color=colors(job_id), 
#                     edgecolor='black', 
#                     label=f'Job {job_id}' if machine_id == 0 else ""
#                 )
#                 ax.text(
#                     start_time + (end_time - start_time) / 2, 
#                     machine_id, 
#                     f'J{job_id}', 
#                     ha='center', 
#                     va='center'
#                 )
        
#         # Remove duplicate labels
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
#         ax.set_yticks(range(n_machines))
#         ax.set_yticklabels([f'M{i}' for i in range(n_machines)])
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Machine')
#         ax.set_title('FJSP Schedule')
#         plt.tight_layout()
#         plt.show()




# name: str: FJSPEvaluation
# Parameters:
# timeout_seconds: int: 60
# end
from __future__ import annotations

from typing import Tuple, Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.fjsp_ls.get_instance import GetData, FJSPInstance
from llm4ad.task.optimization.fjsp_ls.template import template_program, task_description
from .ls import local_search

__all__ = ['FJSPEvaluation']

perturbation_moves = 5
iter_limit = 1000


def calculate_cost(inst: FJSPInstance, schedule: np.ndarray) -> float:
    # Calculate the total cost of the given schedule
    total_cost = 0.0
    for job in range(inst.n_jobs):
        for machine in range(inst.n_machines):
            total_cost += inst.processing_times[job, machine] * schedule[job, machine]
    return total_cost


def solve_with_time(inst: FJSPInstance, eva) -> Tuple[float, float]:
    try:
        result, running_time = local_search(inst.processing_times, eva, perturbation_moves, iter_limit)
        cost = calculate_cost(inst, result)
    except Exception as e:
        cost, running_time = float("inf"), float("inf")
    return cost, running_time


def evaluate(instance_data, n_ins, prob_size, eva: callable) -> np.ndarray:
    objs = np.zeros((n_ins, 2))

    for i in range(n_ins):
        obj = solve_with_time(instance_data[i], eva)
        objs[i] = np.array(obj)

    obj = np.mean(objs, axis=0)
    return -obj


class FJSPEvaluation(Evaluation):
    """Evaluator for flexible job-shop scheduling problem."""

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=60
        )

        self.n_instance = 16
        self.problem_size = 100
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets = getData.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self._datasets, self.n_instance, self.problem_size, callable_func)


if __name__ == '__main__':
    from llm4ad.task.optimization.fjsp_ls.get_instance import GetData

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
    
    fjsp = FJSPEvaluation()
    fjsp.evaluate_program('_', update_schedule)