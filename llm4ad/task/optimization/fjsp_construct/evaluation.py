# Module Name: JSSPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the Job Shop Scheduling Problem (JSSP).
#              Given a set of jobs and machines, the goal is to schedule jobs on machines
#              in a way that minimizes the total makespan (completion time of all jobs).
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 20).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - n_jobs: Number of jobs to schedule: int (default: 10).
#    - n_machines: Number of machines available: int (default: 5).
# 
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------


from __future__ import annotations
from typing import Any, List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt

from llm4ad.base import Evaluation
from llm4ad.task.optimization.jssp_construct.get_instance import GetData
from llm4ad.task.optimization.jssp_construct.template import template_program, task_description

__all__ = ['FJSPEvaluation']

class FJSPEvaluation(Evaluation):
    """Evaluator for Flexible Job Shop Scheduling Problem (FJSP)."""

    def __init__(self,
                 timeout_seconds=20,
                 n_instance=16,
                 n_jobs=10,
                 n_machines=5,
                 **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.n_instance = n_instance
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        getData = GetData(self.n_instance, self.n_jobs, self.n_machines)
        self._datasets = getData.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def schedule_jobs(self, processing_times, n_jobs, n_machines, eva):
        """
        Schedule jobs on machines for FJSP using a greedy constructive heuristic.
        Args:
            processing_times: A list of lists of lists. For each job, a list of operations, each operation is a tuple (machine_id_list, processing_time_list).
            n_jobs: Number of jobs.
            n_machines: Number of machines.
        Returns:
            The makespan and the operation sequence.
        """
        machine_status = [0] * n_machines
        job_status = [0] * n_jobs
        operation_sequence = [[] for _ in range(n_jobs)]
        # 初始化每个作业的下一个工序索引
        job_next_op = [0] * n_jobs
        # 统计每个作业的工序数
        n_ops = [len(job_ops) for job_ops in processing_times]
        total_ops = sum(n_ops)
        scheduled_ops = 0
        while scheduled_ops < total_ops:
            feasible_operations = []
            for job_id in range(n_jobs):
                op_idx = job_next_op[job_id]
                if op_idx < n_ops[job_id]:
                    machine_id_list, processing_time_list = processing_times[job_id][op_idx]
                    # 检查所有可选机器，是否有机器可用
                    for machine_id, processing_time in zip(machine_id_list, processing_time_list):
                        if job_status[job_id] <= machine_status[machine_id]:
                            feasible_operations.append((job_id, machine_id_list, processing_time_list))
                            break
            if not feasible_operations:
                # 如果没有可调度操作，跳过（理论上不会发生）
                break
            next_op = eva({'machine_status': machine_status, 'job_status': job_status}, feasible_operations)
            job_id, machine_id_list, processing_time_list = next_op
            op_idx = job_next_op[job_id]
            # 选择该工序中最优的机器分配
            min_time = float('inf')
            best_machine = None
            for machine_id, processing_time in zip(machine_id_list, processing_time_list):
                if job_status[job_id] <= machine_status[machine_id] and processing_time < min_time:
                    min_time = processing_time
                    best_machine = machine_id
            start_time = max(job_status[job_id], machine_status[best_machine])
            end_time = start_time + min_time
            machine_status[best_machine] = end_time
            job_status[job_id] = end_time
            operation_sequence[job_id].append((best_machine, start_time, end_time))
            job_next_op[job_id] += 1
            scheduled_ops += 1
        makespan = max(job_status)
        return makespan, operation_sequence

    def evaluate(self, eva: Callable) -> float:
        makespans = []
        for instance in self._datasets[:self.n_instance]:
            processing_times, n1, n2 = instance
            makespan, solution = self.schedule_jobs(processing_times, n1, n2, eva)
            makespans.append(makespan)
        average_makespan = np.mean(makespans)
        return -average_makespan

if __name__ == '__main__':
    def determine_next_operation(current_status, feasible_operations):
        """
        Greedy scheduling for FJSP: select the job-machine pair with the shortest processing time.
        Each feasible operation is (job_id, machine_id_list, processing_time_list)
        """
        best = None
        min_time = float('inf')
        for op in feasible_operations:
            job_id, machine_id_list, processing_time_list = op
            for machine_id, processing_time in zip(machine_id_list, processing_time_list):
                if processing_time < min_time:
                    min_time = processing_time
                    best = (job_id, machine_id_list, processing_time_list)
        return best

    fjsp = FJSPEvaluation()
    makespan = fjsp.evaluate_program('_', determine_next_operation)
    print(makespan)
