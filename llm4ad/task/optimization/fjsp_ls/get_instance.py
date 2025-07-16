# import numpy as np


# class GetData:
#     def __init__(self, n_instance: int, n_jobs: int, n_machines: int, min_machines_per_op: int = 1, max_machines_per_op: int = None):
#         """
#         Initialize the GetData class for FJSP.

#         Args:
#             n_instance: Number of instances to generate.
#             n_jobs: Number of jobs.
#             n_machines: Number of machines.
#             min_machines_per_op: Minimum number of candidate machines per operation.
#             max_machines_per_op: Maximum number of candidate machines per operation.
#         """
#         self.n_instance = n_instance
#         self.n_jobs = n_jobs
#         self.n_machines = n_machines
#         self.min_machines_per_op = min_machines_per_op
#         self.max_machines_per_op = max_machines_per_op or n_machines

#     def generate_instances(self):
#         """
#         Generate instances for the Flexible Job Shop Scheduling Problem (FJSP).

#         Returns:
#             A list of tuples, where each tuple contains:
#             - processing_times: A list of lists, each job is a list of operations, each operation is (machine_id_list, processing_time_list)
#             - n_jobs: Number of jobs.
#             - n_machines: Number of machines.
#         """
#         np.random.seed(2024)  # Set seed for reproducibility
#         instance_data = []

#         for _ in range(self.n_instance):
#             processing_times = []
#             for _ in range(self.n_jobs):
#                 job_ops = []
#                 n_ops = self.n_machines  # Each job has exactly n_machines operations
#                 for _ in range(n_ops):
#                     num_candidates = np.random.randint(self.min_machines_per_op, self.max_machines_per_op + 1)
#                     machine_id_list = np.random.choice(self.n_machines, size=num_candidates, replace=False).tolist()
#                     processing_time_list = np.random.randint(10, 100, size=num_candidates).tolist()
#                     job_ops.append((machine_id_list, processing_time_list))
#                 processing_times.append(job_ops)
#             instance_data.append((processing_times, self.n_jobs, self.n_machines))

#         return instance_data


import numpy as np
import numpy.typing as npt

class GetData:
    def __init__(self, n_instance, n_size):
        self.n_instance = n_instance
        self.n_size = n_size  # Size of the problem (jobs and machines)

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            n_jobs = self.n_size
            n_machines = max(3, self.n_size // 3)  # Reasonable number of machines
            
            # Generate processing times matrix
            # For FJSP, some job-machine combinations might be invalid (processing_time = 0)
            processing_times = np.random.randint(1, 100, size=(n_jobs, n_machines)).astype(np.float32)
            
            # Make some job-machine combinations invalid (20% chance)
            mask = np.random.random((n_jobs, n_machines)) < 0.2
            processing_times[mask] = 0
            
            # Ensure each job can be processed on at least one machine
            for j in range(n_jobs):
                if np.all(processing_times[j] == 0):
                    m = np.random.randint(0, n_machines)
                    processing_times[j, m] = np.random.randint(1, 100)
            
            instance_data.append(FJSPInstance(processing_times))
        
        return instance_data

class FJSPInstance:
    def __init__(self, processing_times: npt.NDArray[np.float_]):
        self.processing_times = processing_times
        self.n_jobs, self.n_machines = processing_times.shape