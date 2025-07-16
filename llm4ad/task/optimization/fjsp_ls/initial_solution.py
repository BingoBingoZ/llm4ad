import random
import numpy as np
from typing import List, Tuple, Dict, Any

class GeneticAlgorithm:
    def __init__(self, processing_times, n_jobs, n_machines, 
                 pop_size=50, max_generations=100, 
                 crossover_prob=0.8, mutation_prob=0.2):
        """
        Initialize the genetic algorithm for FJSP.
        
        Args:
            processing_times: Processing time data for each job-operation-machine.
            n_jobs: Number of jobs.
            n_machines: Number of machines.
            pop_size: Population size.
            max_generations: Maximum number of generations.
            crossover_prob: Crossover probability.
            mutation_prob: Mutation probability.
        """
        self.processing_times = processing_times
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Calculate the total number of operations
        self.n_operations = sum(len(job) for job in processing_times)
        
    def generate_initial_population(self):
        """Generate an initial population of random solutions."""
        population = []
        
        for _ in range(self.pop_size):
            # Generate operation sequence
            operation_sequence = []
            for job_id in range(self.n_jobs):
                n_ops = len(self.processing_times[job_id])
                for op_idx in range(n_ops):
                    operation_sequence.append((job_id, op_idx))
            
            # Shuffle operation sequence
            random.shuffle(operation_sequence)
            
            # Generate machine assignment
            machine_assignment = {}
            for job_id in range(self.n_jobs):
                for op_idx in range(len(self.processing_times[job_id])):
                    machine_id_list, proc_time_list = self.processing_times[job_id][op_idx]
                    # Randomly select a machine
                    idx = random.randrange(len(machine_id_list))
                    machine_assignment[(job_id, op_idx)] = machine_id_list[idx]
            
            population.append((operation_sequence, machine_assignment))
        
        return population
    
    def decode_chromosome(self, chromosome):
        """
        Decode a chromosome into a schedule.
        
        Args:
            chromosome: A tuple (operation_sequence, machine_assignment)
            
        Returns:
            The schedule and its makespan.
        """
        operation_sequence, machine_assignment = chromosome
        
        # Initialize machine and job completion times
        machine_completion_time = [0] * self.n_machines
        job_completion_time = [0] * self.n_jobs
        job_current_op = [0] * self.n_jobs
        
        # Initialize schedule
        schedule = [[] for _ in range(self.n_jobs)]
        
        # Decode operation sequence
        for job_id, op_idx in operation_sequence:
            # Skip if this operation is already processed
            if op_idx < job_current_op[job_id]:
                continue
            
            # Skip if earlier operations in this job are not processed
            if op_idx > job_current_op[job_id]:
                continue
            
            # Get machine assignment
            machine_id = machine_assignment[(job_id, op_idx)]
            
            # Get processing time
            machine_id_list, proc_time_list = self.processing_times[job_id][op_idx]
            proc_time = proc_time_list[machine_id_list.index(machine_id)]
            
            # Calculate start and end time
            start_time = max(machine_completion_time[machine_id], job_completion_time[job_id])
            end_time = start_time + proc_time
            
            # Update times
            machine_completion_time[machine_id] = end_time
            job_completion_time[job_id] = end_time
            job_current_op[job_id] += 1
            
            # Add to schedule
            schedule[job_id].append((machine_id, start_time, end_time))
        
        # Calculate makespan
        makespan = max(job_completion_time)
        
        return schedule, makespan
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns:
            Two offspring chromosomes
        """
        # Extract components
        op_seq1, m_assign1 = parent1
        op_seq2, m_assign2 = parent2
        
        # Precedence preserving crossover for operation sequence
        crossover_point = random.randint(1, len(op_seq1) - 1)
        
        # Create child operation sequences
        child1_op_seq = op_seq1[:crossover_point] + [op for op in op_seq2 if op not in op_seq1[:crossover_point]]
        child2_op_seq = op_seq2[:crossover_point] + [op for op in op_seq1 if op not in op_seq2[:crossover_point]]
        
        # Uniform crossover for machine assignment
        child1_m_assign = {}
        child2_m_assign = {}
        
        for key in m_assign1.keys():
            if random.random() < 0.5:
                child1_m_assign[key] = m_assign1[key]
                child2_m_assign[key] = m_assign2[key]
            else:
                child1_m_assign[key] = m_assign2[key]
                child2_m_assign[key] = m_assign1[key]
        
        return (child1_op_seq, child1_m_assign), (child2_op_seq, child2_m_assign)
    
    def mutation(self, chromosome):
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: A chromosome to mutate
            
        Returns:
            A mutated chromosome
        """
        op_seq, m_assign = chromosome
        
        # Mutation for operation sequence (swap mutation)
        if random.random() < self.mutation_prob:
            idx1, idx2 = random.sample(range(len(op_seq)), 2)
            op_seq[idx1], op_seq[idx2] = op_seq[idx2], op_seq[idx1]
        
        # Mutation for machine assignment (reassignment)
        if random.random() < self.mutation_prob:
            # Randomly select an operation
            job_id = random.randrange(self.n_jobs)
            if len(self.processing_times[job_id]) > 0:
                op_idx = random.randrange(len(self.processing_times[job_id]))
                
                # Get eligible machines and processing times
                machine_id_list, _ = self.processing_times[job_id][op_idx]
                
                # Only reassign if there are multiple eligible machines
                if len(machine_id_list) > 1:
                    # Select a different machine
                    current_machine = m_assign[(job_id, op_idx)]
                    new_machine = current_machine
                    while new_machine == current_machine:
                        idx = random.randrange(len(machine_id_list))
                        new_machine = machine_id_list[idx]
                    
                    # Reassign
                    m_assign[(job_id, op_idx)] = new_machine
        
        return op_seq, m_assign
    
    def run(self):
        """
        Run the genetic algorithm.
        
        Returns:
            The best schedule found and its makespan.
        """
        # Initialize population
        population = self.generate_initial_population()
        
        # Evaluate initial population
        fitness = []
        for chromosome in population:
            _, makespan = self.decode_chromosome(chromosome)
            fitness.append(1 / makespan)  # Higher fitness for lower makespan
        
        best_makespan = float('inf')
        best_schedule = None
        
        # Main loop
        for generation in range(self.max_generations):
            # Select parents (tournament selection)
            new_population = []
            
            for _ in range(self.pop_size // 2):
                # Select parents
                parent1_idx = self.tournament_selection(fitness)
                parent2_idx = self.tournament_selection(fitness)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if random.random() < self.crossover_prob:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Evaluate new population
            population = new_population
            fitness = []
            
            for chromosome in population:
                schedule, makespan = self.decode_chromosome(chromosome)
                fitness.append(1 / makespan)
                
                # Update best solution
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}: Best makespan = {best_makespan}")
        
        return best_schedule, best_makespan
    
    def tournament_selection(self, fitness, tournament_size=3):
        """
        Perform tournament selection.
        
        Args:
            fitness: List of fitness values
            tournament_size: Size of the tournament
            
        Returns:
            Index of the selected individual
        """
        tournament = random.sample(range(len(fitness)), tournament_size)
        return tournament[np.argmax([fitness[i] for i in tournament])]

def generate_initial_solution(processing_times, n_jobs, n_machines):
    """
    Generate an initial solution using the genetic algorithm.
    
    Args:
        processing_times: Processing time data for each job-operation-machine.
        n_jobs: Number of jobs.
        n_machines: Number of machines.
        
    Returns:
        The best schedule found and its makespan.
    """
    ga = GeneticAlgorithm(
        processing_times=processing_times,
        n_jobs=n_jobs,
        n_machines=n_machines,
        pop_size=50,
        max_generations=50
    )
    
    return ga.run()