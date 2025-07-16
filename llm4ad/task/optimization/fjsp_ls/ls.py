import time
import numpy as np
import numpy.typing as npt
import numba as nb
from typing import Tuple, Callable

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
usecache = True

@nb.njit(nb.float32(nb.float32[:,:], nb.uint8[:,:]), nogil=True, cache=usecache)
def _calculate_cost(processing_times, schedule):
    """Calculate the total cost of the given schedule"""
    total_cost = 0.0
    n_jobs, n_machines = processing_times.shape
    for job in range(n_jobs):
        for machine in range(n_machines):
            total_cost += processing_times[job, machine] * schedule[job, machine]
    return total_cost

@nb.njit(nb.float32(nb.float32[:,:], nb.uint8[:,:]), nogil=True, cache=usecache)
def _swap_once(processing_times, schedule):
    """Try to swap job assignments to improve the solution"""
    n_jobs, n_machines = processing_times.shape
    delta = 0.0
    best_delta = 0.0
    best_i, best_j, best_m1, best_m2 = -1, -1, -1, -1
    
    for i in range(n_jobs):
        m1 = np.argmax(schedule[i])
        for j in range(i+1, n_jobs):
            m2 = np.argmax(schedule[j])
            if m1 == m2:
                continue
                
            # Calculate cost change if we swap machines for these jobs
            current_cost = (processing_times[i, m1] * schedule[i, m1] + 
                           processing_times[j, m2] * schedule[j, m2])
            new_cost = (processing_times[i, m2] + processing_times[j, m1])
            
            delta = new_cost - current_cost
            if delta < best_delta and processing_times[i, m2] > 0 and processing_times[j, m1] > 0:
                best_delta = delta
                best_i, best_j, best_m1, best_m2 = i, j, m1, m2
    
    if best_delta < -1e-6:
        # Perform the swap
        schedule[best_i, best_m1] = 0
        schedule[best_i, best_m2] = 1
        schedule[best_j, best_m2] = 0
        schedule[best_j, best_m1] = 1
        return best_delta
    
    return 0.0

@nb.njit(nb.float32(nb.float32[:,:], nb.uint8[:,:]), nogil=True, cache=usecache)
def _reassign_once(processing_times, schedule):
    """Try to reassign a job to improve the solution"""
    n_jobs, n_machines = processing_times.shape
    delta = 0.0
    best_delta = 0.0
    best_job, best_old_m, best_new_m = -1, -1, -1
    
    for job in range(n_jobs):
        current_m = np.argmax(schedule[job])
        current_cost = processing_times[job, current_m] * schedule[job, current_m]
        
        for new_m in range(n_machines):
            if new_m == current_m or processing_times[job, new_m] <= 0:
                continue
                
            new_cost = processing_times[job, new_m]
            delta = new_cost - current_cost
            
            if delta < best_delta:
                best_delta = delta
                best_job = job
                best_old_m = current_m
                best_new_m = new_m
    
    if best_delta < -1e-6:
        # Perform the reassignment
        schedule[best_job, best_old_m] = 0
        schedule[best_job, best_new_m] = 1
        return best_delta
    
    return 0.0

@nb.njit(nb.float32(nb.float32[:,:], nb.uint8[:,:], nb.uint16), nogil=True, cache=usecache)
def _local_search(processing_times, schedule, count=1000):
    """Perform local search using swap and reassign operations"""
    sum_delta = 0.0
    delta = -1.0
    
    while delta < 0 and count > 0:
        delta = 0.0
        delta += _swap_once(processing_times, schedule)
        delta += _reassign_once(processing_times, schedule)
        count -= 1
        sum_delta += delta
        
    return sum_delta

@nb.njit(nb.uint8[:,:](nb.float32[:,:]), nogil=True, cache=usecache)
def _init_greedy_schedule(processing_times):
    """Initialize a schedule by assigning each job to its fastest machine"""
    n_jobs, n_machines = processing_times.shape
    schedule = np.zeros((n_jobs, n_machines), dtype=np.uint8)
    
    for job in range(n_jobs):
        # Find valid machines (processing time > 0)
        valid_machines = np.where(processing_times[job] > 0)[0]
        if len(valid_machines) > 0:
            # Find the machine with minimum processing time
            best_machine = valid_machines[np.argmin(processing_times[job, valid_machines])]
            schedule[job, best_machine] = 1
    
    return schedule

def _perturbation(processing_times, schedule, schedule_history, update_schedule, perturbation_moves=5):
    """Perturb the current solution to escape local optima"""
    n_jobs, n_machines = processing_times.shape
    
    # Update the schedule using the provided function
    updated_schedule = update_schedule(processing_times, schedule, schedule_history)
    
    # Update schedule history
    for job in range(n_jobs):
        for machine in range(n_machines):
            if updated_schedule[job, machine] == 1:
                schedule_history[job, machine] += 1
    
    return updated_schedule

def _guided_local_search(
    processing_times, update_schedule_func, perturbation_moves=5, iter_limit=1000
) -> Tuple[npt.NDArray[np.uint8], float]:
    """Perform guided local search for FJSP"""
    start_time = time.monotonic()
    n_jobs, n_machines = processing_times.shape
    
    # Initialize schedule history to track assignments
    schedule_history = np.zeros((n_jobs, n_machines), dtype=np.float32)
    
    # Create initial schedule
    best_schedule = _init_greedy_schedule(processing_times)
    _local_search(processing_times, best_schedule, 1000)
    best_cost = _calculate_cost(processing_times, best_schedule)
    
    current_schedule = best_schedule.copy()
    
    for _ in range(iter_limit):
        # Perform perturbation
        current_schedule = _perturbation(
            processing_times, current_schedule, schedule_history, 
            update_schedule_func, perturbation_moves
        )
        
        # Perform local search
        _local_search(processing_times, current_schedule, 1000)
        
        # Update best solution if improved
        current_cost = _calculate_cost(processing_times, current_schedule)
        if current_cost < best_cost:
            best_schedule = current_schedule.copy()
            best_cost = current_cost
            
        running_time = time.monotonic() - start_time
        if running_time > 60:
            break
    
    return best_schedule, running_time

def local_search(
    processing_times: FloatArray,
    update_schedule_func: Callable,
    perturbation_moves: int = 5,
    iter_limit: int = 1000
) -> Tuple[npt.NDArray[np.uint8], float]:
    """Public interface for guided local search algorithm"""
    return _guided_local_search(
        processing_times.astype(np.float32),
        update_schedule_func,
        perturbation_moves=perturbation_moves,
        iter_limit=iter_limit
    )