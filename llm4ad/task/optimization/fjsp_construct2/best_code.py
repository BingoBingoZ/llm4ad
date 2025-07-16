# def determine_next_operation(current_status, feasible_operations):\n    \"\"\"\n 
# 
def determine_next_operation(current_status, feasible_operations):
    """
    Determine the next operation to schedule for FJSP.
    
    Args:
        current_status: A dictionary with machine_status and job_status
        feasible_operations: A list of (job_id, machine_id_list, processing_time_list) tuples
        
    Returns:
        A tuple (job_id, best_machine, best_processing_time)
    """
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







































