# filepath: /home/zgb/llm4ad/llm4ad/task/optimization/fjsp_ls/gls/solution.py
from typing import List, Tuple, Dict, Any
import copy

class FJSPSolution:
    """FJSP解决方案的表示"""
    
    def __init__(self, schedule, n_jobs, n_machines, processing_times):
        """
        初始化FJSP解决方案
        
        Args:
            schedule: 调度方案，包含job_id, op_idx, machine_id, start_time, end_time
            n_jobs: 作业数量
            n_machines: 机器数量
            processing_times: 处理时间
        """
        self.schedule = sorted(schedule, key=lambda x: x['start_time'])
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.processing_times = processing_times
        
        # 生成操作序列和机器分配
        self._generate_encoding()
    
    def _generate_encoding(self):
        """从调度方案生成操作序列和机器分配"""
        # 操作序列：按开始时间排序的操作全局索引
        self.operation_sequence = []
        # 机器分配：(job_id, op_idx) -> machine_id
        self.machine_assignment = {}
        
        for op in self.schedule:
            job_id = op['job_id']
            op_idx = op['op_idx']
            machine_id = op['machine_id']
            
            # 生成全局操作索引
            global_op_idx = job_id * 100 + op_idx
            self.operation_sequence.append(global_op_idx)
            
            # 记录机器分配
            self.machine_assignment[(job_id, op_idx)] = machine_id
    
    @classmethod
    def from_encoding(cls, operation_sequence, machine_assignment, n_jobs, n_machines, processing_times):
        """从编码生成解决方案"""
        # 解码操作序列和机器分配
        return decode_solution(operation_sequence, machine_assignment, n_jobs, n_machines, processing_times)
    
    def copy(self):
        """创建解决方案的深拷贝"""
        return FJSPSolution(
            copy.deepcopy(self.schedule),
            self.n_jobs,
            self.n_machines,
            self.processing_times
        )
    
    def evaluate(self):
        """评估解决方案，返回makespan"""
        if not self.schedule:
            return float('inf')
        
        return max(op['end_time'] for op in self.schedule)
    
    def get_operation_sequence(self):
        """获取操作序列"""
        return self.operation_sequence
    
    def get_machine_assignment(self):
        """获取机器分配"""
        return self.machine_assignment
    
    def swap_operations(self, pos1, pos2):
        """交换操作序列中的两个操作"""
        if 0 <= pos1 < len(self.operation_sequence) and 0 <= pos2 < len(self.operation_sequence):
            # 交换操作序列
            self.operation_sequence[pos1], self.operation_sequence[pos2] = \
                self.operation_sequence[pos2], self.operation_sequence[pos1]
            
            # 重新解码生成调度方案
            solution = decode_solution(
                self.operation_sequence,
                self.machine_assignment,
                self.n_jobs,
                self.n_machines,
                self.processing_times
            )
            
            # 更新自身
            self.schedule = solution.schedule
            self._generate_encoding()
    
    def reassign_machine(self, job_id, op_idx, new_machine_id):
        """重新分配机器"""
        if (job_id, op_idx) in self.machine_assignment:
            # 更新机器分配
            self.machine_assignment[(job_id, op_idx)] = new_machine_id
            
            # 重新解码生成调度方案
            solution = decode_solution(
                self.operation_sequence,
                self.machine_assignment,
                self.n_jobs,
                self.n_machines,
                self.processing_times
            )
            
            # 更新自身
            self.schedule = solution.schedule
            self._generate_encoding()
    
    def get_available_machines(self, job_id, op_idx):
        """获取操作可用的机器列表"""
        if 0 <= job_id < len(self.processing_times) and 0 <= op_idx < len(self.processing_times[job_id]):
            machine_ids, _ = self.processing_times[job_id][op_idx]
            return machine_ids
        return []

def decode_solution(operation_sequence, machine_assignment, n_jobs, n_machines, processing_times):
    """解码操作序列和机器分配，生成调度方案"""
    # 初始化状态
    machine_status = [0] * n_machines  # 机器可用时间
    job_status = [0] * n_jobs          # 作业可用时间
    job_next_op = [0] * n_jobs         # 每个作业的下一工序
    n_ops = [len(job_ops) for job_ops in processing_times]
    
    schedule = []
    
    # 按照操作序列依次调度
    for global_op_idx in operation_sequence:
        job_id = global_op_idx // 100
        op_idx = global_op_idx % 100
        
        # 检查工序约束
        if op_idx != job_next_op[job_id]:
            continue  # 跳过违反工序约束的操作
        
        # 获取机器分配
        machine_id = machine_assignment.get((job_id, op_idx))
        if machine_id is None:
            continue
        
        # 检查机器有效性
        available_machines, proc_times = processing_times[job_id][op_idx]
        if machine_id not in available_machines:
            continue
        
        # 获取处理时间
        machine_idx = available_machines.index(machine_id)
        proc_time = proc_times[machine_idx]
        
        # 计算开始和结束时间
        start_time = max(job_status[job_id], machine_status[machine_id])
        end_time = start_time + proc_time
        
        # 更新状态
        machine_status[machine_id] = end_time
        job_status[job_id] = end_time
        job_next_op[job_id] += 1
        
        # 记录调度
        schedule.append({
            'job_id': job_id,
            'op_idx': op_idx,
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': end_time
        })
    
    return FJSPSolution(schedule, n_jobs, n_machines, processing_times)