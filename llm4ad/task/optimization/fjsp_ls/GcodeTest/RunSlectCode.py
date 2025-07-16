import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# 添加项目根目录到系统路径以便导入模块
sys.path.append('/home/zgb/llm4ad')

# 导入数据加载模块
from llm4ad.task.optimization.fjsp_ls.GcodeTest.load_data import load_fjsp_data

# 导入进化算法初始解生成方法
from llm4ad.task.optimization.fjsp_ls.gls.ea_initializer import generate_initial_solution

# 导入大模型生成的局部搜索策略
# 假设这是由EoH框架找到的最佳函数，可能需要替换为实际路径
from llm4ad.task.optimization.fjsp_ls.gls.ls_neighborhood import apply_neighborhood

def schedule_ea_ls(processing_times, n_jobs, n_machines, ea_iterations=10, ls_iterations=100, verbose=True):
    """
    使用进化算法+局部搜索方法调度FJSP实例
    
    Args:
        processing_times: 处理时间数据
        n_jobs: 作业数量
        n_machines: 机器数量
        ea_iterations: 进化算法迭代次数
        ls_iterations: 局部搜索迭代次数
        verbose: 是否打印详细信息
        
    Returns:
        makespan: 完工时间
        schedule: 调度方案
        computation_time: 计算时间
        ea_time: 进化算法时间
        ls_time: 局部搜索时间
        initial_makespan: 初始解makespan
        improved_makespan: 改进后makespan
    """
    if verbose:
        print("开始进化算法初始解生成...")
    
    start_time = time.time()
    
    # 第一阶段：使用进化算法生成初始解
    initial_solution = generate_initial_solution(
        processing_times, 
        n_jobs, 
        n_machines, 
        population_size=20, 
        generations=ea_iterations
    )
    
    initial_makespan = initial_solution.evaluate()
    ea_time = time.time() - start_time
    
    if verbose:
        print(f"进化算法完成，初始解makespan: {initial_makespan}, 耗时: {ea_time:.2f}秒")
        print("开始局部搜索优化...")
    
    # 第二阶段：使用局部搜索优化初始解
    ls_start_time = time.time()
    improved_solution = apply_neighborhood(
        initial_solution,
        processing_times,
        n_jobs,
        n_machines,
        max_iterations=ls_iterations
    )
    ls_time = time.time() - ls_start_time
    improved_makespan = improved_solution.evaluate()
    
    if verbose:
        print(f"局部搜索完成，优化后makespan: {improved_makespan}, 耗时: {ls_time:.2f}秒")
        improvement = (initial_makespan - improved_makespan) / initial_makespan * 100
        print(f"改进比例: {improvement:.2f}%")
    
    # 从解决方案对象中提取调度信息
    makespan = improved_solution.evaluate()
    schedule = [[] for _ in range(n_jobs)]
    
    for op in improved_solution.schedule:
        job_id = op['job_id']
        machine_id = op['machine_id']
        start = op['start_time']
        end = op['end_time']
        schedule[job_id].append((machine_id, start, end))
    
    # 确保每个作业的工序按工序索引排序
    for job_id in range(n_jobs):
        schedule[job_id].sort(key=lambda x: x[1])  # 按开始时间排序
    
    computation_time = time.time() - start_time
    
    return makespan, schedule, computation_time, ea_time, ls_time, initial_makespan, improved_makespan

def draw_gantt(schedule, n_jobs, n_machines, title="FJSP Gantt Chart"):
    """绘制甘特图"""
    # 将按作业组织的数据转换为按机器组织的数据
    machine_schedules = [[] for _ in range(n_machines)]
    
    # 初始化每个机器的时间表
    for i in range(n_machines):
        machine_schedules[i] = [i]  # 第一个元素是机器ID
    
    # 填充机器时间表
    for job_idx, operations in enumerate(schedule):
        for op_idx, operation in enumerate(operations):
            machine, start_time, end_time = operation
            # 添加到对应机器的时间表 [开始时间, 作业ID, 工序ID, 结束时间]
            machine_schedules[machine].append([start_time, job_idx, op_idx, end_time])
    
    # 按照开始时间排序每个机器的操作
    for machine_idx in range(n_machines):
        machine_schedules[machine_idx][1:] = sorted(machine_schedules[machine_idx][1:], key=lambda x: x[0])
    
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 颜色映射字典，为每个工件分配一个唯一的颜色
    color_map = {}
    for machine_schedule in machine_schedules:
        for task_data in machine_schedule[1:]:
            job_idx = task_data[1]
            if job_idx not in color_map:
                # 为新工件分配一个随机颜色
                color_map[job_idx] = (random.random(), random.random(), random.random())

    # 遍历机器
    for machine_idx, machine_schedule in enumerate(machine_schedules):
        for task_data in machine_schedule[1:]:
            start_time, job_idx, operation_idx, end_time = task_data
            color = color_map[job_idx]  # 获取工件的颜色

            # 绘制甘特图条形，使用工件的颜色
            ax.barh(machine_idx, end_time - start_time, left=start_time, height=0.4, color=color)

            # 在色块内部标注工件-工序
            label = f'{job_idx + 1}-{operation_idx + 1}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='white',
                    fontsize=10)

    # 设置Y轴标签为机器名称
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i + 1}' for i in range(n_machines)])

    # 设置X轴标签
    plt.xlabel("Time")

    # 添加标题
    plt.title(title)

    return fig

def test_single_instance(filepath, ea_iterations=10, ls_iterations=100, draw_charts=True):
    """测试单个FJSP实例"""
    instance_name = os.path.basename(filepath).split('.')[0]
    print(f"\n测试实例: {instance_name}")
    
    # 加载实例数据
    processing_times, n_jobs, n_machines = load_fjsp_data(filepath)
    
    # 打印问题信息
    print(f"问题规模: {n_jobs}个作业, {n_machines}台机器")
    print(f"各作业工序数: {[len(job_ops) for job_ops in processing_times]}")
    total_ops = sum(len(job_ops) for job_ops in processing_times)
    print(f"总工序数: {total_ops}")
    
    # 使用进化算法+局部搜索方法调度
    print("\n使用进化算法+大模型局部搜索进行调度...")
    makespan, schedule, total_time, ea_time, ls_time, initial_makespan, improved_makespan = schedule_ea_ls(
        processing_times, n_jobs, n_machines, ea_iterations, ls_iterations
    )
    
    # 计算改进比例
    improvement = (initial_makespan - improved_makespan) / initial_makespan * 100
    
    # 输出详细结果
    print("\n调度结果:")
    print(f"初始解makespan: {initial_makespan}")
    print(f"优化后makespan: {improved_makespan}")
    print(f"改进比例: {improvement:.2f}%")
    print(f"总计算时间: {total_time:.2f}秒 (进化算法: {ea_time:.2f}秒, 局部搜索: {ls_time:.2f}秒)")
    
    # 检查每个作业的工序数
    for job_id, operations in enumerate(schedule):
        print(f"作业 {job_id}: {len(operations)}/{len(processing_times[job_id])} 个工序")
    
    # 绘制甘特图
    if draw_charts:
        # 进化算法+局部搜索方法的甘特图
        fig = draw_gantt(schedule, n_jobs, n_machines, 
                       f"EA+LS混合优化 - {instance_name} - Makespan: {makespan}")
        
        # 显示图形
        plt.figure(fig.number)
        plt.savefig(f"{instance_name}_ea_ls.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'instance': instance_name,
        'n_jobs': n_jobs,
        'n_machines': n_machines,
        'total_ops': total_ops,
        'initial_makespan': initial_makespan,
        'improved_makespan': improved_makespan,
        'improvement': improvement,
        'total_time': total_time,
        'ea_time': ea_time,
        'ls_time': ls_time
    }

def batch_test(data_dir, instances=None, ea_iterations=10, ls_iterations=100):
    """批量测试多个FJSP实例"""
    if instances is None:
        # 默认使用Public目录下的所有实例
        instances_dir = os.path.join(data_dir, 'Public')
        instances = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]
    
    results = []
    
    for instance in instances:
        filepath = os.path.join(data_dir, 'Public', instance)
        if os.path.exists(filepath):
            result = test_single_instance(filepath, ea_iterations, ls_iterations, draw_charts=False)
            results.append(result)
        else:
            print(f"实例文件不存在: {filepath}")
    
    # 汇总结果
    print("\n\n==== 批量测试结果汇总 ====")
    
    # 准备表格数据
    table_data = []
    for r in results:
        table_data.append([
            r['instance'],
            f"{r['n_jobs']}x{r['n_machines']}",
            r['total_ops'],
            r['initial_makespan'],
            r['improved_makespan'],
            f"{r['improvement']:.2f}%",
            f"{r['total_time']:.2f}",
            f"{r['ea_time']:.2f}",
            f"{r['ls_time']:.2f}"
        ])
    
    # 计算平均改进率
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    
    # 输出表格
    try:
        from tabulate import tabulate
        headers = ["实例", "规模", "工序数", "初始解", "优化后", "改进率", "总时间(秒)", "EA时间(秒)", "LS时间(秒)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        # 如果tabulate不可用，使用简单格式输出
        print("实例\t规模\t工序数\t初始解\t优化后\t改进率\t总时间\tEA时间\tLS时间")
        for row in table_data:
            print("\t".join(str(cell) for cell in row))
    
    print(f"\n平均改进率: {avg_improvement:.2f}%")
    
    # 将结果保存到CSV文件
    with open('fjsp_ea_ls_results.csv', 'w') as f:
        f.write("实例,规模,工序数,初始解,优化后,改进率,总时间(秒),EA时间(秒),LS时间(秒)\n")
        for row in table_data:
            f.write(','.join(str(cell) for cell in row) + '\n')
    
    print("\n结果已保存到 fjsp_ea_ls_results.csv")
    
    return results

def main():
    # 设置数据目录
    data_dir = '/home/zgb/llm4ad/llm4ad/task/optimization/fjsp_construct2/GcodeTest/data_test'
    
    # 测试模式选择
    test_mode = input("选择测试模式 (1:单个实例, 2:批量测试): ")
    
    if test_mode == '1':
        # 单实例测试
        instance_name = input("输入实例名称 (如 v_la01.fjs): ")
        if not instance_name.endswith('.fjs'):
            instance_name += '.fjs'
        
        filepath = os.path.join(data_dir, 'Public', instance_name)
        if os.path.exists(filepath):
            # 设置迭代次数
            ea_iterations = int(input("设置进化算法迭代次数 (默认: 10): ") or "10")
            ls_iterations = int(input("设置局部搜索迭代次数 (默认: 100): ") or "100")
            
            test_single_instance(filepath, ea_iterations, ls_iterations)
        else:
            print(f"实例文件不存在: {filepath}")
    else:
        # 批量测试
        instance_filter = input("输入要测试的实例前缀 (回车测试所有实例): ")
        
        if instance_filter:
            # 获取匹配前缀的实例
            instances_dir = os.path.join(data_dir, 'Public')
            instances = [f for f in os.listdir(instances_dir) 
                        if f.endswith('.fjs') and f.startswith(instance_filter)]
        else:
            instances = None
            
        # 设置迭代次数
        ea_iterations = int(input("设置进化算法迭代次数 (默认: 10): ") or "10")
        ls_iterations = int(input("设置局部搜索迭代次数 (默认: 100): ") or "100")
        
        batch_test(data_dir, instances, ea_iterations, ls_iterations)

if __name__ == "__main__":
    main()