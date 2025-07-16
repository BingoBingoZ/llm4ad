from llm4ad.task.optimization.fjsp_ls import FJSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
        host='api.chatanywhere.tech',
        key='sk-eFcUvNG4QWDfgVtr1Sea0DVps03MGkgnZbnLuJmRUYCBMrJE',
        model='gpt-4o-mini',
        timeout=80
    )
    
    task = FJSPEvaluation()
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/fjsp_ls', log_style='simple'),
        evaluation=task,
        max_sample_nums=50,
        max_generations=15,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False
    )
    
    method.run()