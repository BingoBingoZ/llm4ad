from llm4ad.task.optimization.fjsp_construct import FJSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
        host='api.chatanywhere.tech',   # your host endpoint, e.g., api.openai.com, api.deepseek.com（api.chatanywhere.org）
        key='sk-eFcUvNG4QWDfgVtr1Sea0DVps03MGkgnZbnLuJmRUYCBMrJE', # your key, e.g., sk-xxxxxxxxxx
        model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
        timeout=20
    )
    task = FJSPEvaluation()
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh', log_style='simple'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False
    )
    method.run()