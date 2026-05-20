import time, json, requests, statistics
from pathlib import Path

BASE_URL   = 'http://127.0.0.1:9000'   # your FastAPI backend
VLLM_URL   = 'http://127.0.0.1:8000/v1/chat/completions'
VLLM_MODEL = 'checkpoints/alive-llama-lora/merged'
N_RUNS     = 20
TEST_Q     = 'Explain the principle of filtered backprojection in CT imaging.'

results = {'retrieval_llm': [], 'tts': [], 'avatar_total': []}
print(f'Running {N_RUNS} iterations against vllm directly...')

for i in range(N_RUNS):
    # Retrieval + LLM (direct vllm timing)
    t0 = time.time()
    r = requests.post(VLLM_URL, json={
        'model': VLLM_MODEL,
        'messages': [{'role': 'user', 'content': f'Answer concisely: {TEST_Q}'}],
        'temperature': 0.0,
        'max_tokens': 150,
    }, timeout=30)
    results['retrieval_llm'].append(time.time() - t0)

    if (i+1) % 5 == 0:
        print(f'  Run {i+1}/{N_RUNS}  |  last latency: {results["retrieval_llm"][-1]:.2f}s')

print(f'\n{"Component":<20} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 56)
for k, v in results.items():
    if v:
        print(f'{k:<20} {statistics.mean(v):>8.3f} {statistics.stdev(v):>8.3f} {min(v):>8.3f} {max(v):>8.3f}')

# Also add known latencies from Paper 1 eval
paper1 = {'asr': 3.0, 'retrieval': 0.1, 'llm_base': 1.5, 'tts': 0.2, 'avatar': 5.0}
full = {
    'llm_finetuned_mean':  statistics.mean(results['retrieval_llm']),
    'llm_finetuned_std':   statistics.stdev(results['retrieval_llm']),
    'llm_finetuned_min':   min(results['retrieval_llm']),
    'llm_finetuned_max':   max(results['retrieval_llm']),
    'paper1_reference':    paper1,
    'raw': results,
}
Path('evaluation/latency_results.json').write_text(json.dumps(full, indent=2))
print('\nSaved -> evaluation/latency_results.json')
