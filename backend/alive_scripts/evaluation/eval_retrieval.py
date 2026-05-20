import json, math, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

STORE_DIR = Path('rag_store')
GT_FILE   = Path('evaluation/retrieval_gt.json')

if not GT_FILE.exists():
    # Create a stub with 10 examples from your actual lectures
    # Fill relevant_slides by running:
    # python3 -c "
    # from pathlib import Path
    # for lec in sorted(Path('Lectures').iterdir()):
    #     for f in sorted((lec/'Texts').glob('*.txt')):
    #         t = f.read_text().lower()
    #         if 'backprojection' in t: print(lec.name, f.stem, f.read_text()[:80])
    # "
    stub = [
        {"question": "What is filtered backprojection in CT?",         "pause_time": 300.0,  "relevant_slides": []},
        {"question": "How does the Fourier slice theorem work?",        "pause_time": 900.0,  "relevant_slides": []},
        {"question": "What causes streak artifacts in CT imaging?",     "pause_time": 600.0,  "relevant_slides": []},
        {"question": "Explain k-space in MRI acquisition",             "pause_time": 1200.0, "relevant_slides": []},
        {"question": "What is the role of gradient coils in MRI?",     "pause_time": 1500.0, "relevant_slides": []},
        {"question": "How does PET imaging detect annihilation photons?","pause_time": 800.0, "relevant_slides": []},
        {"question": "What is the Hounsfield unit for water in CT?",    "pause_time": 400.0,  "relevant_slides": []},
        {"question": "Explain the convolution backprojection algorithm","pause_time": 700.0,  "relevant_slides": []},
        {"question": "What is the difference between T1 and T2 in MRI?","pause_time": 1100.0,"relevant_slides": []},
        {"question": "How does Radon transform relate to CT sinogram?", "pause_time": 500.0,  "relevant_slides": []},
    ]
    GT_FILE.write_text(json.dumps(stub, indent=2))
    print(f'Created stub -> {GT_FILE}')
    print('IMPORTANT: Fill in relevant_slides for each question before running eval.')
    print('Use this to find slides:')
    print('  python3 -c "')
    print('  from pathlib import Path')
    print('  q = \"backprojection\"')
    print('  for lec in sorted(Path(\"Lectures\").iterdir()):')
    print('      for f in sorted((lec/\"Texts\").glob(\"*.txt\")):')
    print('          if q.lower() in f.read_text().lower(): print(lec.name, f.stem)"')
    exit(0)

questions = json.loads(GT_FILE.read_text())
filled    = [q for q in questions if q['relevant_slides']]
if not filled:
    print('ERROR: relevant_slides is empty for all questions.')
    print(f'Edit {GT_FILE} and add slide IDs like: "relevant_slides": ["Lecture 3__Slide12"]')
    exit(1)

print(f'Loaded {len(filled)} questions with ground truth')

meta   = json.loads((STORE_DIR / 'meta.json').read_text())
chunks = json.loads((STORE_DIR / 'chunks.json').read_text())
index  = faiss.read_index(str(STORE_DIR / 'faiss.index'))
model  = SentenceTransformer(meta['embedding_model'], device='cuda')
LAMBDA = 0.4

def retrieve(question, pause_time=None, k=5):
    qv = model.encode([question], normalize_embeddings=True)
    D, I = index.search(qv.astype('float32'), k * 3)
    results = []
    for score, idx in zip(D[0], I[0]):
        c = chunks[idx]
        if pause_time is not None:
            mid   = (c['start'] + c['end']) / 2
            score = score - LAMBDA * abs(mid - pause_time) / 60
        results.append((score, c['slide_id']))
    results.sort(key=lambda x: -x[0])
    return [r[1] for r in results[:k]]

def ndcg(retrieved, relevant_set, k):
    rels  = [1 if s in relevant_set else 0 for s in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    def dcg(r): return sum(v / math.log2(i+2) for i, v in enumerate(r))
    idcg  = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0

def hit_at_k(retrieved, relevant_set, k):
    return 1 if any(s in relevant_set for s in retrieved[:k]) else 0

res = {c: {k: [] for k in [1,3,5]} for c in ['text_only','timestamp_aware']}
hits = {c: {k: [] for k in [1,3,5]} for c in ['text_only','timestamp_aware']}

for q in filled:
    rel = set(q['relevant_slides'])
    for cond, pt in [('text_only', None), ('timestamp_aware', q['pause_time'])]:
        ids = retrieve(q['question'], pause_time=pt, k=5)
        for k in [1,3,5]:
            res[cond][k].append(ndcg(ids, rel, k))
            hits[cond][k].append(hit_at_k(ids, rel, k))

print(f'\n{"Condition":<22} {"NDCG@1":>8} {"NDCG@3":>8} {"NDCG@5":>8} {"Hit@1":>8} {"Hit@5":>8}')
print('-' * 62)
for cond in ['text_only','timestamp_aware']:
    ndcg_vals = [f'{sum(res[cond][k])/len(res[cond][k]):.4f}' for k in [1,3,5]]
    hit_vals  = [f'{sum(hits[cond][k])/len(hits[cond][k]):.4f}' for k in [1,5]]
    print(f'{cond:<22} {ndcg_vals[0]:>8} {ndcg_vals[1]:>8} {ndcg_vals[2]:>8} {hit_vals[0]:>8} {hit_vals[1]:>8}')

Path('evaluation/ndcg_results.json').write_text(json.dumps(res, indent=2))
print('\nSaved -> evaluation/ndcg_results.json')
