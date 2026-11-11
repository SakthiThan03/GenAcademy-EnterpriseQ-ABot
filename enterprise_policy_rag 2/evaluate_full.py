import csv, json
from pathlib import Path
from rag_full import FullPolicyRAG
root = Path(__file__).parent
cases = json.loads((root / 'evaluation_cases.json').read_text())
rag = FullPolicyRAG()
rows = []
for case in cases:
    result = rag.ask(case['question'])
    sources = {c['source'] for c in result['citations']}
    expected = set(case['expected_sources'])
    recall = len(sources & expected) / len(expected) if expected else (1.0 if not sources else 0.0)
    precision = len(sources & expected) / len(sources) if sources else (1.0 if not expected else 0.0)
    phrase = case['expected_phrase']
    passed = phrase.lower() in result['answer'].lower() if phrase else result['status'] == 'abstained'
    rows.append({'id':case['id'], 'category':case['category'], 'question':case['question'],
                 'expected_sources':'; '.join(sorted(expected)), 'retrieved_sources':'; '.join(sorted(sources)),
                 'precision':round(precision,2), 'recall':round(recall,2), 'answer_check_pass':int(passed),
                 'status':result['status'], 'answer':result['answer']})
    print(f"Question {case['id']}: {result['status']} — {'PASS' if passed else 'REVIEW'}", flush=True)
with (root / 'evaluation_results_full.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
summary = {'questions':len(rows), 'mean_source_precision':round(sum(r['precision'] for r in rows)/len(rows),3),
           'mean_source_recall':round(sum(r['recall'] for r in rows)/len(rows),3),
           'phrase_or_abstention_pass_rate':round(sum(r['answer_check_pass'] for r in rows)/len(rows),3)}
(root / 'evaluation_summary_full.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
