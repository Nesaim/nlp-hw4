"""Collect actual results, per-query evidence and submission files after training."""
import csv
import hashlib
import json
import pickle
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import T5TokenizerFast
from utils import compute_metrics, load_queries_and_records, compute_record_F1


def category(error, same_records):
    if not error and same_records:
        return 'record_correct'
    text = error.lower()
    if any(x in text for x in ['syntax error', 'incomplete input', 'unrecognized token']):
        return 'SQL syntax error'
    if any(x in text for x in ['no such table', 'no such column', 'ambiguous column']):
        return 'Schema reference error'
    if error:
        return 'Other execution error or timeout'
    return 'Executable SQL returning incorrect records'


def main():
    reports = Path('reports')
    reports.mkdir(exist_ok=True)
    gt_sql, gt_records, _ = load_queries_and_records('data/dev.sql', 'records/ground_truth_dev.pkl')
    pred_sql, pred_records, errors = load_queries_and_records('results/t5_ft_experiment_dev.sql', 'records/t5_ft_experiment_dev.pkl')
    questions = Path('data/dev.nl').read_text().splitlines()
    assert len(questions) == len(gt_sql) == len(pred_sql) == len(pred_records) == len(errors)
    sql_em, record_em, record_f1, _ = compute_metrics(
        'data/dev.sql', 'results/t5_ft_experiment_dev.sql',
        'records/ground_truth_dev.pkl', 'records/t5_ft_experiment_dev.pkl')
    rows = []
    for i, (question, gold, predicted, gr, pr, error) in enumerate(zip(questions, gt_sql, pred_sql, gt_records, pred_records, errors)):
        same_records = set(gr) == set(pr)
        rows.append({'index': i, 'question': question, 'gold_sql': gold,
                     'predicted_sql': predicted, 'execution_error': error,
                     'record_f1': float(compute_record_F1([gr], [pr])),
                     'record_em': int(same_records),
                     'sql_em': int(gold == predicted),
                     'automatic_category': category(error, same_records)})
    with (reports / 'q6_dev_queries.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row['automatic_category'] for row in rows)
    metrics = {'dev_record_f1': float(record_f1), 'dev_record_em': float(record_em),
               'dev_sql_em': float(sql_em), 'dev_query_count': len(rows),
               'dev_execution_error_fraction': sum(bool(e) for e in errors) / len(errors),
               'automatic_error_categories': dict(counts),
               'test_record_f1': None,
               'test_score_note': 'Obtain the hidden-test score from Gradescope; do not substitute the dev score.'}
    (reports / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    config_path = Path('checkpoints/ft_experiments/experiment/run_config.json')
    config = json.loads(config_path.read_text())
    lines = ['# Part 2 experiment evidence', '',
             'These are actual run outputs, not a completed written assignment.', '',
             '## Q5 configuration', '', '```json', json.dumps(config, indent=2), '```', '',
             'All pretrained T5-small parameters were fine-tuned. Text was stripped at line boundaries, with no truncation. Both sides use the default T5 tokenizer; decoder start is pad (0), followed by shifted SQL targets with EOS.', '',
             '## Q6 results', '',
             f'Dev Record F1: {100*record_f1:.3f}; Record EM: {100*record_em:.3f}; SQL EM: {100*sql_em:.3f}.', '',
             'Test Record F1: awaiting Gradescope. The dev result does not establish whether test F1 >=65.', '',
             '## Error evidence for manual analysis', '',
             'Automatic categories below classify execution messages and record mismatches. They do not identify the semantic cause of every wrong query. Manually inspect q6_dev_queries.csv and explain at least three observed error classes with examples, reasons, and COUNT/TOTAL as required.', '',
             '| Automatic category | COUNT/TOTAL |', '|---|---:|']
    for cat in ['SQL syntax error', 'Schema reference error', 'Other execution error or timeout', 'Executable SQL returning incorrect records', 'record_correct']:
        lines.append(f'| {cat} | {counts[cat]}/{len(rows)} |')
    for cat in ['SQL syntax error', 'Schema reference error', 'Other execution error or timeout', 'Executable SQL returning incorrect records']:
        samples = [row for row in rows if row['automatic_category'] == cat][:3]
        if samples:
            lines.extend(['', f'### {cat}'])
            for row in samples:
                lines.extend(['', f'Example {row["index"]}: {row["question"]}',
                              '', 'Gold:', '```sql', row['gold_sql'], '```',
                              'Predicted:', '```sql', row['predicted_sql'], '```',
                              f'Execution message: {row["execution_error"] or "none"}'])
    lines.extend(['', '## Remaining work', '',
                  '- Review Q4 statistics and their token-counting convention.',
                  '- Write Q5/Q6 explanations in your own words; do not invent ablations or unobserved error types.',
                  '- Get the test score from Gradescope and enter it in the written report.',
                  '- Provide your GitHub code link and Drive checkpoint link in the final PDF.',
                  '- This model-only checkpoint does not support exact optimizer/scheduler resumption.'])
    (reports / 'q5_q6_evidence.md').write_text('\n'.join(lines))

    test_count = len(Path('data/test.nl').read_text().splitlines())
    test_queries, test_records, test_errors = load_queries_and_records(
        'results/t5_ft_experiment_test.sql', 'records/t5_ft_experiment_test.pkl')
    assert len(test_queries) == len(test_records) == len(test_errors) == test_count
    best = Path('checkpoints/ft_experiments/experiment/best')
    T5TokenizerFast.from_pretrained('google-t5/t5-small').save_pretrained(best)
    submission = Path('submission')
    submission.mkdir(exist_ok=True)
    for source in [Path('results/t5_ft_experiment_test.sql'), Path('records/t5_ft_experiment_test.pkl')]:
        shutil.copy2(source, submission / source.name)
    manifest = {'test_examples': test_count, 'test_order': 'Unshuffled DataLoader; same order as data/test.nl',
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'run_config': config, 'best_checkpoint': str(best.resolve()),
                'submission_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in submission.iterdir() if p.is_file()}}
    (reports / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2))
    root = Path.cwd()
    with zipfile.ZipFile('part2_results_and_checkpoint.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for directory in [submission, reports, best]:
            for p in sorted(directory.rglob('*')):
                if p.is_file(): z.write(p, p.relative_to(root) if p.is_absolute() else p)
        for p in sorted(Path('.').glob('*.py')):
            z.write(p, 'code/' + p.name)
        for name in ['source_manifest.json', 'metrics_history.jsonl']:
            if Path(name).exists(): z.write(name)
    print('Final test file counts checked:', test_count, flush=True)
    print('Results/checkpoint archive:', (root / 'part2_results_and_checkpoint.zip'), flush=True)

if __name__ == '__main__':
    main()
