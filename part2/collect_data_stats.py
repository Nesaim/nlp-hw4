"""Q4 evidence using the same T5 tokenizer; no invented experiment numbers."""
import json
import statistics
from pathlib import Path
from transformers import T5TokenizerFast

def main():
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    result = {'convention': 'Lengths include EOS; vocabulary counts observed non-special token IDs. Before: newline removed; after: strip(), as in load_data.py. No truncation or augmentation.', 'splits': {}}
    for split in ['train', 'dev']:
        stats = {}
        for extension in ['nl', 'sql']:
            lines = Path(f'data/{split}.{extension}').read_text().splitlines()
            versions = {'before': lines, 'after': [line.strip() for line in lines]}
            for version, texts in versions.items():
                ids = tokenizer(texts, truncation=False, padding=False,
                                add_special_tokens=True)['input_ids']
                lengths = [len(row) for row in ids]
                stats[f'{extension}_{version}'] = {
                    'number_of_examples': len(ids),
                    'mean_token_length': statistics.mean(lengths),
                    'max_token_length': max(lengths),
                    'observed_vocabulary_size': len({i for row in ids for i in row if i not in tokenizer.all_special_ids}),
                    'unknown_token_count': sum(row.count(tokenizer.unk_token_id) for row in ids),
                    'examples_with_unknown_tokens': sum(tokenizer.unk_token_id in row for row in ids),
                }
        assert stats['nl_after']['number_of_examples'] == stats['sql_after']['number_of_examples']
        result['splits'][split] = stats
    Path('reports').mkdir(exist_ok=True)
    Path('reports/q4_data_stats.json').write_text(json.dumps(result, indent=2))
    lines = ['# Q4 data statistics', '', result['convention'], '']
    for version in ['before', 'after']:
        lines.extend([f'## {version}', '', '| Statistic | Train | Dev |', '|---|---:|---:|'])
        for label, extension, key in [
            ('Number of examples', 'nl', 'number_of_examples'),
            ('Mean NL token length', 'nl', 'mean_token_length'),
            ('Mean SQL token length', 'sql', 'mean_token_length'),
            ('NL observed vocabulary', 'nl', 'observed_vocabulary_size'),
            ('SQL observed vocabulary', 'sql', 'observed_vocabulary_size'),
            ('SQL examples with unknown tokens', 'sql', 'examples_with_unknown_tokens'),
        ]:
            values = [result['splits'][s][f'{extension}_{version}'][key] for s in ['train', 'dev']]
            lines.append(f'| {label} | {values[0]} | {values[1]} |')
        lines.append('')
    lines.append('Unknown SQL tokens are a baseline limitation; retain this observation in the report. A passing smoke check does not resolve it.')
    Path('reports/q4_data_stats.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
