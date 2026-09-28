from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def source_question_id(row):
    for key in ('_id', 'id', 'question_id'):
        if row.get(key) is not None and str(row[key]):
            return str(row[key])
    raise ValueError('Source dataset lacks an original question ID')


def select_2wiki(records, start=0, count=1000):
    if start < 0 or count < 1:
        raise ValueError('Invalid subset range')
    indexed = [(source_question_id(row), index, row) for index, row in enumerate(records)]
    if len({x[0] for x in indexed}) != len(indexed):
        raise ValueError('Duplicate original question IDs')
    indexed.sort(key=lambda x: x[0])
    if start + count > len(indexed):
        raise ValueError('Dataset does not contain the complete requested subset')
    return [dict(row, question_id=qid, source_question_id=qid, source_index=index)
            for qid, index, row in indexed[start:start + count]]


def export_manifest_ids(manifest, output):
    ids = manifest.get('question_ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError('A frozen manifest with nonempty string question_ids is required')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs')
    source_ids = manifest.get('source_question_ids', [''] * len(ids))
    indices = manifest.get('source_indices', [''] * len(ids))
    if len(source_ids) != len(ids) or len(indices) != len(ids):
        raise ValueError('Manifest ID and source-index lengths do not match')
    subset = manifest.get('subset', '')
    if subset not in ('', 'B0', 'B1', 'B2', 'custom', 'MuSiQue'):
        raise ValueError('Unsupported subset label')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['subset', 'question_id', 'source_question_id', 'source_index'])
        writer.writerows((subset, qid, source_id, index) for qid, source_id, index in zip(ids, source_ids, indices))


def main():
    parser = argparse.ArgumentParser(description='Export exact question IDs from an existing frozen experiment manifest.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_manifest_ids(json.loads(args.manifest.read_text()), args.output)


if __name__ == '__main__':
    main()
