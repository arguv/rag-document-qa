"""Evaluate a JSONL export of retrieval and answer predictions, entirely offline."""
import argparse
import json
from statistics import mean
from app.application.evaluation import retrieval_metrics, answer_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    results = []
    with open(args.dataset, encoding='utf-8') as source:
        for line in source:
            if line.strip():
                case = json.loads(line)
                scores = retrieval_metrics(case['retrieved'], case['relevant'], args.k)
                if 'answer' in case and 'expected_answer' in case:
                    scores.update(answer_metrics(case['answer'], case['expected_answer']))
                results.append(scores)
    if not results:
        parser.error('Dataset is empty')
    keys = set().union(*results)
    print(json.dumps({'cases': len(results), 'metrics': {key: mean(row[key] for row in results if key in row) for key in sorted(keys)}}, indent=2))


if __name__ == '__main__':
    main()
