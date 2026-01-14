import argparse
import json
import os
from lib.hybrid_search import (
    ranked_search
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

def open_file():
    try:
        with open(DATA_PATH, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        return []
    
def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    # run evaluation logic here
    golden_dataset = open_file()
    for test_case in golden_dataset["test_cases"]:
        # each test_case has query and relevant_docs
        results = ranked_search(test_case["query"], limit=args.limit)

        result_titles = [res[1]["document"]["title"] for res in results]
        intersection = set(result_titles) & set(test_case["relevant_docs"])

        precision = len(intersection) / args.limit
        recall = len(intersection) / len(test_case["relevant_docs"])
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {test_case["query"]}\n- Precision@{args.limit}: {precision}\n- Recall@{args.limit}: {recall}\n- F1 Score: {f1}\nCalculated results: {", ".join(result_titles)}\nResults should include: {", ".join(test_case["relevant_docs"])}\n")



if __name__ == "__main__":
    main()