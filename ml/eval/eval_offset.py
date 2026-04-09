"""Evaluate focus offset predictor."""

def evaluate(model_path: str, test_data: str):
    raise NotImplementedError("Implement offset evaluation with MAE, RMSE")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test_data)
