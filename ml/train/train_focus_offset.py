"""Train focus offset MLP (distillation from LightGBM teacher)."""
import argparse


def train(config: dict):
    raise NotImplementedError("Implement offset MLP training with distillation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-model", help="Path to LightGBM teacher model")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(vars(args))
