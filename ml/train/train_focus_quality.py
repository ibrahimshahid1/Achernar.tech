"""Train focus quality estimator (FQE)."""
import argparse


def train(config: dict):
    raise NotImplementedError("Implement FQE training loop with PyTorch Lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(vars(args))
