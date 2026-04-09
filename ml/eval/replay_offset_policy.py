"""
replay_offset_policy — Section 7.10.3.
Offline policy replay for evaluating focus offset policies on historical data.
"""

def replay(policy_path: str, historical_runs: str):
    raise NotImplementedError("Implement offline policy replay — Section 7.10.3")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    replay(args.policy, args.runs)
