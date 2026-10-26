import numpy as np
from n1_learners import run_learner, generate_channel_window, T_S

def main():
    H = generate_channel_window(12345, 0.99, 1.4, T_S, 1000)
    new = np.array(run_learner("seq_warm", H, 12345, 0.99, 1.4, U=10, n_run=60)["minrs"])
    old = np.array(np.load("block1_single_results.npy", allow_pickle=True).item()[("seqwarm_aware", 0.99, 12345)]["minrs"][:60])
    print("max |new - Block1 stored| over first 60 steps: %.2e" % np.abs(new - old).max())
    print("new[:5] ", np.round(new[:5], 4)); print("old[:5] ", np.round(old[:5], 4))
    print("REPRODUCES" if np.abs(new - old).max() < 1e-6 else "MISMATCH")

if __name__ == "__main__":
    main()
