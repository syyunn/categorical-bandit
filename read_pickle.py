import pickle
from env import CategoricalBanditEnv

with open("/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment/results_K112_C26_N2000_B_5_L1_priorTrue_priorTemp_0_expid0/env.pickle", "rb") as f:
    envp = pickle.load(f)
    print("hi")