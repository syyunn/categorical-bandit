import matplotlib.pyplot as plt
import numpy as np
from main import experiment

if __name__ == "__main__":
    # number of categories in LDA (so called "general issue code", e.g. TRA = Transportation) = 79
    # number of legislators in US Congress
        # There are a total of 535 Members of Congress. 100 serve in the U.S. Senate and 435 serve in the U.S. House of Representatives.
    experiment(N=5000, C=79, K=535, show=True)