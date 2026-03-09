import numpy as np
import pandas as pd
import random

# --- Part 1: Data Generation (from Notebook) ---
n = 60
b0_true = 5
b1_true = np.array([2, -3])
mue = 0
sigmae = 5
xl, xh = 0, 10

def genSample(n, b0, b1, sigmae, xLow, xHigh, seedit=199, size=1):
    if type(seedit) == int:
        np.random.seed(seedit)
        Er = np.random.normal(mue, sigmae, n)
        x = []
        for k in range(size):
            np.random.seed(seedit + k)
            x.append(np.random.uniform(xl, xh, n))
    else:
        np.random.seed()
        Er = np.random.normal(mue, sigmae, n)
        x = []
        for k in range(size):
            np.random.seed()
            x.append(np.random.uniform(xl, xh, n))
            
    y = b0 + Er
    for k in range(size):
        y += b1[k] * x[k]
    
    if size == 1:
        return (x[0], y, Er)
    else:
        return (x, y, Er)

# Generate Data
(x, y, Er) = genSample(n, b0_true, b1_true, sigmae, xl, xh, seedit=199, size=2)

# --- Part 2: SGD Implementation ---

def LinReg_SGD(T, m, eta, printit=True):
    # Initialize
    np.random.seed(42) # For reproducibility in this test
    b1_init = np.random.uniform()
    b2_init = np.random.uniform()
    b0_init = np.random.uniform()
    
    b0 = b0_init
    b1 = b1_init
    b2 = b2_init
    
    b0List = [b0]
    b1List = [b1]
    b2List = [b2]
    
    for t in range(T):
        # Select mini-batch
        if m < n:
            indx = np.random.choice(np.arange(n), size=m, replace=False)
        else:
            indx = np.arange(n)
            
        y_batch = y[indx]
        # x is a list of arrays: [x1_array, x2_array]
        x_batch1 = x[0][indx]
        x_batch2 = x[1][indx]
        
        # Prediction
        y_hat = b0 + b1*x_batch1 + b2*x_batch2
        
        # Errors
        errors = y_batch - y_hat
        
        # Gradients
        grad_b0 = -2/m * np.sum(errors)
        grad_b1 = -2/m * np.sum(errors * x_batch1)
        grad_b2 = -2/m * np.sum(errors * x_batch2)
        
        # Update
        b0 = b0 - eta * grad_b0
        b1 = b1 - eta * grad_b1
        b2 = b2 - eta * grad_b2
        
        b0List.append(b0)
        b1List.append(b1)
        b2List.append(b2)
        
    return b0, b1, b2

# --- Part 3: Verification Experiments ---
print(f"True Parameters: b0={b0_true}, b1={b1_true[0]}, b2={b1_true[1]}")

experiments = [
    (1, 0.04), (1, 0.01), (1, 0.001),
    (10, 0.04), (10, 0.01), (10, 0.001),
    (60, 0.04), (60, 0.01), (60, 0.001)
]

for m, eta in experiments:
    try:
        b0_est, b1_est, b2_est = LinReg_SGD(1000, m, eta, printit=False)
        print(f"m={m:2d}, eta={eta:.3f} -> b0={b0_est:.3f}, b1={b1_est:.3f}, b2={b2_est:.3f}")
    except Exception as e:
        print(f"m={m:2d}, eta={eta:.3f} -> Failed/Diverged ({e})")
