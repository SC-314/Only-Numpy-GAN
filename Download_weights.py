import numpy as np
# Loading weights
loaded_weights = np.load(r'C:\Users\Sam\Desktop\Generative Adversarial Network\final_weights.npz')

DW0 = loaded_weights['DW0']
DW2 = loaded_weights['DW2']
DW4 = loaded_weights['DW4']
GW0 = loaded_weights['GW0']
GW1 = loaded_weights['GW1']
GW2 = loaded_weights['GW2']
GW3 = loaded_weights['GW3']
GW4 = loaded_weights['GW4']
GW5 = loaded_weights['GW5']
GW6 = loaded_weights['GW6']

print("Weights loaded successfully.")
