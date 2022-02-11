import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



fp_min = []
fp_max = []
Z0_min = []
Z0_max =[]
Z1_min =[]
Z1_max = []
Z2_min = []
Z2_max = []

df = pd.read_csv("2x2_matrix_frobenius.csv")
fp_min.append(df['fp32'].max())
print(fp_min)
df = pd.read_csv("4x4_matrix_frobenius.csv")

df = pd.read_csv("8x8_matrix_frobenius.csv")

df = pd.read_csv("16x16_matrix_frobenius.csv")

df = pd.read_csv("32x32_matrix_frobenius.csv")

df = pd.read_csv("64x64_matrix_frobenius.csv")



x =['2x2x2', '4x4x4', '8x8x8', '16x16x16', '32x32x32', '64x64x64']

plt.plot(x, fp_u, label = "fp32", linestyle="-")
plt.plot(x, Z0_u, label = "Z0", linestyle="--")
plt.plot(x, Z1_u, label = "Z1", linestyle="-.")
plt.plot(x, Z2_u, label = "Z2", linestyle=":")
plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Random uniform distribution on [-1,1]')
plt.ylim(0.00000001, 0.01)
plt.savefig('repeat.png')
#plt.show()
