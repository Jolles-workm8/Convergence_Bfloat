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
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())

df = pd.read_csv("4x4_matrix_frobenius.csv")
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())

df = pd.read_csv("8x8_matrix_frobenius.csv")
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())

df = pd.read_csv("16x16_matrix_frobenius.csv")
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())

df = pd.read_csv("32x32_matrix_frobenius.csv")
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())

df = pd.read_csv("64x64_matrix_frobenius.csv")
fp_min.append(df['fp32'].min())
Z0_min.append(df['Z0'].min())
Z1_min.append(df['Z1'].min())
Z2_min.append(df['Z2'].min())

fp_max.append(df['fp32'].max())
Z0_max.append(df['Z0'].max())
Z1_max.append(df['Z1'].max())
Z2_max.append(df['Z2'].max())


x =['2x2x2', '4x4x4', '8x8x8', '16x16x16', '32x32x32', '64x64x64']

plt.plot(x, fp_min, label = "fp32", linestyle="-", color="darkred")
plt.plot(x, fp_max, linestyle="-", color="darkred")
plt.plot(x, Z0_min, label = "Z0", linestyle="--", color="darkorange")
plt.plot(x, Z0_max, linestyle="--", color="darkorange")
plt.plot(x, Z1_min, label = "Z1", linestyle="-.", color="darkblue")
plt.plot(x, Z1_max, linestyle="-.", color="darkblue")
plt.plot(x, Z2_min, label = "Z2", linestyle=":", color="darkgreen")
plt.plot(x, Z2_max, linestyle=":", color="darkgreen")
plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Random uniform distribution on [0,max(float)]')
plt.ylim(0.0000000001, 0.1)
plt.savefig('u0max_minmax.png')
#plt.show()
