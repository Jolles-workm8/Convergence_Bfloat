import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("matrix_u[1,2]_frobenius.csv")
fp_u = []
Z0_u = []
Z1_u = []
Z2_u = []
Z3_u = []
Z4_u = []
for fp in df['fp32']:
    fp_u.append(fp)
for z in df['Z0']:
    Z0_u.append(z)
for z in df['Z1']:
    Z1_u.append(z)
for z in df['Z2']:
    Z2_u.append(z)
for z in df['Z3']:
    Z3_u.append(z)
for z in df['Z4']:
    Z4_u.append(z)




x =['2x2x2', '4x4x4', '8x8x8', '16x16x16', '32x32x32', '64x64x64']

plt.plot(x, fp_u, label = "fp32", linestyle="-")
plt.plot(x, Z0_u, label = "Z0", linestyle="--")
plt.plot(x, Z1_u, label = "Z1", linestyle="-.")
plt.plot(x, Z2_u, label = "Z2", linestyle=":")
plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Random uniform distribution on [1,2]')
plt.ylim(0.00000001, 0.01)
plt.savefig('uniform[1,2].png')
#plt.show()


df2 = pd.read_csv("matrix_range_frobenius.csv")

fp_rng = []
Z0_rng = []
Z1_rng = []
Z2_rng = []
Z3_rng = []
Z4_rng = []
for fp in df2['fp32']:
    fp_rng.append(fp)
for z in df2['Z0']:
    Z0_rng.append(z)
for z in df2['Z1']:
    Z1_rng.append(z)
for z in df2['Z2']:
    Z2_rng.append(z)
for z in df2['Z3']:
    Z3_rng.append(z)
for z in df2['Z4']:
    Z4_rng.append(z)

plt.plot(x, fp_rng, label = "fp32", linestyle="-")
plt.plot(x, Z0_rng, label = "Z0", linestyle="--")
plt.plot(x, Z1_rng, label = "Z1", linestyle="-.")
plt.plot(x, Z2_rng, label = "Z2", linestyle=":")
plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Random uniform distribution over [0,2^128]')
plt.ylim(0.00000001, 0.01)
plt.savefig('uniform[0,2^128].png')
#plt.show()
