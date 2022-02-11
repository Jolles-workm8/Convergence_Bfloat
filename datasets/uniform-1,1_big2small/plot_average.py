import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



fp = []
Z0 = []
Z1 =[]
Z2 = []


df = pd.read_csv("2x2_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)


df = pd.read_csv("4x4_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)


df = pd.read_csv("8x8_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)


df = pd.read_csv("16x16_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)


df = pd.read_csv("32x32_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)


df = pd.read_csv("64x64_matrix_frobenius.csv")
fp.append(df['fp32'].sum()/1000)
Z0.append(df['Z0'].sum()/1000)
Z1.append(df['Z1'].sum()/1000)
Z2.append(df['Z2'].sum()/1000)



x =['2x2x2', '4x4x4', '8x8x8', '16x16x16', '32x32x32', '64x64x64']

plt.plot(x, fp, label = "fp32", linestyle="-", color="darkred")
plt.plot(x, Z0, label = "Z0", linestyle="--", color="darkorange")
plt.plot(x, Z1, label = "Z1", linestyle="-.", color="darkblue")
plt.plot(x, Z2, label = "Z2", linestyle=":", color="darkgreen")

plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Random uniform distribution on [-1,1], wrong order')
plt.ylim(0.00000001, 0.01)
plt.savefig('u-11_av_wo.png')
#plt.show()
