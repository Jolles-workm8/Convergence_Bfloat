import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



fp_worst = []
fp_best = []
Z0_worst = []
Z0_best = []
Z1_worst = []
Z1_best = []
Z2_worst = []
Z2_best = []


df1 = pd.read_csv("../Uniform0,max/2x2_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/2x2_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)


df1 = pd.read_csv("../Uniform0,max/4x4_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/4x4_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)


df1 = pd.read_csv("../Uniform0,max/8x8_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/8x8_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)


df1 = pd.read_csv("../Uniform0,max/16x16_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/16x16_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)


df1 = pd.read_csv("../Uniform0,max/32x32_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/32x32_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)


df1 = pd.read_csv("../Uniform0,max/64x64_matrix_frobenius.csv")
df2 = pd.read_csv("../Uniform-1,1/64x64_matrix_frobenius.csv")

fp_worst.append(df1['fp32'].sum()/1000)
fp_best.append(df2['fp32'].sum()/1000)
Z0_worst.append(df1['Z0'].sum()/1000)
Z0_best.append(df2['Z0'].sum()/1000)
Z1_worst.append(df1['Z1'].sum()/1000)
Z1_best.append(df2['Z1'].sum()/1000)
Z2_worst.append(df1['Z2'].sum()/1000)
Z2_best.append(df2['Z2'].sum()/1000)



x =['2x2x2', '4x4x4', '8x8x8', '16x16x16', '32x32x32', '64x64x64']

plt.plot(x, fp_best, label = "fp32 u[-1,1]", linestyle="-", color="darkred")
plt.plot(x, fp_worst, label = "fp32 u[0,max]", linestyle="-", color="orangered")
plt.plot(x, Z0_best, label= "Z0 u[-1,1]", linestyle="--", color="darkorange")
plt.plot(x, Z0_worst, label = "Z0 u[0,max]",linestyle="--", color="yellow")
plt.plot(x, Z1_best, label= "Z1 u[-1,1]",linestyle="-.", color="darkblue")
plt.plot(x, Z1_worst, label = "Z1 u[0,max]",linestyle="-.", color="deepskyblue")
plt.plot(x, Z2_best, label= "Z2 u[-1,1]",linestyle=":", color="darkgreen")
plt.plot(x, Z2_worst, label = "Z2 u[0,max]",linestyle=":", color="lime")

plt.yscale('log')
plt.xlabel('matrix sizes')
plt.ylabel('accuracy')
plt.legend()
plt.title('Comparison between u[-1,1] and u[0,max(float)]')
plt.ylim(0.00000001, 0.01)
plt.savefig('comparison_avg.png')
#plt.show()
