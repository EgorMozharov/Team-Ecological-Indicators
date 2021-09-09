import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_S1_norm = pd.read_csv('S1_norm.csv', delimiter=',')
data_S2_norm = pd.read_csv('S2_norm.csv', delimiter=',')
data_S1 = pd.read_csv('S1.csv', delimiter=',')
data_S2 = pd.read_csv('S2.csv', delimiter=',')

# --------------------------------------
X = data_S2_norm['Algae cell density']
X_norm = data_S2_norm['Algae cell density norm']
u1 = data_S2_norm['Total phosphorus']
u2 = data_S2_norm['Total nitrogen']
u3 = data_S2_norm['Ammonia nitrogen']
u4 = data_S2_norm['Nirate nitrogen']
v1 = data_S2_norm['Temperature']
v2 = data_S2_norm['pH']
v3 = data_S2_norm['Alkalinity']
v4 = data_S2_norm['Turbidity']
U = (-0.6839) * u1 + 0.6114 * u2 + 0.9218 * u3 + 0.5557 * u4
V = (-1.0331) * v1 + 0.4553 * v2 + 0.7114 * v3 - 0.4683 * v4

# Solve by multi reg method. SOS
# tmp = []
# for i in range(0, len(U)):
#     tmp.append(-2*U[i]*X[i])
#     tmp.append(-V[i])
#     tmp.append(1)
#
# B = np.array(tmp)
# B = B.reshape(19, 3)
#
# tmp = []
# for i in range(0, len(U)):
#     tmp.append(4*(X[i]**3))
# b = np.array(tmp)
#
#
# answer = np.linalg.solve(B, b)
# print(answer)
# ans = - 1.9586, - 0.3610,  −0.3914
K = [-1.9586, -0.3610, -0.3914]

#----------------------------------
# fitted(a)

tmp = []
plt.figure(figsize=(10, 5))
for i in range(0, len(data_S2_norm['Date'])):
    tmp.append(K[0]*(-2*U[i]*X_norm[i])+K[1]*(-V[i])+K[2])
plt.scatter(data_S2_norm['Date'], tmp, color='green', label='DCV')
plt.plot(data_S2_norm['Date'], tmp, "g--", alpha=0.4)
# Normalized
tmp = 4*((data_S2_norm['Algae cell density norm'])**3)
plt.plot(data_S2_norm['Date'], tmp, label='Observed value')
plt.ylabel("Normalized algae cell denisity")
plt.xlabel("Date")
plt.legend()
plt.show()

# fitted(b)
tmp = []
plt.figure(figsize=(10, 5))
for i in range(0, len(data_S2_norm['Date'])):
    tmp.append(K[0]*(-2*u1[i]*X_norm[i])+K[1]*(-V[i])+K[2])
plt.scatter(data_S2_norm['Date'], tmp, color='green', label='TP-CV')
plt.plot(data_S2_norm['Date'], tmp, "g--", alpha=0.4)
# Normalized
tmp = 4*((data_S2_norm['Algae cell density norm'])**3)
plt.plot(data_S2_norm['Date'], tmp, label='Observed value')
plt.ylabel("Normalized algae cell denisity")
plt.xlabel("Date")
plt.legend()
plt.show()
#--------------------------------
# DCCPI

DCCPI=[]

for i in range(0,len(data_S2_norm['Date'])):
    DCCPI.append(8*(K[0]*u1[i])**3 + 27*(K[1]*V[i]-K[2])**2)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(data_S2_norm['Date'], DCCPI, color='green', label='DCCPI')
tmp = 4*((data_S2_norm['Algae cell density norm'])**3)
ax2.plot(data_S2_norm['Date'], tmp, 'b-', label='Observed value')

ax1.set_xlabel("Date")
ax1.legend()
ax1.set_ylabel("DCCPI", color='g', name='fefe')
ax2.set_ylabel("Observed", color='b')
plt.figure(figsize=(10, 5))
plt.show()
