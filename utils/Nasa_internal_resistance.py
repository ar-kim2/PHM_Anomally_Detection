import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.Entropy import Moving_Avg_Filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import operator
from scipy import integrate


dir_path = "../data/Nasa_data/BatteryAgingARC_change/B0025"
dir_path2 = "../data/Nasa_data/BatteryAgingARC_change/B0026"
dir_path3 = "../data/Nasa_data/BatteryAgingARC_change/B0028"

check_ri = np.load(dir_path + '/check_ri.npy', allow_pickle=True)
check_ri2 = np.load(dir_path2 + '/check_ri.npy', allow_pickle=True)
check_ri3 = np.load(dir_path3 + '/check_ri.npy', allow_pickle=True)

cap = np.load(dir_path + '/capacity.npy', allow_pickle=True)
cap2 = np.load(dir_path2 + '/capacity.npy', allow_pickle=True)
cap3 = np.load(dir_path3 + '/capacity.npy', allow_pickle=True)

Ri_Exptend = np.append(check_ri, check_ri2)
Ri_Exptend = np.append(Ri_Exptend, check_ri3)

Cap_Extend = np.append(cap, cap2)
Cap_Extend = np.append(Cap_Extend, cap3)

# # B0005
# capacity = [1.8148, 1.7730, 1.5547, 1.4124, 1.3090]
# resistance = [0.3847, 0.3552, 0.3682, 0.3707, 0.3752]
#
# # B0006
# capacity = [ 1.7971, 1.7024, 1.5041, 1.2532, 1.175]
# resistance = [ 0.3747, 0.3738, 0.3813, 0.3914, 0.3984]
#
#
# # B0006
# capacity = [ 1.8043, 1.4964, 1.3462]
# resistance = [ 0.3327, 0.3345, 0.3469]
#
# # B0006
# capacity = [ 1.6348, 1.4368, 1.4034, 1.3177, 1.5979, 0.0687]
# resistance = [ 0.3983, 0.4253, 0.4487, 0.412, 0.3512, 0.4017]
#
#
# capacity = [1.8148, 1.7730, 1.5547, 1.4124, 1.3090,  1.7971, 1.7024, 1.5041, 1.2532, 1.175, 1.8043, 1.4964, 1.3462, 1.6348, 1.4368, 1.4034, 1.3177, 1.5979, 0.0687]
# resistance = [0.3847, 0.3552, 0.3682, 0.3707, 0.3752, 0.3747, 0.3738, 0.3813, 0.3914, 0.3984, 0.3327, 0.3345, 0.3469,  0.3983, 0.4253, 0.4487, 0.412, 0.3512, 0.4017]



SOH=[]
for i in range(len(Cap_Extend)):
    SOH.append(Cap_Extend[i]/2)


print(np.shape(SOH))
print(SOH)

print(np.shape(Ri_Exptend))
print(Ri_Exptend)



def exp_fun(x,a, b):
 return a*x + b

popt, pcov = curve_fit(exp_fun, SOH, Ri_Exptend, p0=[0, 0])

print(popt)

x_idx = np.linspace(0, 1, 10)

print(x_idx)

plt.figure()
plt.plot(exp_fun(x_idx, popt[0], popt[1]))
plt.scatter(SOH,Ri_Exptend)
# plt.ylim(0,70)
#plt.plot(x,y)

plt.show()




