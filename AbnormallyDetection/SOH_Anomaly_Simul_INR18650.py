from AbnormallyDetection.dataloader import ReadData_first
from AbnormallyDetection.dataloader import get_KnD
from AbnormallyDetection.dataloader import SOCcurve
import os
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import BatterySpec
from random import *

'''
INR-18650 배터리 데이터에 시뮬레이터로 이상치를 추가하고
이상징후를 검출한 결과
'''

# #
# # Model_Dir = os.path.abspath('./Model')
# # Model_Dir = os.path.join(Model_Dir, 'SOH_iteration6000_seq20_unit5_f2_predict1_withCoeffV34T35_drop05_07')
# # Estimator = SOHestimator(data_load=True, feature_size=2, drop_out=True) # SOH 추정 모델 클래스 로드
# # Estimator.load_model(model=Model_Dir) # 학습된 추정 모델 로드
# #
# # selected_cycle = 200 # Anomaly Detection 수행할 Cycle. Estimator가 해당 cycle의 SOH추정
# # ####################################둘 중 하나 선택#########################################
# # # 방전전류가 일정한 데이터셋
# # # test_battery_list = ['CS2_33']
# # # CAP_data, DC_Entropy, DC_prob, C_entropy = Estimator.data_loader(test_battery_list)
# #
# # # 방전전류가변화하는 데이터셋
# # test_battery_list = ['CS2_3']
# # CAP_data, DC_Entropy, DC_prob, C_entropy = Estimator.data_loader_changing(test_battery_list)
# # ############################################################################################
# # # 엔트로피 인덱스로부터 input 생성
# # InputEntropy = Entropy.concatenateEntropy_withdischarge(DC_Entropy, C_entropy)
# # #############################################################################################
# # # Entmin_value= Estimator.session.run("Ent_min_value:0")
# # # Entmax_value= Estimator.session.run("Ent_max_value:0")
# # # InputEntropy = tools.MinMaxScaler(InputEntropy, Entmin_value, Entmax_value)
# # InputEntProb = InputEntropy
# #
# # def train_data_make(result_list, input_list, future_len, seq_len, pred_cyc):
# #     # Input [?, seq_length, feature_size] 형태로 변환
# #     appended_list = np.append(np.full((seq_len - 1, np.shape(input_list)[-1]), input_list[0], dtype=np.float32),
# #                               input_list, axis=0)
# #     appended_list = [appended_list[idx:idx + seq_len + future_len] for idx in
# #                      range(len(appended_list) + 1 - seq_len - pred_cyc)]
# #
# #     if result_list is None:
# #         result_list = appended_list
# #     else:
# #         result_list.extend(appended_list)
# #
# #     return result_list, appended_list
# #
# # TestP1_1 = None
# # TestLabelP1 = []
# # for i, _data in enumerate(InputEntProb):
# #     TestP1_1, _ = train_data_make(result_list=TestP1_1,
# #                                       input_list=_data,
# #                                       future_len=0,
# #                                       seq_len=Estimator.seq_length,
# #                                       pred_cyc=0)
# #
# #     TestLabelP1.extend(CAP_data[i][:, 0])
# #
# # ############################### 전체 cycle에서 SOH추정 결과 #################################
# # # Estimated_SOH = Estimator.session.run(Estimator.out_list[-1],
# # #                       feed_dict={Estimator.X : TestP1_1,
# # #                                  Estimator.keep_prob: 1})
# # # ln1 = plt.plot(TestLabelP1, label='real capacity')
# # # ln2 = plt.plot(Estimated_SOH, label ='estimated cpacity')
# # # lns = ln1 + ln2
# # # labs = [l.get_label() for l in lns]
# # plt.legend(lns, labs, loc=9,fontsize=10)
# # plt.xlabel('Cycles', fontsize=14)
# # plt.ylabel('Capacity(Ah)', fontsize=14)
# # plt.ylim([0.2, 1.2])
# # plt.show()
# #########################################################################################3
# print(Estimator.session.run(Estimator.rmse, feed_dict={Estimator.X : TestP1_1,
#                                                        Estimator.Y: np.array(TestLabelP1)[:, np.newaxis],
#                                                        Estimator.keep_prob:1}))
# Estimated_SOH = Estimator.session.run(Estimator.out_list[-1],
#                       feed_dict={Estimator.X : np.array(TestP1_1[selected_cycle])[np.newaxis, :, :],
#                                  Estimator.keep_prob:1})
# print(Estimated_SOH[0,0])
# # calculate SOH at cycle k => SOH_k : 구현완료
# ########################################################################################


# 0 < SOH <= rated_capacity
# SOH_k = Estimated_SOH[0,0] # estimated SOH at k, it would be estimate based on deep learning model..
SOH_k = 0.9 # estimated SOH at k, it would be estimate based on deep learning model..

# Read Data from first cycle
# train_battery_list = ['CS2_33']
# dir_path = "./data/dis_current_constant/CS2_XX_0"

train_battery_list = ['base_ocv']
dir_path = "../data/battery_device"

for battery in train_battery_list:
    print('====Reading {} Battery Data for Train ===='.format(battery))
    BatteryDataDir = os.path.join(dir_path, battery)
    FileNameList = os.listdir(BatteryDataDir)

    # SOC-OCV curve 위해 1번째 데이터 load
    K_1, D_1, SOC_OCV_1, Capacity_1, T_1, true_OCV = ReadData_first(FileNameList, BatteryDataDir)

    OCV_base, SOC_base = SOCcurve(SOC_OCV_1, Capacity_1, T_1, 1 * BatterySpec.spec_full_capacitor)
    OCV_k, SOC_k = SOCcurve(SOC_OCV_1, Capacity_1, T_1, SOH_k * BatterySpec.spec_full_capacitor)        # charging direction

    K_base, D_base = get_KnD(OCV_base, SOC_base)  # discharging direction
    K_k, D_k = get_KnD(OCV_k, SOC_k) # discharging direction

    # 양 옆에 그래프 지움.
    for i in range(10):
        K_k = np.delete(K_k, 0)
        D_k = np.delete(D_k, 0)
        K_k = np.delete(K_k, -1)
        D_k = np.delete(D_k, -1)

        K_base = np.delete(K_base, 0)
        D_base = np.delete(D_base, 0)
        K_base = np.delete(K_base, -1)
        D_base = np.delete(D_base, -1)

        SOC_k = SOC_k[1:-1]
        SOC_base = SOC_base[1:-1]

    plt.figure()
    plt.plot(np.flip(OCV_base, 0).reshape(-1), label='base battery')
    plt.plot(np.flip(OCV_k, 0).reshape(-1) , label='test battery')
    plt.xlabel('Time step')
    plt.ylabel('OCV')
    plt.title('OCV Curve')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.flip(SOC_base, 0).reshape(-1), label='base battery')
    plt.plot(np.flip(SOC_k, 0).reshape(-1), label='test battery')
    plt.xlabel('Time step')
    plt.ylabel('SOC')
    plt.title('SOC Curve')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(SOC_base, K_base, label='base battery k')
    plt.plot(SOC_base, D_base, label='base battery d')
    plt.plot(SOC_k, K_k, label='test battery k')
    plt.plot(SOC_k, D_k, label='test battery d')
    plt.xlabel('Soc')
    plt.title('K, D')
    plt.legend()
    plt.show()

    soc_k_mapping = InterpolatedUnivariateSpline(SOC_k, np.flip(K_k, 0), k=2)
    soc_d_mapping = InterpolatedUnivariateSpline(SOC_k, np.flip(D_k, 0), k=2)

    break

########################### CPF ###########################
rated_capacity = 3.35 # BatterySpec.spec_full_capacitor
init_soc = SOH_k-0.02 # / rated_capacity
init_data_size = len(K_1)

Ccs = 90
Ccb = 12240  # Ccb = 3370.7558     1.1 * 3600  ,  3.4 * 3600
Rt = 0.06

# internal Resistance with calculated capacity
init_Ri = (-0.0357 * SOH_k) + 0.17
#init_Ri = 0.7684 - SOH_k * 0.5349

Ts = 1

n_iter = int(init_data_size*30)             # 30초 sampling한 data로 simulation하면서 T =1 로 설정했기 때문에, k,d구하면서 잘라낸 부분 고려하면 *50정도는 해주어야 배터리 노화가 simuation된다.

kp = soc_k_mapping(init_soc)
d = soc_d_mapping(init_soc)

Ad = np.array([[1, 0],
               [0, 1 - (Ts / (Rt * Ccs))]])
Bd = np.array([[Ts / (kp * Ccb)],
               [Ts / Ccs]])
Cd = np.array([[kp, 1]])
Dd = init_Ri
Ri = init_Ri

AdSim = np.array([[1, 0],
                  [0, 1 - (Ts / (Rt * Ccs))]])
BdSim = np.array([[Ts / (kp * Ccb)],
                  [Ts / Ccs]])
CdSim = np.array([[kp, 1]])

DdSim = init_Ri

dSim = d
#####################################################
Q = np.zeros((n_iter, 2, 2))
for i in range(0, n_iter):
    Q[i] = np.array([[0.000004, 0], [0, 0.00001]])

R = np.zeros((n_iter,))
for i in range(0, n_iter):
    R[i] = 0.01 #   1
# variance in Simulation, different with Q
Qsim = np.zeros((n_iter, 2, 2))
for i in range(0, n_iter):
    Qsim[i] = np.array([[0.003, 0], [0, 0.001]])

Rsim = np.zeros((n_iter,))
for i in range(0, n_iter):
    Rsim[i] = 0.01
####################################################
xsimulation = np.zeros((n_iter, 2, 1))


xsimulation[0] = np.array([[init_soc],
                           [0]])  # 3.2172783618548144
xsimulation_noisy = np.zeros((n_iter, 2, 1))
xsimulation_noisy[0] = np.array([[init_soc],
                                 [0]])

ysimulation = np.zeros((n_iter,))
ysimulation[0] = 4.1

ysimulation_noisy = np.zeros((n_iter,))
ysimulation_noisy[0] = 4.1
#####################################################
u = np.zeros((n_iter,))
for i in range(0, n_iter):
    u[i] = -0.5

xhat = np.zeros((n_iter, 2, 1))
xhat[0] = np.array([[init_soc],
                    [0]])


P = np.zeros((n_iter, 2, 2))          # a posteri error estimate
P[0] = np.array([[1, 0], [0, 1]])
xhatminus = np.zeros((n_iter, 2, 1))  # a priori estimate of x
Pminus = np.zeros((n_iter, 2, 2))     # a priori error estimate
K = np.zeros((n_iter, 2, 1))          # gain or blending factor
yhat = np.zeros((n_iter,))
yhat[0] = 4.1
SigmaY = np.zeros((n_iter,))

xaugmentedhat = np.zeros((n_iter, 3, 1))
xaugmentedfuse = np.zeros((n_iter, 3, 1))
Paugmented = np.zeros((n_iter, 3, 3))
M = np.zeros((n_iter, 3, 2))
M[:, :2, :2] = np.identity(2)

dist = np.zeros((n_iter,))
dist1 = np.zeros((n_iter,))
dist2 = np.zeros((n_iter,))

_Pminus = np.matrix([[0, 0],
                     [0, 0]])

_xaugmentedhat = np.matrix([[0],
                            [0],
                            [0]])

_xaugmentedfuse = np.matrix([[0],
                             [0],
                             [0]])

_Paugmented = np.matrix([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])
_M = np.matrix([[1, 0],
                [0, 1],
                [0, 0]])

xtide = np.zeros((n_iter, 3, 1))
Ptide = np.zeros((n_iter, 3, 3))

k_sequence_record = np.zeros((n_iter,))
d_sequence_record = np.zeros((n_iter,))
k_sequence_record[0] = Cd[0, 0]
d_sequence_record[0] = d

k_sequence_model_record = np.zeros((n_iter,))
d_sequence_model_record = np.zeros((n_iter,))
k_sequence_model_record[0] = Cd[0, 0]
d_sequence_model_record[0] = d

# For Simulation, abnormal size
ab_size = 0.01

isSaveData = False
saveXDataName = 'xdata10_5000_8000'
saveYDataName = 'data10_5000_8000'

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def AdaptiveCovarianceProjectionFilter(Ad, Bd, Cd, Dd, Ri, u, d, P, Q, R, xhat, ytrue, k):
    '''
    # ZHAO comments:
    # Two sources of estimation are stored in variable xhatminus and ytrue
    # Their variance matrices are Pminus and R, respectively
    # M is the constraint, which you don't need to worry about
    # xtide is the fused estimation
    # Ptide is the variance of xtide
    # dist can be used to detect anomaly
    # dist1 can be used to detect whether xhatminus is the anomaly source
    # dist2 can be used to detect whether ytrue is the anomaly source
    '''
    global Ts

    xhatminus[k] = np.dot(Ad, xhat[k-1]) + Bd*u[k-1]  # EKF Step 1a: State estimate time update
    Pminus[k] = np.dot(Ad, P[k-1]).dot(Ad.T) + Q[k-1]  # EKF Step 1b: Error covariance time update
    yhat[k] = np.dot(Cd, xhatminus[k]) + Ri*u[k] + d  # EKF Step 1c: Estimate system output

    # SigmaY[k] = np.dot(Cd, Pminus[k]).dot(Cd.T) + R[k-1]  # EKF Step 2a: Compute Kalman gain matrix
    # K[k] = np.dot(Pminus[k], Cd.T)/SigmaY[k]
    # xhat[k] = xhatminus[k]+K[k]*(ytrue[k]-yhat[k])  # EKF Step 2b: State estimate measurement update
    # P[k] = Pminus[k]-np.dot(K[k], SigmaY[k]).dot(K[k].T)  # EKF Step 2c: Error covariance measurement update

    xaugmentedfuse[k, :2, :1] = xhatminus[k]
    xaugmentedfuse[k, 2, 0] = ytrue[k] - Ri*u[k] - d
    Paugmented[k, :2, :2] = Pminus[k]
    Paugmented[k, 2, 2] = R[k-1]
    M[k, 2] = Cd # k in Cd is k sequence in certain cycle.

    _Pminus = np.asmatrix(Pminus[k])
    _xaugmentedfuse = np.asmatrix(xaugmentedfuse[k])
    _Paugmented = np.asmatrix(Paugmented[k])
    _M = np.asmatrix(M[k])

    ''' whitening transform matrix '''
    # W = D^(-1/2) * E, D is eigenvalue of P, E is eigenvector matrix of P
    W_D = np.linalg.eig(_Paugmented)[0]
    W_D = np.array([[element**(-1/2)] for element in W_D])
    W_E = np.array(np.linalg.eig(_Paugmented)[1])

    W = np.asmatrix(W_D*W_E.T)

    xtide[k] = _M*(_M.T*_Paugmented.I*_M).I*_M.T*_Paugmented.I*_xaugmentedfuse
    xtide[k] = xtide[k] + np.array([[0], [0], [Ri*u[k] + d]])
    xhat[k] = xtide[k, :2, :1]
    Ptide[k] = _M*(_M.T*_Paugmented.I*_M).I*_M.T
    P[k] = Ptide[k, :2, :2]

    # Distance calculation
    xaugmentedhat[k, :2, :1] = xhatminus[k]
    xaugmentedhat[k, 2, 0] = ytrue[k]

    xaugmentedhat_w = np.array(W.dot(np.asmatrix(xaugmentedhat[k])))
    xtide_w = np.array(W.dot(np.asmatrix(xtide[k])))

    dist[k] = euclidean_distance(xaugmentedhat_w, xtide_w)
    dist1[k] = euclidean_distance(xaugmentedhat_w[:2, :1], xtide_w[:2, :1])
    dist2[k] = euclidean_distance(xaugmentedhat_w[2, :1], xtide_w[2, :1])

    # dist[k] = (xaugmentedhat[k] - xtide[k]).T.dot(_Paugmented.I).dot(xaugmentedhat[k] - xtide[k])
    # dist1[k] = (xhatminus[k] - xtide[k, :2, :1]).T.dot(_Pminus.I).dot(xhatminus[k] - xtide[k, :2, :1])
    # dist2[k] = (ytrue[k] - xtide[k, 2, :1]).T*R[k]*(ytrue[k] - xtide[k, 2, :1])

    if dist2[k] > 0.01:
        Paugmented2 = _Paugmented.copy()
        Paugmented2[2, 2] = 999999999999
        xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
        xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
        xhat[k] = xtide[k, :2, :1]
        Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
        P[k] = Ptide[k, :2, :2]

        Q[i] = np.array([[0.0000004, 0], [0, 0.000001]])

    xaugmentedhat[k, :2, :1] = xhatminus[k]
    xaugmentedhat[k, 2, 0] = ytrue[k]

for k in range(1, n_iter):
    _w = np.dot(Qsim[k-1], np.random.normal(0, 1))
    w, _ = np.linalg.eig(_w)        # w는 eigen Vector
    w = w.reshape(2, 1)
    v = np.random.normal(0, Rsim[k-1])

    xsimulation[k] = np.dot(AdSim, xsimulation[k - 1]) + np.dot(BdSim, u[k - 1])
    ysimulation[k] = np.dot(CdSim, xsimulation[k]) + Ri * u[k] + dSim
    ysimulation_noisy[k] = ysimulation[k] + v # noise to y
    xsimulation_noisy[k] = xsimulation[k] + w # noise to x

    if k> 5000 and k< 10000: # 7500:  #10000

        CdSim[0, 0] = soc_k_mapping(xsimulation[k, 0])+ ab_size
        BdSim[0, 0] = 1 / ((soc_k_mapping(xsimulation[k, 0])+ab_size) * Ccb)
        dSim = soc_d_mapping(xsimulation[k, 0])+ab_size

        # ab_size = 1
        # ab_size = ab_size+0.0003
        ab_size = random()
    else: # else, changing rate is same for both cases, model and simulation.
        CdSim[0, 0] = soc_k_mapping(xsimulation[k, 0])
        BdSim[0, 0] = 1 / (soc_k_mapping(xsimulation[k, 0]) * Ccb)
        dSim = soc_d_mapping(xsimulation[k, 0])


    Cd[0, 0] = soc_k_mapping(xsimulation[k, 0])
    Bd[0, 0] = 1 / (soc_k_mapping(xsimulation[k, 0]) * Ccb)
    d = soc_d_mapping(xsimulation[k, 0])

    k_sequence_record[k] = CdSim[0, 0]
    d_sequence_record[k] = dSim

    k_sequence_model_record[k] = Cd[0, 0]
    d_sequence_model_record[k] = d

    AdaptiveCovarianceProjectionFilter(Ad, Bd, Cd, Dd, init_Ri, u, d, P, Q, R, xhat, ysimulation_noisy, k)

if isSaveData is True:
    np.save('../data/annormally_data/'+saveXDataName, xsimulation_noisy)
    np.save('../data/annormally_data/'+saveYDataName, ysimulation_noisy)

path = os.path.dirname(__file__)

plt.figure()
x = range(1, n_iter + 1)
plt.plot(x, xsimulation_noisy[:, 0], color='blue', label='Simulation Soc') # 의도적으로 만든 데이터
plt.plot(x, xhat[:, 0], color='red', label='Estimated Soc')
plt.legend()
plt.title('Simulation $\it{S_{OC}}$ vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Soc')
plt.show()

plt.figure()
x = range(1, n_iter + 1)
plt.plot(x, k_sequence_record, label='Simulation kp')
plt.plot(x, d_sequence_record, label='Simulation d')
plt.plot(x, k_sequence_model_record, label='Model kp')
plt.plot(x, d_sequence_model_record, label='Model d')
plt.legend(loc='upper right')
plt.title('kp and d vs. iteration step', fontweight='bold')
plt.show()

plt.figure()
plt.plot(x, dist, label='distance ', color='blue')

np.save(path+'/../temp_save_data_simul3.npy', dist)
#plt.plot(x, dist1, label='distance for state update', color='red')
#plt.plot(x, dist2, label='distance for measurement', color='yellow')
plt.legend()
plt.title('Distance vs. iteration step', fontweight='bold')
plt.xlabel("Time stamp (s)")
plt.ylabel("Distance")
plt.show()

