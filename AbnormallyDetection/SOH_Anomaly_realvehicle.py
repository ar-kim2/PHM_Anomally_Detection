from AbnormallyDetection.dataloader import ReadData_first
from AbnormallyDetection.dataloader import get_KnD
from AbnormallyDetection.dataloader import SOCcurve
from utils.BatteryDevice_dataloader import ReadAbnormalData_realVehicle
import os
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import BatterySpec

'''
실제 차량에 들어가는 배터리에 대한 이상징후 검출.
차량에 들어가는 배터리는 INR-18650배터리의 직렬 96cell, 병렬 32cell로 이루어져 있기 때문에
각 파라메터를 해당 스펙에 맞추어 변경
'''

SOH_k = 0.9 #0.95   #   97   # estimated SOH at k, it would be estimate based on deep learning model..

train_battery_list = ['base_ocv']
dir_path = "../data/battery_device"


tmp_iter = 0

# 추정한 SOH와 초기 OCV, SOC를 통해 k, d를 구함.
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

    # Temp Data (Error로 인한 k,d의 유의 범위, 차이를 표현하기 위함)
    OCV_k_min, SOC_k_min = SOCcurve(SOC_OCV_1, Capacity_1, T_1, (SOH_k-0.026) * BatterySpec.spec_full_capacitor)        # charging direction
    OCV_k_max, SOC_k_max = SOCcurve(SOC_OCV_1, Capacity_1, T_1, (SOH_k+0.026) * BatterySpec.spec_full_capacitor)        # charging direction

    K_k_min, D_k_min = get_KnD(OCV_k_min, SOC_k_min) # discharging direction
    K_k_max, D_k_max = get_KnD(OCV_k_max, SOC_k_max) # discharging direction

    print("check OCV : ", len(OCV_k), " SOC : ", len(SOC_k))

    # plt.figure()
    # plt.plot(SOC_k, OCV_k[:])
    # plt.xlabel('SOC')
    # plt.ylabel('OCV')
    # plt.title('k Cycle SOC-OCV Curve')
    # plt.show()

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

        # Temp Data (Error로 인한 k,d의 유의 범위, 차이를 표현하기 위함)
        K_k_min = np.delete(K_k_min, 0)
        D_k_min = np.delete(D_k_min, 0)
        K_k_min = np.delete(K_k_min, -1)
        D_k_min = np.delete(D_k_min, -1)

        K_k_max = np.delete(K_k_max, 0)
        D_k_max = np.delete(D_k_max, 0)
        K_k_max = np.delete(K_k_max, -1)
        D_k_max = np.delete(D_k_max, -1)

        SOC_k_min = SOC_k_min[1:-1]
        SOC_k_max = SOC_k_max[1:-1]

    tmp_OCV = np.flip(OCV_base, 0).reshape(-1)
    x_tmp1 = [t*30 for t in range(len(tmp_OCV))]

    tmp_OCVk = np.flip(OCV_k, 0).reshape(-1)
    x_tmp2 = [t*30 for t in range(len(tmp_OCVk))]

    plt.figure()
    #plt.plot(x_tmp1, np.flip(OCV_base, 0).reshape(-1), label='OCV Curve')
    plt.plot(x_tmp1, np.flip(OCV_base, 0).reshape(-1), label='Base OCV')
    plt.plot(x_tmp2, np.flip(OCV_k, 0).reshape(-1) , label='k Cycle OCV')
    plt.xlabel('Time step (s)')
    plt.ylabel('OCV (v)')
    plt.title('OCV Curve')
    #plt.legend()
    plt.show()

    tmp_OCV = np.flip(SOC_base, 0).reshape(-1)
    x_tmp1 = [t*30 for t in range(len(tmp_OCV))]

    tmp_OCVk = np.flip(SOC_k, 0).reshape(-1)
    x_tmp2 = [t*30 for t in range(len(tmp_OCVk))]

    plt.figure()
    plt.plot(x_tmp1, np.flip(SOC_base, 0).reshape(-1), label='Base SOC')
    plt.plot(x_tmp2, np.flip(SOC_k, 0).reshape(-1), label='k Cycle SOC')
    plt.xlabel('Time step (s)')
    plt.ylabel('SOC')
    plt.title('SOC Curve')
    plt.legend()
    plt.show()

    # Error Bar Calculate
    check_SOC_list = [0.2, 0.4, 0.6, 0.8]
    error_x = []
    error_ky = []
    error_dy = []

    error_k_upper = []
    error_k_lower = []

    error_d_upper = []
    error_d_lower = []

    for checkSoc in check_SOC_list:
        print("CHECK SOc ", checkSoc)
        check_idx = np.where(SOC_k > checkSoc)

        print("CHECK : ", K_k[check_idx[0][0]])

        ref_k = K_k[check_idx[0][0]]
        ref_d = D_k[check_idx[0][0]]

        check_min_idx = np.where(SOC_k_min > checkSoc)
        check_max_idx = np.where(SOC_k_max > checkSoc)

        # print("HECK 11 ", check_min_idx[0])
        # print("HECK 22 ", K_k_min[check_min_idx[0]])

        error_k_lower.append(np.abs(ref_k - K_k_min[check_min_idx[0][0]]))
        error_k_upper.append(np.abs(K_k_max[check_max_idx[0][0]]-ref_k))

        error_d_lower.append(np.abs(ref_d - D_k_min[check_min_idx[0][0]]))
        error_d_upper.append(np.abs(D_k_max[check_max_idx[0][0]]-ref_d))

        error_x.append(SOC_k[check_idx[0][0]])
        error_ky.append(ref_k)
        error_dy.append(ref_d)

    yerr_k = [(error_k_lower[0], error_k_lower[1], error_k_lower[2], error_k_lower[3]), (error_k_upper[0], error_k_upper[1], error_k_upper[2], error_k_upper[3])]
    yerr_d = [(error_d_lower[0], error_d_lower[1], error_d_lower[2], error_d_lower[3]), (error_d_upper[0], error_d_upper[1], error_d_upper[2], error_d_upper[3])]

    plt.figure()
    plt.plot(SOC_base, K_base, label='Base k')
    plt.plot(SOC_base, D_base, label='Base d')
    plt.plot(SOC_k, K_k, label='p Cycle k')
    plt.plot(SOC_k, D_k, label='p Cycle d')

    # plt.plot(SOC_k_min, K_k_min, label='p Cycle k', c='r')
    # plt.plot(SOC_k_max, K_k_max, label='p Cycle k', c='r')
    # plt.plot(SOC_k_min, D_k_min, label='p Cycle d', c='b')
    # plt.plot(SOC_k_max, D_k_max, label='p Cycle d', c='b')

    # plt.errorbar(error_x, error_ky, linestyle='none', yerr=yerr_k, c='dimgray', lolims=True, uplims=True)
    plt.errorbar(error_x, error_ky, yerr=yerr_k, linestyle='none', fmt="ob", capsize=4, capthick=1.5, markersize=2, elinewidth=1.5, ecolor='k', markerfacecolor='g', markeredgecolor='g')
    plt.errorbar(error_x, error_dy, yerr=yerr_d, linestyle='none', fmt="ob", capsize=4, capthick=1.5, markersize=2, elinewidth=1.5, ecolor='k', markerfacecolor='r', markeredgecolor='r')
    # plt.errorbar(error_x, error_dy, linestyle='none', yerr=yerr_d, c='dimgray', lolims=True, uplims=True)


    plt.xlabel('SOC')
    plt.title('K, D')
    plt.legend()
    plt.show()

    print('k max : ',np.max(K_base))
    print('k min : ', np.min(K_base))
    print('d max : ', np.max(D_base))
    print('d min : ', np.min(D_base))

    soc_k_mapping = InterpolatedUnivariateSpline(SOC_k, np.flip(K_k, 0), k=2)
    soc_d_mapping = InterpolatedUnivariateSpline(SOC_k, np.flip(D_k, 0), k=2)

    break


########################### CPF ###########################
#TODO:: Test 배터리 종류가 바뀌면 아래 배터리 특성값을 수정해 주어야함.
rated_capacity = 3.25 * 32 #  3.35 # BatterySpec.spec_full_capacitor
init_soc = SOH_k-0.02 # / rated_capacity
init_data_size = len(K_1)

Ccs = (32/96)*90  # 90
Ccb =  (32/96)*12240  #12240  # Ccb = 3370.7558     1.1 * 3600  ,  3.4 * 3600
Rt = (96/32)*0.06

# internal Resistance with calculated capacity

init_Ri = (96/32)*((-0.0357 * SOH_k) + 0.17)   #0.0357   0.17
#init_Ri = 0.7684 - SOH_k * 0.5349

Ts = 1

isAbnormal = False
Abnormal_Cnt = 0
AbnormalStart_idx = 0

def euclidean_distance(x, y):
    return np.sum((x - y) ** 2)

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
    global isAbnormal
    global Abnormal_Cnt
    global AbnormalStart_idx
    global tmp_iter


    xhatminus[k] = np.dot(Ad, xhat[k-1]) + Bd*u[k-1]  # EKF Step 1a: State estimate time update
    Pminus[k] = np.dot(Ad, P[k-1]).dot(Ad.T) + Q[k-1]  # EKF Step 1b: Error covariance time update
    yhat[k] = np.dot(Cd, xhatminus[k]) + Ri*u[k] + d  # EKF Step 1c: Estimate system output

    # SigmaY[k] = np.dot(Cd, Pminus[k]).dot(Cd.T) + R[k-1]  # EKF Step 2a: Compute Kalman gain matrix
    # K[k] = np.dot(Pminus[k], Cd.T)/SigmaY[k]
    # xhat[k] = xhatminus[k]+K[k]*(ytrue[k]-yhat[k])  # EKF Step 2b: State estimate measurement update
    # P[k] = Pminus[k]-np.dot(K[k], SigmaY[k]).dot(K[k].T)  # EKF Step 2c: Error covariance measurement update

    xaugmentedfuse[k, :2, :1] = xhatminus[k]
    xaugmentedfuse[k, 2, 0] = ytrue - Ri*u[k] - d
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

    ''' Distance calculation '''
    xaugmentedhat[k, :2, :1] = xhatminus[k]
    xaugmentedhat[k, 2, 0] = ytrue

    # xaugmentedhat_w = np.array(W.dot(np.asmatrix(xaugmentedhat[k])))
    # xtide_w = np.array(W.dot(np.asmatrix(xtide[k])))

    xaugmentedhat_w = np.array(W.dot(np.asmatrix(xaugmentedhat[k])))
    xtide_w = np.array(W.dot(np.asmatrix(xtide[k])))

    # dist[k] = euclidean_distance(xaugmentedhat_w, xtide_w)
    # dist1[k] = euclidean_distance(xaugmentedhat_w[:2, :1], xtide_w[:2, :1])
    # dist2[k] = euclidean_distance(xaugmentedhat_w[2, :1], xtide_w[2, :1])

    dist[k] = np.asmatrix(xaugmentedhat_w-xtide_w).T.dot(np.asmatrix(xaugmentedhat_w-xtide_w))
    dist1[k] = np.asmatrix(xaugmentedhat_w[:2, :1]-xtide_w[:2, :1]).T.dot(np.asmatrix(xaugmentedhat_w[:2, :1]-xtide_w[:2, :1]))
    dist2[k] = np.asmatrix(xaugmentedhat_w[2, :1]-xtide_w[2, :1]).T.dot(np.asmatrix(xaugmentedhat_w[2, :1]-xtide_w[2, :1]))

    #
    dist[k] = (xaugmentedhat[k] - xtide[k]).T.dot(_Paugmented.I).dot(xaugmentedhat[k] - xtide[k])
    dist1[k] = (xhatminus[k] - xtide[k, :2, :1]).T.dot(_Pminus.I).dot(xhatminus[k] - xtide[k, :2, :1])
    dist2[k] = (ytrue - xtide[k, 2, :1]).T*R[k]*(ytrue - xtide[k, 2, :1])

    #dist2[k] = (ytrue - Ri*u[k] - d - xtide[k, 2, 0]).T * R[k] * (ytrue - Ri*u[k] - d - xtide[k, 2, 0])

    if dist[k] > 7 : #3.8:   # 7
        Paugmented2 = _Paugmented.copy()
        Paugmented2[2, 2] = 99999999999  # Paugmented2[2, 2] *100 #99999999999

        #Q[k] = np.array([[0.0008, 0], [0, 0.002]])

        # if dist2[k] > 3.8:
        #      Paugmented2[2, 2] = 99999999999 # Paugmented2[2, 2] *100 #99999999999
        # elif dist1[k] > 5.9:
        #      Paugmented2[0, 0] = Paugmented2[0, 0] + 0.01 #99999999999
        #      Paugmented2[1, 1] = Paugmented2[0, 0] + 0.01 #99999999999

        xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
        xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
        xhat[k] = xtide[k, :2, :1]
        Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
        P[k] = Ptide[k, :2, :2]

        Q[k] = np.array([[0.0000008, 0], [0, 0.000002]])
        #Q[k] = np.array([[0.08, 0], [0, 0.2]])
        #Q[k] = np.array([[0.0008, 0], [0, 0.002]])
        # R[k] = 0.00001
        Q[k] = np.array([[800, 0], [0, 20]])

    # elif dist[k] > 2.7:  # 3.8:   # 7
    #     Paugmented2 = _Paugmented.copy()
    #     Paugmented2[2, 2] = 99999999999  # Paugmented2[2, 2] *100 #99999999999
    #
    #     # if dist2[k] > 3.8:
    #     #      Paugmented2[2, 2] = 99999999999 # Paugmented2[2, 2] *100 #99999999999
    #     # elif dist1[k] > 5.9:
    #     #      Paugmented2[0, 0] = Paugmented2[0, 0] + 0.01 #99999999999
    #     #      Paugmented2[1, 1] = Paugmented2[0, 0] + 0.01 #99999999999
    #
    #     xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
    #     xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
    #     xhat[k] = xtide[k, :2, :1]
    #     Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
    #     P[k] = Ptide[k, :2, :2]
    #
    #     Q[k] = np.array([[0.04, 0], [0, 0.1]])
    #     Q[k] = np.array([[800, 0], [0, 20]])
    #     R[k] = 0.00001

    elif dist1[k] > 5:  # 3.8:   # 7
        Paugmented2 = _Paugmented.copy()
        # Paugmented2[0, 0] = 99999999999 #Paugmented2[0, 0] + 0.01
        # Paugmented2[1, 1] =  99999999999  #Paugmented2[0, 0] + 0.01 #
        Paugmented2[2, 2] = 99999999999

        # if dist2[k] > 3.8:
        #      Paugmented2[2, 2] = 99999999999 # Paugmented2[2, 2] *100 #99999999999
        # elif dist1[k] > 5.9:
        #      Paugmented2[0, 0] = Paugmented2[0, 0] + 0.01 #99999999999
        #      Paugmented2[1, 1] = Paugmented2[0, 0] + 0.01 #99999999999

        xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
        xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
        xhat[k] = xtide[k, :2, :1]
        Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
        P[k] = Ptide[k, :2, :2]

        Q[k] = np.array([[0.0000004, 0], [0, 0.000001]])
        R[k] = 0.00001








    # if  dist[k] > 0.08 :   #0.01
    #     Paugmented2 = _Paugmented.copy()
    #     Paugmented2[0, 0] = 999999999999
    #     Paugmented2[2, 2] = 999999999999
    #     xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
    #     xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
    #     xhat[k] = xtide[k, :2, :1]
    #     Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
    #     P[k] = Ptide[k, :2, :2]
    # if dist2[k] > 0.5 : # 1.6 : #    0.004:
    #     Paugmented2 = _Paugmented.copy()
    #     Paugmented2[2, 2] = 999999999999
    #     xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
    #     xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
    #     xhat[k] = xtide[k, :2, :1]
    #     Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
    #     P[k] = Ptide[k, :2, :2]

    if dist2[k] > 1.6: #   0.004:    #0.005
        Abnormal_Cnt = Abnormal_Cnt + 1
    else:
        Abnormal_Cnt = 0

    if isAbnormal is False and Abnormal_Cnt > 10:
        isAbnormal = True
        AbnormalStart_idx = k- 40

    tmp_iter = tmp_iter+1

    # elif dist1[k] > 0.01:
    #     Paugmented2 = _Paugmented.copy()
    #     Paugmented2[0, 0] = 999999999999
    #     xtide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T * Paugmented2.I * _xaugmentedfuse
    #     xtide[k] = xtide[k] + np.array([[0], [0], [Ri * u[k] + d]])
    #     xhat[k] = xtide[k, :2, :1]
    #     Ptide[k] = _M * (_M.T * Paugmented2.I * _M).I * _M.T
    #     P[k] = Ptide[k, :2, :2]


#TODO:: 측정 데이터가 바뀌면 아래 주소값을 바꾸어주어야한다.
#dir_path = "../data/annormally_data/06t13A_10m20m"
dir_path = "../data/annormally_data/065A_data_2"
#dir_path = "../data/annormally_data/0226_06At13A_10m20m_10ohm"
FileNameList = os.listdir(dir_path)

Abnormal_Dection_Result = np.zeros(np.size(FileNameList))

for data in range(np.size(FileNameList)):
    ymeasure, ymeasure_cur, measure_ts = ReadAbnormalData_realVehicle(dir_path +'/' +FileNameList[data])
    print("check vol : ", ymeasure, " cur : ", ymeasure_cur)

    if np.size(ymeasure) == 0:
        continue


    #TODO:: 측정값 Time step과 State Equation Time step을 맞추어주어야함.
    # 이 경우 State Equation Equation Time step(Ts)를 1로 하고 측정은 30초 마다 한뒤, 측정값길이가 70정도 여서 70*30 을 n_iter로 설정함.
    n_iter =  measure_ts * (np.size(ymeasure))   #1600 #  1600 #10000 #900 #6500

    isAbnormal = False
    Abnormal_Cnt = 0
    AbnormalStart_idx = 0

    kp = soc_k_mapping(init_soc)
    d = soc_d_mapping(init_soc)

    Ad = np.array([[1, 0],
                   [0, 1 - (Ts / (Rt * Ccs))]])
    Bd = np.array([[Ts / (kp * Ccb)],
                   [Ts / Ccs]])
    Cd = np.array([[kp, 1]])
    Dd = init_Ri
    Ri = init_Ri

    Cd_mes = np.array([[kp, 1]])

    #####################################################
    Q = np.zeros((n_iter, 2, 2))
    for i in range(0, n_iter):
        Q[i] = np.array([[0.000004, 0], [0, 0.00001]])


    R = np.zeros((n_iter,))
    for i in range(0, n_iter):
        R[i] = 0.001



    # R[0] = 1
    # for i in range(1, n_iter):
    #     R[i] = R[i-1] - 0.001
    #
    #     if i > 160:
    #         R[i] = 0.001 # 0.0001

    # ymeasure_noisy = np.zeros((n_iter,))
    # ymeasure_noisy[0] = 4.1

    ####################################################

    u = np.zeros((n_iter,))
    for i in range(0, n_iter):
        u[i] = ymeasure_cur[0][0]/1000

    xhat = np.zeros((n_iter, 2, 1))
    xhat[0] = np.array([[init_soc],
                        [0]])

    P = np.zeros((n_iter, 2, 2))          # a posteri error estimate
    #P[0] = np.array([[1, 0], [0, 1]])
    P[0] = np.array([[100, 0], [0, 100]])


    xhatminus = np.zeros((n_iter, 2, 1))  # a priori estimate of x
    Pminus = np.zeros((n_iter, 2, 2))     # a priori error estimate
    K = np.zeros((n_iter, 2, 1))          # gain or blending factor
    yhat = np.zeros((n_iter,))
    yhat[0] = 3.8 * 96 # 4.1
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


    for k in range(1, n_iter):
        Cd[0, 0] = soc_k_mapping(xhat[k-1, 0])
        Bd[0, 0] = 1 / (soc_k_mapping(xhat[k-1, 0]) * Ccb)
        d = soc_d_mapping(xhat[k-1, 0])

        y_idx = int(k/measure_ts)

        u[k] = ymeasure_cur[0][y_idx] / 1000

        AdaptiveCovarianceProjectionFilter(Ad, Bd, Cd, Dd, init_Ri, u, d, P, Q, R, xhat, ymeasure[0][y_idx], k)


        #if k >9 and k < 30:


    x = range(1, n_iter - 2)

    if isAbnormal is True:
        if AbnormalStart_idx < 0:
            print("Abnormal Detection is fail, Abnormal index is ", AbnormalStart_idx)
        else:
            print("Abnormal occure at about ", AbnormalStart_idx, "s")
            Abnormal_Dection_Result[data] = 1
    else:
        if FileNameList[data].__contains__('normal'):
            print("This data is normal.")
            Abnormal_Dection_Result[data] = 1

    plt.figure()
    # plt.plot(x, dist1[3:], label='distance for state update', color='red')
    # plt.plot(x, dist2[3:], label='distance for measurement', color='yellow')
    plt.plot(x, dist[3:], label='distance ', color='blue')
    plt.legend()
    plt.title('Distance vs. iteration step', fontweight='bold')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Distance')
    #plt.ylim(0, 0.01)
    plt.show()
    #
    time_index = range(1, n_iter, measure_ts)
    plt.figure()
    plt.plot(time_index, ymeasure[0][:(np.size(ymeasure[0]))])
    plt.xlabel("Time step (s)")
    plt.ylabel('Voltage (v)')
    plt.title(FileNameList[data])
    plt.show()

    # time_index = range(1, n_iter, measure_ts)
    # plt.figure()
    # plt.plot(time_index, ymeasure_cur[0][:(np.size(ymeasure_cur[0]))])
    # # plt.ylim(-690, -670)
    # plt.xlabel("Time step (s)")
    # plt.ylabel('Current (v)')
    # plt.title(FileNameList[data])
    # plt.show()

    # plt.figure()
    # x = range(1, n_iter + 1)
    # # plt.plot(x, xsimulation_noisy[:, 0], color='blue', label='Simulation Soc')  # 의도적으로 만든 데이터
    # plt.plot(x, xhat[:, 0], color='red', label='Estimated Soc')
    # plt.legend()
    # plt.title('Simulation $\it{S_{OC}}$ vs. iteration step', fontweight='bold')
    # plt.xlabel('Iteration')
    # plt.ylabel('Soc')
    # plt.show()


Accuracy = np.count_nonzero(Abnormal_Dection_Result == 1)/np.size(Abnormal_Dection_Result)*100
print("Accuracy is ", Accuracy, " %")
# current = np.load('../data/annormally_data/0219_068A_10t1ohm/discharge_current.npy', allow_pickle=True)
# time_index = range(1, n_iter, 30)
# plt.figure()
# plt.plot(time_index, np.abs(current[0][:(np.size(current[0]) - 10)]))
# plt.xlabel("Time step (s)")
# plt.ylabel('Current (mA)')
# plt.ylim(500, 800)
# plt.show()
