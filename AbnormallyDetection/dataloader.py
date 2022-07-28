import numpy as np
import os
import pandas as pd
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import BatterySpec

'''
OCV-SOC Curve를 구하기 위해 
파일로 부터 초기 배터리 데이터를 읽고, 읽은 값을 토대로 OCV, SOC Curve, OCV-SOC Curve를 구함.
'''

def ReadData_first(FileNameList, BatteryDataDir):
    '''
    Read Data from first cycle.
    :param FileNameList:
    :param BatteryDataDir:
    :return: k values in one cycle, d values in one cycle, SOC-OCV
             Capacity : initial Capacity
    '''
    for FirstFile in FileNameList:
        print("First Data Loading : {}".format(FirstFile))
        if not FirstFile.endswith("xls") or FirstFile.endswith("xlsx"):
            continue
        else:
            break

    xlsxFile = os.path.join(BatteryDataDir,FirstFile)
    xls = pd.ExcelFile(xlsxFile)
    _data = pd.DataFrame()

    sheet_index = xls.sheet_names[1:]

    # for sheets in sheet_index:
    df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
    _data = pd.concat([_data, df_xls], axis=0)

    cycle_1_data = _data.loc[_data['跳转'] != 0]

    # C_? : Charging, D_? : Discharging
    C_V, C_C, C_SOC, Capacity = SOC_OCV(cycle_1_data, mode='Charge')
    D_V, D_C, D_SOC,  _ = SOC_OCV(cycle_1_data, mode='Discharge')

    if np.size(C_V) > np.size(D_V):
        C_V = C_V[0:np.size(D_V)]
    else:
        D_V = D_V[0:np.size(C_V)]

    true_OCV = (C_V + D_V)/2 # when charging
    T = true_OCV.shape[0] # data 길이(개수)

    SOC_OCV_curve = InterpolatedUnivariateSpline(D_SOC, true_OCV)

    # x_tmp = [i*30 for i in range(len(C_V), 0, -1)]
    #
    # plt.figure()
    # plt.plot(x_tmp, C_V, label='Charge Voltage')
    # plt.plot(x_tmp, D_V, label='Discharge Voltage')
    # plt.plot(x_tmp, true_OCV, label='OCV', color='grey')
    # #plt.gca().invert_xaxis()
    # plt.legend()
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage(v)')
    # plt.show()

    K, D = get_KnD(true_OCV, C_SOC)

    # plt.figure()
    # plt.plot(C_V, label='charge voltage')
    # plt.plot(D_V, label='discharge voltage')
    # plt.xlabel('OCV')
    # plt.ylabel('SOC')
    # plt.legend()
    # plt.title('OCV-SOC Curve')
    #
    # plt.figure()
    # plt.plot(true_OCV, D_SOC)
    # plt.xlabel('OCV')
    # plt.ylabel('SOC')
    # plt.title('OCV-SOC Curve')
    #
    # plt.show()


    return K, D, SOC_OCV_curve, Capacity, T, true_OCV

def SOC_OCV(in_data, mode = None):
    whole_soc = (in_data['容量(mAh)'])
    whole_soc = (whole_soc.to_frame()).values
    max_whole_soc = np.max(whole_soc)

    if mode is 'Charge':
        selected_data = in_data.loc[in_data['状态'] == '恒流恒压充电']                     # data 엑셀 파일에 보면 Step_Index가 2일 떄 충전임. input data바뀌면 이부분 바꾸어 줘야함.
    elif mode is 'Discharge':
        selected_data = in_data.loc[in_data['状态'] == '恒流放电']
    else:
        print(
            'Using SOC_OCV function with right mode : Charge or Discharge'
        )
        exit()

    Voltage = selected_data['电压(V)'].values
    Current = selected_data['电流(mA)'].values
    if mode is 'Discharge':
        Current = -Current

    # Step_time = selected_data['Step_Time(s)'].values
    if mode is 'Discharge':
        Socseries = (max_whole_soc - selected_data['容量(mAh)']) / BatterySpec.spec_full_capacitor   # SOC값은 %로 표현된다. 0~100%
    elif mode is 'Charge':
        Socseries = (selected_data['容量(mAh)']) / BatterySpec.spec_full_capacitor  # SOC값은 %로 표현된다. 0~100%
    Socframe = Socseries.to_frame()
    TrueSoc = Socframe.values

    if mode is 'Discharge':
        TrueSoc = np.flip(TrueSoc, 0).reshape(-1)
        Voltage = np.flip(Voltage, 0).reshape(-1)
    else:
        TrueSoc = TrueSoc.reshape(-1)
        Voltage = Voltage.reshape(-1)

    max_capacity = max(TrueSoc) * BatterySpec.spec_full_capacitor

    return Voltage, Current, TrueSoc, max_capacity


def get_KnD(true_OCV, C_SOC):
    '''
    :param true_OCV: OCV
    :param C_SOC: SOC in charging
    :return: k, d, SOC_OCV_curve(interpolated)
    '''

    if C_SOC.shape[0]<true_OCV.shape[0]:
        charge_index_number = C_SOC.shape[0]
    else:
        charge_index_number = true_OCV.shape[0]

    k_table = np.zeros((int(charge_index_number),))
    d_table = np.zeros((int(charge_index_number),))
    true_OCV = np.flip(true_OCV, 0)
    C_SOC = np.flip(C_SOC, 0)
    # calculate k and d from OCV and SOC curve

    interval = 10

    for i in range(interval, charge_index_number - interval):
        k_table[i] = (true_OCV[i + interval] - true_OCV[i - interval]) / (C_SOC[i + interval] - C_SOC[i - interval])
        d_table[i] = (true_OCV[i - interval] * C_SOC[i + interval] - C_SOC[i - interval] * true_OCV[i + interval]) / (
                C_SOC[i + interval] - C_SOC[i - interval])
    # k_table[0] = k_table[1]
    # k_table[charge_index_number - 1] = k_table[charge_index_number - 2]
    # d_table[0] = d_table[1]
    # d_table[charge_index_number - 1] = d_table[charge_index_number - 2]

    return k_table, d_table

def SOCcurve(first_SOC_OCV, init_capacity, init_T, capacity):
    '''
    노화에 따른 SOC, OCV 생성
    초기 배터리를 방전시키는 시간에 초기 용량대비 줄어든 용량(rated)를 곱하여
    줄어든 방전 시간(Tk) 계산
    초기 SOC에서 sampling , 노화된 OCV 생성(OCV_k), 노화된 SOC 생성(SOC_k)
    :param first_SOC_OCV: SOC_OCV curve at first cycle(interpolated)
    :param init_capacity: capacity at first cycle(Q0) 1.1Ah
    :param init_T: Time at first cycle
    :param capacity: capacity at present cycle(Qk)
    :return: OCV at kth cycle, SOC axis at kth cycle
    '''
    rated = capacity/init_capacity
    Tk = init_T * rated # Time at kth cycle , K cycle에서의 데이터 길이

    # SOC_k는 OCV그래프에서 X축 SOC를 의미 , Tk를 넣어줌으로 써 x축의 개수가 Tk만큼 생긴다.
    SOC_idx = np.arange(0.01, init_capacity / BatterySpec.spec_full_capacitor, init_capacity / (BatterySpec.spec_full_capacitor * Tk))
    OCV_k = first_SOC_OCV(SOC_idx)  # OCV 그래프에서 Y축 의미
    SOC_k = np.arange(0.01, capacity / BatterySpec.spec_full_capacitor, capacity / (BatterySpec.spec_full_capacitor * Tk))
    #SOC_k = np.arange(0.01, init_capacity / BatterySpec.spec_full_capacitor, init_capacity / (BatterySpec.spec_full_capacitor * Tk))

    return OCV_k, SOC_k