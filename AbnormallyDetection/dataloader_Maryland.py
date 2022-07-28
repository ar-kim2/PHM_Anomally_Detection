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

def Read_AnomalData(FileNameList, BatteryDataDir):
    '''
    Read Data from first cycle.

    :param FileNameList:
    :param BatteryDataDir:
    :return: k values in one cycle, d values in one cycle, SOC-OCV
             Capacity : initial Capacity
    '''

    result_vol = []
    result_cur = []


    for FirstFile in FileNameList:
        print("Abnornal Data Loading : {}".format(FirstFile))
        if not FirstFile.endswith("xls") or FirstFile.endswith("xlsx"):
            continue
        else:
            break

    xlsxFile = os.path.join(BatteryDataDir,FirstFile)
    xls = pd.ExcelFile(xlsxFile)
    _data = pd.DataFrame()
    if 'Statistics' in xls.sheet_names[-1]:
        sheet_index = xls.sheet_names[1:-1]
    else:
        sheet_index = xls.sheet_names[1:]

    for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheets)
        _data = pd.concat([_data, df_xls], axis=0)

    cycle_number = max(_data['Cycle_Index'])
    cycle_1_data = _data.loc[_data['Cycle_Index'] == 1]

    # C_? : Charging, D_? : Discharging
    D_V, D_C, D_SOC, D_SOC_V, _ = SOC_OCV(cycle_1_data, mode='Discharge')

    D_V = np.flip(D_V, 0)

    result_vol.append(D_V)
    result_cur.append(D_C)


    return result_vol, result_cur, 20

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
    if 'Statistics' in xls.sheet_names[-1]:
        sheet_index = xls.sheet_names[1:-1]
    else:
        sheet_index = xls.sheet_names[1:]

    for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheets)
        _data = pd.concat([_data, df_xls], axis=0)

    cycle_number = max(_data['Cycle_Index'])
    cycle_1_data = _data.loc[_data['Cycle_Index'] == 1]

    # C_? : Charging, D_? : Discharging
    C_V, C_C, C_SOC, C_SOC_V, Capacity = SOC_OCV(cycle_1_data, mode='Charge')
    D_V, D_C, D_SOC, D_SOC_V, _ = SOC_OCV(cycle_1_data, mode='Discharge')

    # 충전과 방전 시의 전압 변화를 1/2 하여 OCV curve를 얻는다.
    D_OCV = D_SOC_V(C_SOC)
    C_OCV = C_SOC_V(C_SOC)

    # plt.figure()
    # plt.plot(D_OCV, label="discharge")
    # plt.plot(C_OCV, label="charge")
    # plt.legend
    # plt.show()



    true_OCV = (D_OCV + C_OCV)/2 # when charging
    T = true_OCV.shape[0]

    SOC_OCV_curve = InterpolatedUnivariateSpline(C_SOC, true_OCV)
    # K와 D를 계산한다.
    K, D = get_KnD(true_OCV, C_SOC)
    #
    # plt.plot(K, label='k')
    # plt.plot(D, label='d')
    # plt.legend()
    # plt.show()

    return K, D, SOC_OCV_curve, Capacity, T, true_OCV


def SOC_OCV(in_data, mode = None):
    whole_soc = (in_data['Charge_Capacity(Ah)'] - in_data['Discharge_Capacity(Ah)']) / 1.1
    whole_soc = (whole_soc.to_frame()).values
    min_whole_soc = np.min(whole_soc)

    if mode is 'Charge':
        selected_data = in_data.loc[in_data['Step_Index'] == 2]
    elif mode is 'Discharge':
        selected_data = in_data.loc[in_data['Step_Index'] == 7]
    else:
        print(
            'Using SOC_OCV function with right mode : Charge or Discharge'
        )
        exit()

    Voltage = selected_data['Voltage(V)'].values
    Current = selected_data['Current(A)'].values
    # Step_time = selected_data['Step_Time(s)'].values
    Socseries = (selected_data['Charge_Capacity(Ah)'] - selected_data['Discharge_Capacity(Ah)']) / 1.1
    Socframe = Socseries.to_frame()
    TrueSoc = Socframe.values
    # minimum_Soc_charge = np.min(TrueSoc)
    if min_whole_soc < 0:
        TrueSoc = (TrueSoc - min_whole_soc)
    if mode is 'Discharge':
        TrueSoc = np.flip(TrueSoc, 0).reshape(-1)
        Voltage = np.flip(Voltage, 0).reshape(-1)
    else:
        TrueSoc = TrueSoc.reshape(-1)
        Voltage = Voltage.reshape(-1)
    max_capacity = max(TrueSoc) * 1.1
    SOC_V_curve = InterpolatedUnivariateSpline(TrueSoc, Voltage) # ..(x, y)
    return Voltage, Current, TrueSoc, SOC_V_curve, max_capacity


def get_KnD(true_OCV, C_SOC):
    '''

    :param true_OCV: OCV
    :param C_SOC: SOC in charging
    :return: k, d, SOC_OCV_curve(interpolated)
    '''

    charge_index_number = true_OCV.shape[0]
    k_table = np.zeros((charge_index_number,))
    d_table = np.zeros((charge_index_number,))
    true_OCV = np.flip(true_OCV, 0)
    C_SOC = np.flip(C_SOC, 0)
    for i in range(1, charge_index_number - 1):
        k_table[i] = (true_OCV[i + 1] - true_OCV[i - 1]) / (C_SOC[i + 1] - C_SOC[i - 1])
        d_table[i] = (true_OCV[i - 1] * C_SOC[i + 1] - C_SOC[i - 1] * true_OCV[i + 1]) / (
                C_SOC[i + 1] - C_SOC[i - 1])

    k_table = np.delete(k_table, 0)
    d_table = np.delete(d_table, 0)
    k_table = np.delete(k_table, -1)
    d_table = np.delete(d_table, -1)
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
    Tk = init_T * rated # Time at kth cycle
    SOC_k = np.arange(0, init_capacity/1.1, init_capacity/(1.1*Tk))
    OCV_k = first_SOC_OCV(SOC_k)
    SOC_k = np.arange(0, init_capacity/1.1, init_capacity/(1.1*Tk))
    #SOC_k = np.arange(0, capacity/1.1, capacity/(1.1*Tk))

    return OCV_k, SOC_k