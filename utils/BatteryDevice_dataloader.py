import numpy as np
import os
import pandas as pd
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import BatterySpec

def ReadData(FileNameList, BatteryDataDir):
    D_Voltage = []
    C_Voltage = []
    D_Current = []
    C_Current = []
    Capacity = []
    SOH = []
    last_cycle = 0

    for FileName in FileNameList:
        if not FileName.endswith("xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[1:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['循环'][len(_data['循环']) - 1]

        for cycle in range(1, last_index+1):
            try:
                list(_data['循环'].values).index(cycle)

                cycle_data = _data.loc[_data['循环'] == cycle]

                discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
                charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

                D_Voltage.append(discharge_data['电压(V)'].values)
                D_Current.append(discharge_data['电流(mA)'].values)

                C_Voltage.append(charge_data['电压(V)'].values)
                C_Current.append(charge_data['电流(mA)'].values)

                whole_cap = cycle_data.loc[_data['状态'] == 'SOH']['容量(mAh)'].values[0]

                Capacity.append(whole_cap)
                SOH.append(whole_cap / BatterySpec.spec_full_capacitor)

                last_cycle = last_cycle + 1
            except:
                continue

    return D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle

def ReadAbnormalData(FileName):
    D_Voltage = []
    D_Current = []

    if not FileName.endswith("xls") or FileName.endswith("xlsx"):
        return D_Voltage

    print("Data Loading : {}".format(FileName))

    xlsxFile = FileName
    xls = pd.ExcelFile(FileName)
    _data = pd.DataFrame()

    sheet_index = xls.sheet_names[1:]

    # for sheets in sheet_index:
    df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
    _data = pd.concat([_data, df_xls], axis=0)

    last_index = _data['循环'][len(_data['循环']) - 1]

    for cycle in range(1, last_index+1):
        try:
            list(_data['循环'].values).index(cycle)

            cycle_data = _data.loc[_data['循环'] == cycle]

            discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
            charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

            D_Voltage.append(discharge_data['电压(V)'].values)
            D_Current.append(discharge_data['电流(mA)'].values)

            C_Voltage.append(charge_data['电压(V)'].values)
            C_Current.append(charge_data['电流(mA)'].values)

            whole_cap = cycle_data.loc[_data['状态'] == 'SOH']['容量(mAh)'].values[0]

            Capacity.append(whole_cap)
            SOH.append(whole_cap / BatterySpec.spec_full_capacitor)

            last_cycle = last_cycle + 1
        except:
            continue

    elapsed_time = _data['相对时间(h:min:s.ms)'].values

    first_time = elapsed_time[0].split(':')   #first_time[2]
    first_time = first_time[2].split('.')
    first_time = int(first_time[0])
    second_time = elapsed_time[1].split(':')
    second_time = second_time[2].split('.')
    second_time = int(second_time[0])

    if first_time < second_time:
        time_step = second_time-first_time
    else:
        time_step = first_time- second_time

    return D_Voltage, D_Current, time_step


def ReadAbnormalData_realVehicle(FileName):
    D_Voltage = []
    D_Current = []

    if not FileName.endswith("xls") or FileName.endswith("xlsx"):
        return D_Voltage

    print("Data Loading : {}".format(FileName))

    xlsxFile = FileName
    xls = pd.ExcelFile(FileName)
    _data = pd.DataFrame()

    sheet_index = xls.sheet_names[1:]

    # for sheets in sheet_index:
    df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[2])
    _data = pd.concat([_data, df_xls], axis=0)

    last_index = _data['循环'][len(_data['循环']) - 1]

    for cycle in range(1, last_index+1):
        try:
            list(_data['循环'].values).index(cycle)

            cycle_data = _data.loc[_data['循环'] == cycle]

            discharge_data = cycle_data.loc[_data['状态'] == '恒流放电']
            charge_data = cycle_data.loc[_data['状态'] == '恒流恒压充电']

            D_Voltage.append(discharge_data['电压(V)'].values*96)
            D_Current.append(discharge_data['电流(mA)'].values*32)

            C_Voltage.append(charge_data['电压(V)'].values)
            C_Current.append(charge_data['电流(mA)'].values)

            whole_cap = cycle_data.loc[_data['状态'] == 'SOH']['容量(mAh)'].values[0]

            Capacity.append(whole_cap)
            SOH.append(whole_cap / BatterySpec.spec_full_capacitor)

            last_cycle = last_cycle + 1
        except:
            continue

    elapsed_time = _data['相对时间(h:min:s.ms)'].values

    first_time = elapsed_time[0].split(':')   #first_time[2]
    first_time = first_time[2].split('.')
    first_time = int(first_time[0])
    second_time = elapsed_time[1].split(':')
    second_time = second_time[2].split('.')
    second_time = int(second_time[0])

    if first_time < second_time:
        time_step = second_time-first_time
    else:
        time_step = first_time- second_time

    return D_Voltage, D_Current, time_step

def Density(list):
    all_prob = []
    len_list = []

    min_vol = np.min(list)
    max_vol = np.max(list)

    for idx in range(len(list)):
        hist, bin_edge = np.histogram(list[idx], bins=np.linspace(min_vol, max_vol, 17))     # np.linspace(2,5,17) 2부터 5까지 17개 칸으로 나눔.
        list_size = np.size(list[idx])
        len_list.append(list_size)          # list_size는 총 데이터의 개수
        # 나중에 엔트로피 구할 때, logp(x)를 구해야하는데 p(x)가 0이 되어버리면 logp(x)값이 무한이 되어버려서 에러가 난다. 그래서 0에 가까운 수로 만들어 주기 위해 추가.
        hist = hist + np.ones(16)
        prob = hist/(list_size+16)
        all_prob.append(prob)

    bin_center = 0.5 * (bin_edge[1:] + bin_edge[:-1])

    # plt.figure()
    # plt.plot(bin_center, hist)
    # plt.show()

    return all_prob, len_list

if __name__ == "__main__":
    train_battery_list = ['0218_068A_02ohm']
    dir_path = "../data/annormally_data"

    voltage_list = []

    for battery in train_battery_list:
        BatteryDataDir = os.path.join(dir_path, battery)
        FileNameList = os.listdir(BatteryDataDir)

        D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle = ReadData(FileNameList, BatteryDataDir)

        np.save(BatteryDataDir + '/capacity', Capacity)
        np.save(BatteryDataDir + '/SOH', SOH)
        np.save(BatteryDataDir + '/discharge_data', D_Voltage)
        np.save(BatteryDataDir + '/discharge_current', D_Current)
        np.save(BatteryDataDir + '/charge_data', C_Voltage)
        np.save(BatteryDataDir + '/charge_current', C_Current)
        np.save(BatteryDataDir + '/last_cycle', last_cycle)

        # np.save('../data/annormally_data/make_data_voltage', D_Voltage)
        # np.save('../data/annormally_data/make_data_Soc', D_Soc)

    # Density(voltage_list)

