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
    D_Time = []
    C_Time = []

    All_D_Time = []

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

        sheet_index = xls.sheet_names[:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[0])
        _data = pd.concat([_data, df_xls], axis=0)

        last_index = _data['cycle'][len(_data['cycle']) - 1]

        for cycle in range(1, last_index+1):
            try:
                list(_data['cycle'].values).index(cycle)

                cycle_data = _data.loc[_data['cycle'] == cycle]

                time_idx = list(cycle_data['time'].values[1:]).index(0) + 1


                # discharge_data = cycle_data.loc[_data['current'] <= 0]
                # charge_data = cycle_data.loc[_data['current'] > 0]

                # discharge data가 처음에 올라가는 부분이 있고, 다 떨어지고 나서도 올라가는 부분이 있다.
                # 이부분 제거해줌.
                discharge_Vol = np.round(cycle_data['voltage'].values[time_idx:], 1)
                max_discharge_Vol = max(discharge_Vol)
                #min_discharge_Vol = min(discharge_Vol[30:])
                min_discharge_Vol = 2.8

                # tmp_discharge_Vol = list(discharge_Vol[:, -1])
                # tmp_discharge_start = tmp_discharge_Vol.index(max_discharge_Vol)
                #
                # discharge_start_idx = len(discharge_Vol) - tmp_discharge_start    #
                discharge_start_idx = time_idx + 2 #time_idx + list(discharge_Vol).index(max_discharge_Vol)
                discharge_end_idx = time_idx + list(discharge_Vol).index(min_discharge_Vol)

                #D_Voltage.append(discharge_data['voltage'].values[discharge_start_idx:discharge_end_idx])      # unit V
                #D_Current.append(discharge_data['current'].values[discharge_start_idx:discharge_end_idx])      # unit A

                discharge_voltage = cycle_data['voltage'].values[discharge_start_idx:discharge_end_idx]
                discharge_current = cycle_data['current'].values[discharge_start_idx:discharge_end_idx]
                discharge_time = cycle_data['time'].values[discharge_start_idx:discharge_end_idx]

                for i in range(len(discharge_time)-1):
                    if discharge_time[i+1] < discharge_time[i]:
                        try:
                            print("Time data is strange, idx : ", cycle, " i : ", i)
                            # print("discharge_value[i+1]  : ", discharge_time[i+1])
                            # print("discharge_value[i]: ", discharge_time[i])
                            del discharge_voltage[i+1]
                            del discharge_current[i+1]
                            del discharge_time[i+1]
                        except:
                            continue

                D_Voltage.append(discharge_voltage)
                D_Current.append(discharge_current)
                All_D_Time.append(discharge_time)

                discharge_time = cycle_data['time'].values[discharge_start_idx:discharge_end_idx]

                discharge_Time_value = 0
                for i in range(1,10):
                    tmp_idx = -i
                    if cycle_data['time'].values[:][tmp_idx] > 100:
                        discharge_Time_value = (discharge_time[tmp_idx])

                if discharge_Time_value == 0:
                    print("Time data error. please check")

                D_Time.append(discharge_Time_value)

                # 기존 Maryland 대학은 CC Charge만 했는데
                # Nasa는 CCCV Charge로 CV 데이터도 함께 존재함... 해당 부분 제거...
                charge_Vol = np.around(cycle_data['voltage'].values[:time_idx-1], 3)
                max_charge_vol = max(charge_Vol)
                max_charge_vol_idx = list(charge_Vol).index(max_charge_vol)

                C_Voltage.append(cycle_data['voltage'].values[:max_charge_vol_idx])
                C_Current.append(cycle_data['current'].values[:max_charge_vol_idx])

                charge_Time = cycle_data['time'].values[:max_charge_vol_idx][-1]

                # for i in range(1,10):
                #     tmp_idx = -i
                #     if charge_data['time'].values[:max_charge_vol_idx][tmp_idx] > 100:
                #         charge_Time = (charge_data['time'].values[:max_charge_vol_idx][tmp_idx])/30
                #
                # if charge_Time == 0:
                #     print("Time data error. please check")

                C_Time.append(charge_Time)

                whole_cap = cycle_data['capacity'].values[0]

                Capacity.append(whole_cap)
                SOH.append(whole_cap / 2)           # Nasa Battery rated capacity is 2Ah

                last_cycle = last_cycle + 1

                # print("2. discharge_voltage shape: ", np.shape(D_Voltage[-1]), " len : ", len(D_Voltage))
                # print("2.  discharge_time shape: ", np.shape(All_D_Time[-1]), " len : ", len(All_D_Time))
            except:
                print("Exception index : ", cycle)
                continue

    return D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, D_Time, C_Time, All_D_Time

def ReadResistanceData(FileNameList, BatteryDataDir):
    Resistance = []

    for FileName in FileNameList:
        if not FileName.endswith("resistance.xls") or FileName.endswith("xlsx"):
            continue

        print("Data Loading : {}".format(FileName))

        xlsxFile = os.path.join(BatteryDataDir, FileName)
        xls = pd.ExcelFile(xlsxFile)
        _data = pd.DataFrame()

        sheet_index = xls.sheet_names[:]

        # for sheets in sheet_index:
        df_xls = pd.read_excel(xlsxFile, sheet_name=sheet_index[0])
        _data = pd.concat([_data, df_xls], axis=0)

        resistance_data = _data.values

        resistance = []

        for i in range(len(resistance_data)):
            resistance.append(resistance_data[i][0])

    return resistance

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
    train_battery_list = ['B0025', 'B0027', 'B0028']
    #train_battery_list = ['B0033','B0046','B0047','B0048']
    #train_battery_list = ['B0026']
    #train_battery_list = ['B0005', 'B0006', 'B0018', 'B0046', 'B0047', 'B0048', 'B0033']


    dir_path = "../data/Nasa_data/BatteryAgingARC_change"

    voltage_list = []

    for battery in train_battery_list:
        BatteryDataDir = os.path.join(dir_path, battery)
        FileNameList = os.listdir(BatteryDataDir)

        D_Voltage, C_Voltage, D_Current, C_Current, Capacity, SOH, last_cycle, D_Time, C_Time, All_D_Time = ReadData(FileNameList, BatteryDataDir)

        np.save(BatteryDataDir + '/capacity', Capacity)
        np.save(BatteryDataDir + '/SOH', SOH)
        np.save(BatteryDataDir + '/discharge_data', D_Voltage)
        np.save(BatteryDataDir + '/discharge_current', D_Current)

        np.save(BatteryDataDir + '/charge_data', C_Voltage)
        np.save(BatteryDataDir + '/charge_current', C_Current)
        np.save(BatteryDataDir + '/last_cycle', last_cycle)

        np.save(BatteryDataDir + '/discharge_time', D_Time)
        np.save(BatteryDataDir + '/charge_time', C_Time)

        np.save(BatteryDataDir + '/discharge_time_all', All_D_Time)

        # np.save('../data/annormally_data/make_data_voltage', D_Voltage)
        #np.save('../data/annormally_data/make_data_Soc', D_Soc)

        # resistance = ReadResistanceData(FileNameList, BatteryDataDir)
        #
        # np.save(BatteryDataDir + '/resistance', resistance)


    # Density(voltage_list)

