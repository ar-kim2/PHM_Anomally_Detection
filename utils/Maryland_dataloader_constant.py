import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import csv
import collections
from utils.Entropy import DistEN
from utils.Entropy import drawEntropy
# from utils.Entropy import drawEntropy_imsi
from utils.Entropy import Density
import scipy

def ReadData(FileNamelist, dir_path):
    x_data = []
    y_data = []
    DCList = [] #list for only Discharge [CycleIndex, Voltage]
    CList = [] #list for only Charge [CycleIndex, Voltage]
    lastcycle = 0
    # FileNamelist.pop(0)  # 첫번재 엑셀 파일(실험)이 다른 실험에 비해 특이해서 제외
    for filename in FileNamelist:
        if not filename.endswith("xls") or filename.endswith("xlsx") or filename.startswith('~$'):
            continue
        print("Data Loading : {}".format(filename))
        workbook = xlrd.open_workbook(os.path.join(dir_path,filename))

        # 전체 battery,
        if (workbook.sheet_names().__contains__('Statistics_1-011') or
                workbook.sheet_names().__contains__('Statistics_1-009') or
                workbook.sheet_names().__contains__('Statistics_1-008') or
                workbook.sheet_names().__contains__('Statistics_1-010') or
                workbook.sheet_names().__contains__('Statistics_1-006') or
                workbook.sheet_names().__contains__('Statistics_1-007') or
                workbook.sheet_names().__contains__('Statistics_1-007') or
                workbook.sheet_names().__contains__('Statistics_1-001') or
                workbook.sheet_names().__contains__('Statistics_1-002') or
                workbook.sheet_names().__contains__('Statistics_1-003') or
                workbook.sheet_names().__contains__('Statistics_1-012') or
                workbook.sheet_names().__contains__('Channel_Chart')):
            SheetLength = len(workbook.sheets()) - 1
        else:
            SheetLength = len(workbook.sheets())

        for index in range(1, SheetLength):
            worksheet = workbook.sheet_by_index(index)
            CycleIndex = worksheet.col_values(5, 1)
            StepIndex = worksheet.col_values(4, 1)
            charge_capacity = np.array(worksheet.col_values(8, 1))
            discharge_capacity = np.array(worksheet.col_values(9, 1))
            # test 전에 충전이 되어있는 경우를 고려
            try:
                # 두 번 이상의 cycle이 수행된 경우
                if CycleIndex.index(2.0):
                    TwoIdx = CycleIndex.index(2.0) - 1
                    if (discharge_capacity[TwoIdx] - charge_capacity[TwoIdx]) >= 0:
                        charge_capacity = charge_capacity + (discharge_capacity[TwoIdx] - charge_capacity[TwoIdx])
                    if (discharge_capacity[-1] - charge_capacity[-1]) >= 0:
                        charge_capacity = charge_capacity + (discharge_capacity[-1] - charge_capacity[-1])
            except:
                # 한번 만 수행된 경우
                if (discharge_capacity[-1] - charge_capacity[-1]) >= 0:
                    charge_capacity = charge_capacity + (discharge_capacity[-1] - charge_capacity[-1])
            CycleIndex = [int(x + lastcycle) for x in CycleIndex]
            lastcycle = int(CycleIndex[-1])

            X = [CycleIndex, StepIndex, worksheet.col_values(7, 1), worksheet.col_values(6, 1), worksheet.col_values(1, 1)]   #
            Cap = (charge_capacity - discharge_capacity)
            X = np.transpose(X)

            '''However, sometimes what we want is to append all the elements contained in the 
            list rather the list itself. You can do that manually of         course,
            but a better solution is to use extend() as follows:'''
            x_data.extend(X.tolist())
            y_data.extend(Cap.tolist())

    DCList, DCList_current, DCList_time, DCList_time_all = seperating(x_data, mode='discharge')

    return x_data, y_data, DCList, DCList_current, DCList_time_all

def seperating(x_data, mode='discharge'):
    """
    x_data로 부터 mode (discharge, charge)에 따라서 voltage list, current list를 분리해서 반환한다.
    :param x_data: 배터리 데이터
    :return SlicingDCList: 분리된 voltage list
    :return SlicingDCList_current: 분리된 current list
    """

    # Discharge list = step Index is larger than 6
    IndexList = np.array(x_data)[:, 1]
    DCList = []
    SlicingDCList = []
    SlicingDCList_current = []
    SlicingDCList_time = []
    SlicingDCListall_time = []

    pivot = 0

    for index, value in enumerate(IndexList):
        if mode is 'discharge':
            if value == 7:
                # Cycle Index and Voltage value are appended

                DCList.append([x_data[index][0], x_data[index][2], x_data[index][3], x_data[index][4]])   #
        elif mode is 'charge':
            if value < 7:
                DCList.append([x_data[index][0], x_data[index][2], x_data[index][3], x_data[index][4]])   #

    print("shape ", np.shape(DCList))

    CycleIndex = np.array(DCList)[:, 0]

    for idx in range(1, int(DCList[-1][0])+1):
        start = pivot
        try :
            pivot = list(CycleIndex).index(int(idx+1), pivot)
        except:
            pivot = -1

        tmp_Time = list(np.array(DCList)[start:pivot, 3])
        start_time = tmp_Time[0]

        for i in range(len(tmp_Time)):
            tmp_Time[i] = tmp_Time[i]-start_time

        SlicingDCListall_time.append(tmp_Time)
        SlicingDCList_time.append((tmp_Time[-1]-tmp_Time[0])/30)
        SlicingDCList.append(list(np.array(DCList)[start:pivot, 1]))
        SlicingDCList_current.append(list(np.array(DCList)[start:pivot, 2]))

    return SlicingDCList, SlicingDCList_current, SlicingDCList_time, SlicingDCListall_time    #(# of cycle, ?length of each cycle, 1(voltage))

def write_csv_log(dir_path, line, option='a'):
    csv_path = "{0}{1}".format(dir_path, "capacity_list.csv")
    if option == "w" and os.path.exists(csv_path):
        return False
    f = open(csv_path, option, encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(line)
    f.close()
    return True

def Capacity(x_data, y_data, rated_Capacity):
    '''
    :param x_data: [Cycle index, Step Index, Voltage, Current]
    :param y_data: SOC
    :return: capacity : 충전가능 최대 용량
              int(cpivot) : SOH 0까지의 cycleindex
    '''

    capacity = [0]
    IndexList = list(np.array(x_data)[:, 1])
    # IndexList = [x_data[idx][1] for idx in range(0, len(x_data))]
    CycleIndex = list(np.array(x_data)[:, 0])
    pivot = 0
    smooth_index = 0
    cpivot = 1.0 # 몇번 cycle 확인위한 변수

    capacity_value = 0

    discharge_cap_value = 0

    prev_cap = 0

    for idx in range(0, int(x_data[-1][0])):

        try:
            ChargeStartIdx = IndexList.index(1, pivot)
        except:
            continue

        pivot = ChargeStartIdx
        try:
            IndexList.index(7, pivot) # pivot 부터 step 8 인 index 있는가?
            ChargeEndIdx = IndexList.index(7, pivot) - 1 # EndIdx = step7의 마지막 index
            DischargeStartIdx = IndexList.index(7, pivot) -1
            if CycleIndex[ChargeEndIdx] != cpivot: # 만약 해당 사이클이 step 8까지 가지 않고 조기 종료된 경우 다음 사이클 step1로
                ChargeEndIdx = IndexList.index(7, pivot) - 1
                DischargeStartIdx = IndexList.index(7, pivot) -1
        except:
            ChargeEndIdx = -1
            DischargeStartIdx = -1

        pivot = DischargeStartIdx

        try:
            IndexList.index(8, pivot) # pivot 부터 step 8 인 index 있는가?
            DischargeEndIdx = IndexList.index(8, pivot) - 1 # EndIdx = step7의 마지막 index
            if CycleIndex[DischargeEndIdx] != cpivot: # 만약 해당 사이클이 step 8까지 가지 않고 조기 종료된 경우 다음 사이클 step1로
                DischargeEndIdx = IndexList.index(8, pivot) - 1
        except:
            DischargeEndIdx = -1

        charge_cap_value = y_data[ChargeEndIdx] - y_data[ChargeStartIdx]
        discharge_cap_value = y_data[DischargeStartIdx] - y_data[DischargeEndIdx]

        # 간혹 배터리가 방전됬는데, 방전된만큼 충전되지 않는 경우가 발생.
        # if charge_cap_value < (discharge_cap_value * 0.9):
        #     charge_cap_value = discharge_cap_value
        # elif discharge_cap_value != 0 and charge_cap_value > discharge_cap_value:
        #     charge_cap_value = discharge_cap_value

        print("charge value ", idx, " : ", charge_cap_value )
        print("discharge value ", idx, " : ", discharge_cap_value )

        if prev_cap < 0.1 :
            capacity_value = charge_cap_value + prev_cap #charge_cap_value + 0.01  #capacity_value + charge_cap_value   # y data is capacity
        else:
            capacity_value = charge_cap_value

        prev_cap =  charge_cap_value-discharge_cap_value

        print("cap value ", idx, " : ", capacity_value)

        #capacity_value = capacity_value*rated_Capacity/100  # cuz Cacity_value is in percentage

        # 실험 불완전성 때문에, capacity 감소가 큰 경우가존재함. 이를 막기 위해
        # 90퍼센트 이하로 capacity가 감소하면 이전 capacity와 같은 값을 지니도록함.
        if capacity_value > 1.1:
            capacity.append(1.1)
        elif capacity_value < capacity[-1]*0.9:
            print("cap value : ", capacity_value)
            capacity.append(capacity[-1])
        else:
            capacity.append(capacity_value)

        #capacity_value = capacity_value - discharge_cap_value

        #SOH가 60%일 경우까지만 데이터에 포함하기위해
        # if capacity[-1] < rated_Capacity * 0.6:
        #     print("i : ", idx, " capacity : ", capacity[-1])
        #     print("SOH is 0%")
        #     break
        # capacity.append(capacity_value)
        pivot = DischargeEndIdx + 1
        cpivot += 1.0

    capacity.pop(0)

    return capacity, int(cpivot)

def MF(list, kernal_size):
    '''median filter'''
    new_list = [np.median(list[idx:idx+kernal_size]) for idx in range(len(list) - kernal_size+1)]
    for _ in range(kernal_size-1):
        new_list.append(np.median(list[len(list)-kernal_size:len(list)]))

    return new_list

def smoothListGaussian(list,degree=5):
    window=degree*2-1 # 9
    weight=np.array([1.0]*window)
    weightGauss=[]
    for i in range(window):
        i=i-degree+1 #
        frac=i/float(window)
        gauss=1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight=np.array(weightGauss)*weight
    smoothed=[0.0]*(len(list)-window)
    for i in range(len(smoothed)):
        smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
    return smoothed

def removeEmptyData(DC_list, DC_list_current, C_list, C_list_current, last_cycle):
    print("DC_list : ", np.shape(DC_list))
    print("DC_list_current : ", np.shape(DC_list_current))
    print("C_list : ", np.shape(C_list))
    print("C_list_current : ", np.shape(C_list_current))

    for idx in range(len(DC_list_current)-1):
        if len(DC_list_current[idx]) == 0:
            DC_list.pop(idx)
            DC_list_current.pop(idx)
            C_list.pop(idx)
            C_list_current.pop(idx)
            last_cycle = last_cycle -1

    for idx in range(len(C_list_current)-1):
        if len(C_list_current[idx]) == 0:
            DC_list.pop(idx)
            DC_list_current.pop(idx)
            C_list.pop(idx)
            C_list_current.pop(idx)
            last_cycle = last_cycle -1

    return DC_list, DC_list_current, C_list, C_list_current, last_cycle

#######################여러 배터리 한 그래프로 그릴때#################################
# if __name__ == "__main__":
#     color_list = { 1:'maroon', 2:'red', 3:'coral', 4:'darkgreen', 5:'navy', 6:'cornflowerblue',
#                    7:'b', 8:'skyblue', 9:'peachpuff', 10:'mediumseagreen', 11:'mediumspringgreen', 12:'lightgreen'}
#     dir_path = "D:/CODE/untitled/PHMSoH/SOHdata/dis_current_constant/"
#     battery_list = ['CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
#     maker_list = ['o','v','s','P','D','*']
#     smooth = True
#     all_cap_dic = collections.OrderedDict()
#     for battery in battery_list:
#         battery_path = dir_path + battery
#         FileNamelist = os.listdir(battery_path)
#         capacity_list = np.load(battery_path + '/capacity.npy')
#         if smooth:
#             capacity_list_appended = np.append(np.full((19, 1), capacity_list[0], dtype=np.float32), capacity_list)
#             capacity_list_appended = np.append(capacity_list_appended, np.full((19, 1), capacity_list[-1], dtype=np.float32))
#             smooth_capacity_list = smoothListGaussian(capacity_list_appended, 20)
#             all_cap_dic[battery] = smooth_capacity_list
#         else:
#             all_cap_dic[battery] = capacity_list
#     saver_path = dir_path + "/saver"
#     fig = plt.figure(figsize=(6, 6), dpi=300)
#     gs = gridspec.GridSpec(1, 1)
#     capacity_graph = fig.add_subplot(gs[0,0])
#     idx = 1
#     for battery in all_cap_dic.keys():
#         capacity_graph.plot(range(len(all_cap_dic[battery])), all_cap_dic[battery], color=color_list[idx],
#                             label = battery, marker=maker_list[idx-1], linewidth=0.5, markevery=10, linestyle='--',
#                             markersize=3)
#         idx += 1
#     capacity_graph.axhline(y=1.1 * 0.8, xmin=0.02, xmax=0.98, linestyle='--', color='grey')
#     # capacity_graph.annotate('0.88', xy=(0, 0.2), xycoords='axes fraction')
#     capacity_graph.set_ylabel("Capacity(Ah)")
#     capacity_graph.set_xlabel("Cycles")
#     # capacity_graph.set_xlim(0, 800)
#     capacity_graph.set_ylim(0.80, 1.2)
#     capacity_graph.legend(fontsize='medium')
#     plt_file_name = dir_path + "/training_data_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".tif"
#     plt.savefig(plt_file_name, format='tif')
#     plt.close()
########################################################
if __name__ == "__main__":
    """
    xls 파일로 부터 cycle index, SOH_capacity, Discharge_Volatege, Discharge_Current, Charge_Voltage, Charge_Current를 추출해서 npy 파일로 저장한다.
    """
    #TODO:: entropy부분 entropy 파일로 이동.
    #TODO:: Draw부분 수정

    dir_path = "../data/dis_current_constant/CS2_XX_0/"

    save = True
    smooth = False
    #battery_list = ['CS2_33', 'CS2_34','CS2_35','CS2_36','CS2_37','CS2_38']
    #battery_list = ['CX2_33','CX2_34','CX2_35','CX2_36']
    #battery_list = ['CS2_37']
    battery_list = ['CS2_3/total']


    rated_Capacity = 1.1

    for battery in battery_list:
        battery_path = dir_path + battery
        FileNamelist = os.listdir(battery_path)

        if save:
            print("save START ", battery)
            #x_data, y_data, DC_list, C_list, DC_list_current, C_list_current, DC_list_time, C_list_time = ReadData(FileNamelist, battery_path)
            x_data, y_data, DC_list, DC_list_current, DC_list_time_all = ReadData(FileNamelist, battery_path)

            print("x data shape : ", np.shape(x_data))

            capacity_list1, last_cycle1 = Capacity(x_data, y_data, rated_Capacity)

            np.save(battery_path + '/capacity', capacity_list1)
            np.save(battery_path + '/discharge_data', DC_list)
            np.save(battery_path + '/discharge_current', DC_list_current)
            np.save(battery_path + '/discharge_time_all', DC_list_time_all)

            '''
            capacity            : capacity              [cycle_idx][capacity]
            discharge_data      : discharge volatage    [cycle_idx][voltage]
            discharge_current   : discharge current     [cycle_idx][current]
            charge_data         : charge voltage        [cycle_idx][voltage]
            charge_current      : charge current        [cycle_idx][current]
            last_cycle          : cycle number          int
            '''

        else:
            print("load START")
            capacity_list = np.load(battery_path + '/capacity.npy')
            DC_list = np.load(battery_path  + '/discharge_data.npy')
            DC_list_current = np.load(battery_path  + '/discharge_current.npy')
            C_list = np.load(battery_path + '/charge_data.npy')
            C_list_current = np.load(battery_path + '/charge_current.npy')
            last_cycle = np.load(battery_path + '/last_cycle.npy')

    if save is True:
        print('Done')
        exit()

