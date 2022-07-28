import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def AutoCorr(list, lag):
    AC_list = []
    correlation = np.corrcoef(list, list)
    correlation = correlation[0][1]
    AC_list.append(correlation)
    for shift in range(1, lag+1):
        list_lag = list[shift:]
        list_lag_minus = list[:-shift]
        correlation = np.corrcoef(list_lag, list_lag_minus)[0][1]
        AC_list.append(correlation)

    return AC_list

def DrawGraph(list, file_name):
    f, ax = plt.subplots()
    ax.set_ylim(-0.5, 1.25)
    for idx in range(1, len(list)+1):
        ax.plot(list[idx-1], color=color_list[idx], linewidth = 0.5, label=file_name[idx-1])

    ax.legend(fontsize='x-small')
    plt.show()

def Entropy0(density):
    ''' normal Entropy'''
    # x = np.multiply(x,100)
    # indices = x - 200
    # density_list = [density[int(x)] for x in indices]
    ent = -1 * np.sum(np.multiply(density, np.log10(density)))
    # ent = -1*ent1/np.log2(len(density))

    return ent

def Entropy1(density, current_value, time_value, beta):
    ''' time compensated Entropy'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    integ_CB = integrate.cumtrapz(abs_current, time_value)[-1]

    # x = np.multiply(x,100)
    # indices = x - 200
    # density_list = [density[int(x)] for x in indices]
    ent1 = np.sum(np.multiply(density, np.log10(density)))
    # ent = -1*ent1/np.log2(len(density))
    ent = -1 * beta * ent1 / (time_value[-1])
    #ent = -1*beta*ent1/(current * time_value[-1])
    #ent = -1 * beta * ent1 / (time_value[-1])

    #ent = -1 * beta * ent1 / integ_CB
    #ent = -1 * beta * ent1 / time_value[-1]

    return ent, integ_CB


def Entropy2(density, time_value, current_value, beta):
    '''current constant'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    #time_compenate = time_value[-1]

    integ_CB = integrate.cumtrapz(abs_current, time_value)[-1]

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    # if len_value == 0:
    #     print("len_val is zero")
    # if current == 0:
    #     current = 0.001
    #     print("current is zero ")

    #ent = -1 * beta * ent1 * ((-30.62) * np.exp(0.134 * current) + 31.96 * np.exp(0.12 * current)) / integ_CB
    #ent = -1 * beta * ent1 * ((-30.6166) * np.exp(0.134 * current) + 31.9595 * np.exp(0.12 * current)) / integ_CB

    #ent = -1 * beta * ent1 * (0.068 * np.exp(1.341 * current)+0.697) / (current*time_value[-1])
    ent = -1 * beta * ent1 * (0.0737 * np.exp(1.27 * current) + 0.702) / (current * time_value[-1])
    #ent = -1 * beta * ent1 * (1.411 / np.exp(0.386 * current)) / integ_CB
    #ent = -1 * beta * ent1 * (1.419 / np.exp(0.39 * current)) / integ_CB
    #ent = -1 * beta * ent1 * (1.411 / np.exp(0.386 * current)) / integ_CB

    return ent, np.abs(np.round(current, 2))

def Entropy2_2(density, current_value, beta, time_rate):
    '''current changing'''
    abs_current = np.abs(current_value)

    # current = 2
    # last_time = 0

    # for i in range(3, len(time_rate)-1):
    #     if (time_rate[i+1]-time_rate[i]) > 0:
    #         current = current+(abs_current[i]*(time_rate[i+1]-time_rate[i]))
    #         last_time = last_time + (time_rate[i+1]-time_rate[i])
    #
    # current = current/last_time

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    integ_CB = integrate.cumtrapz(abs_current, time_rate)[-1]

    # integ_CB = 0
    #
    # for i in range(1, len(time_rate)-1):
    #     dx1 = (time_rate[i] + time_rate[i-1]) /2
    #     dx2 = (time_rate[i+1] + time_rate[i]) /2
    #
    #     integ_CB = integ_CB + ( -current_value[i]*(dx2 - dx1))

    # current = integ_CB/time_rate[-1]

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    #ent = -1 * beta * ent1 / ((integ_CB) * (0.663 * np.exp(0.456 * current)))


    ent = -1 * beta * ent1 / ((integ_CB) * ((0.072 * np.exp(1.287 * current)) + 0.7))
    #ent = -1 * beta * ent1 / ((current*time_rate[-1]) * (0.073 * np.exp(1.27 * current) + 0.702))


    #ent = -1 * beta * ent1 * (1.388 / np.exp(0.371 * current)) / (integ_CB)  #(current * len_value)  #

    return ent, np.abs(np.round(current, 2))

def Entropy2_2_prev(density, len_value, current_value, beta, rate, mode, no_coff):
    '''current changing'''
    if mode is 'discharge':
        min_current = np.abs(np.round(np.min(current_value),2))
        if min_current > 2.3:
            abs_min_current = 1
        else:
            abs_min_current = min_current
        ent1 = np.sum(np.multiply(density, np.log10(density)))
        if no_coff: # coefficient 없는 경우
            ent = -1 * beta * ent1 / (abs_min_current * len_value * rate)
        else:
            ent = -1 * beta * ent1 * (1.31 / np.exp(0.227 * abs_min_current)) / (abs_min_current * len_value * rate)

        return ent, min_current
    elif mode is 'charge':
        charge_current = np.abs(np.round(np.max(current_value), 2))
        ent1 = np.sum(np.multiply(density, np.log10(density)))
        ent = -1 * beta * ent1 * (1.31 / np.exp(0.227 * 1))  / (1*len_value)

        return ent, charge_current

    else:
        print('please select discharge and charge mode')
        exit()

def Entropy3(density, len_value, current_value, beta, mode='compensated'):
    '''compensate or not 비교하기위한 함수'''
    min_current = np.min(current_value)
    if min_current > -0.2:
        abs_min_current = 1
    else:
        abs_min_current = np.abs(min_current)
    ent1 = np.sum(np.multiply(density, np.log10(density)))
    if mode is 'compensated':
        ent = -1*beta*ent1*(1.31 / np.exp(0.227*abs_min_current)) /(abs_min_current * len_value)
        # ent = -1*beta*ent1/(abs_min_current * len_value)
    elif mode is 'not':
        print("not compensated enropy")
        ent = -1*ent1
    else:
        print("mode should be 'compensated' or 'not'.")
        raise AssertionError

    # ent = -1*beta*ent1/(np.log(len_value*abs_min_current)/3)
    # ent = -1*beta*ent1/len_value

    return ent, np.round(min_current, 2)

def Entropy4_DC(density, len_value, current_value, beta, rated_cap):
    '''current constant, check Rated Capacity'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    ent2 = -1 * beta * ent1 * (1.31 / np.exp(0.227 * current)) / (current*len_value)

    ent = ent2 * 0.352 * np.exp(0.934 * rated_cap)

    #ent = ent2

    return ent, np.abs(np.round(current, 2))

def Entropy4_C(density, len_value, current_value, beta, rated_cap):
    '''current constant, check Rated Capacity'''
    abs_current = np.abs(current_value)

    current = 0

    for i in range(3, len(abs_current)):
        if abs_current[i] > 0.1:
            current = abs_current[i]
            break

    ent1 = np.sum(np.multiply(density, np.log10(density)))

    ent2 = -1 * beta * ent1 * (1.31 / np.exp(0.227 * current)) / (current*len_value)

    ent = ent2 * 0.637 * np.exp(0.43 * rated_cap)


    #ent = ent2

    return ent, np.abs(np.round(current, 2))

def Moving_Avg_Filter(list, num):
    '''median filter'''
    new_list = []

    for i in range(len(list)):
        if i < num:
            sum = 0
            for j in range(i+1):
                sum = sum + list[j]
            new_list.append(sum/(i+1))
        else:
            sum = 0
            for j in range(i-num+1, i+1):
                sum = sum + list[j]
            new_list.append(sum/num)

    return new_list

def MF(list, kernal_size):
    '''median filter'''
    new_list = [np.median(list[idx:idx+kernal_size]) for idx in range(len(list) - kernal_size+1)]
    for _ in range(kernal_size-1):
        new_list.append(np.median(list[len(list)-kernal_size:len(list)]))

    return new_list

def drawEntropy(density, len_list):
    entropy_list = []
    min_value = min(len_list)
    for idx in range(len(density)):
        entropy = Entropy3(density[idx], len_list[idx], 50)
        entropy_list.append(entropy)
    return entropy_list

def Density(list):
    # all_exp_log_dens = []
    all_prob = []
    len_list = []
    for idx in range(len(list)):
        hist, bin_edges = np.histogram(list[idx], bins=np.linspace(2,5,17))
        list_size = len(list[idx])
        len_list.append(list_size)
        hist = hist + np.ones(16)
        prob = hist/(list_size+16)

        all_prob.append(prob)

    return all_prob, len_list

def EntropyForSOH(list, beta):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    all_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list)):
        prob_list = []
        temp_len = []
        for idx2 in range(len(list[idx])): #cycle
            hist, bin_edges = np.histogram(list[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))
        len_list.append(temp_len)
        all_prob.append(prob_list)

    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list[idx])):
            entropy_list.append(Entropy3(all_prob[idx][idx2], len_list[idx][idx2], beta))
        # entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list

def EntropyAndProb(list, beta):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    all_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list)):
        prob_list = []
        temp_len = []
        for idx2 in range(len(list[idx])): #cycle
            hist, bin_edges = np.histogram(list[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))
        len_list.append(temp_len)
        all_prob.append(prob_list)

    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list[idx])):
            # idx 번째 배터리의 idx2번째 cycle
            entropy_list.append(Entropy3(all_prob[idx][idx2], len_list[idx][idx2], beta))
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list, all_prob

def DistEntropyForSOH(M, list):
    '''
    Distribution Entropy 계산
    :param M: the number of bins
    :param list:
    :return:
    '''
    all_prob_list = []
    for idx in range(len(list)):
        prob_list = []
        for idx2 in range(len(list[idx])):
            UDM = UpperDM(list[idx][idx2])
            hist, bin_edges = np.histogram(UDM, bins=np.linspace(0,2, M+1))
            hist = hist + np.ones(M)
            list_size = len(UDM) + M
            prob_list.append(hist/list_size)
        all_prob_list.append(prob_list)
    all_entropy_list = []
    for idx in range(len(list)):
        entropy_list = []
        for idx2 in range(len(list[idx])):
            entropy_list.append(-1*np.sum(all_prob_list[idx][idx2]*np.log2(all_prob_list[idx][idx2]))/np.log2(M))
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)

    return all_entropy_list

def EntropyForSOH_withCurrent_prev(list_A, current_list, beta, mode):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산'''
    all_prob = []
    return_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list_A)):
        prob_list = []
        temp_len = []
        retrun_prob_list = []
        for idx2 in range(len(list_A[idx])): #cycle
            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list_A[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))

            hist, _ = np.histogram(list_A[idx][idx2], bins=np.linspace(2, 5, 7))
            retrun_prob_list.append(hist/list_size)
        len_list.append(temp_len)
        all_prob.append(prob_list)
        return_prob.append(retrun_prob_list)

    all_entropy_list = [] # 모든 배터리에 대한 엔트로피 리스트
    # all_discharge_current_list = [] # 모든 배터리에 대한 방전전류 리스트
    all_concat_list = []
    for idx in range(len(list_A)):
        pivot_len = 0
        entropy_list = []
        min_current_list = []
        concat_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list_A[idx])):
            entropy_cycle, min_current = Entropy2(all_prob[idx][idx2], len_list[idx][idx2],
                                                  current_list[idx][idx2], beta)
            entropy_list.append(entropy_cycle)
            min_current_list.append(min_current)
        # entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)
        # all_discharge_current_list.append(min_current_list)

    if mode is 'discharge':
        return all_entropy_list, return_prob, len_list
    elif mode is 'charge':
        return all_entropy_list

def EntropyForSOHProb_withCurrent(list_A, current_list, beta, mode, time):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''
    all_prob = []
    return_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list_A)):
        prob_list = []
        temp_len = []
        retrun_prob_list = []
        for idx2 in range(len(list_A[idx])): #cycle
            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list_A[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))

            hist, _ = np.histogram(list_A[idx][idx2], bins=np.linspace(2, 5, 7))
            retrun_prob_list.append(hist/list_size)
        len_list.append(temp_len)
        all_prob.append(prob_list)
        return_prob.append(retrun_prob_list)

    all_entropy_list = [] # 모든 배터리에 대한 엔트로피 리스트
    # all_discharge_current_list = [] # 모든 배터리에 대한 방전전류 리스트
    #all_concat_list = []
    for idx in range(len(list_A)):
        entropy_list = []
        min_current_list = []
        concat_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list_A[idx])):
            entropy_cycle, min_current = Entropy2(all_prob[idx][idx2], time[idx][idx2], current_list[idx][idx2], beta)
            entropy_list.append(entropy_cycle)
            min_current_list.append(min_current)
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)
        #all_concat_list.append(list(np.concatenate((np.array(entropy_list)[:, np.newaxis], np.array(min_current_list)[:, np.newaxis]), axis=1)))
        # all_discharge_current_list.append(min_current_list)

    if mode is 'discharge':
        return all_entropy_list, return_prob
    elif mode is 'charge':
        return all_entropy_list

def EntropyForSOHProb_withCurrent_oneBattery(list_A, current_list, beta, mode):
    '''
    기존의 EntropyForSOHProb_withCurrent는 전체 battery data를 한번에 처리한다.
    이 Function은 1개의 battery data만 처리한다.
    :param list_A: 전압 정보, [cycle][voltage]
    :param current_list: 전류정보, [cycle][curent]
    :param beta: lenght coefficient
    :param mode: discharge/charge mode
    :return entropy_list: entropy [cycle][entropy]
    :return retrun_prob_list : probability [cycle][probability]
    '''
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''

    prob_list = []
    list_len = []
    retrun_prob_list = []

    for idx2 in range(len(list_A)): #cycle
        hist, bin_edges = np.histogram(list_A[idx2], bins=np.linspace(2,5,17)) # 16 bins
        list_size = len(list_A[idx2])
        list_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
        hist = hist + np.ones(16)
        prob_list.append(hist/(list_size+16))

        hist, _ = np.histogram(list_A[idx2], bins=np.linspace(2, 5, 7))
        retrun_prob_list.append(hist/list_size)

    entropy_list = []

    # min_len_value = min(len_list[idx]) # 길이의 최소값

    for idx2 in range(len(list_A)):
        try:
            entropy_cycle, min_current = Entropy2(prob_list[idx2], list_len[idx2], current_list[idx2], beta)
            #entropy_cycle = Entropy3(prob_list[idx2], list_len[idx2], beta)
            entropy_list.append(entropy_cycle)
        except:
            print("Exception idx : ", idx2)
    entropy_list = MF(entropy_list, 3)

    if mode is 'discharge':
        return entropy_list, retrun_prob_list
    elif mode is 'charge':
        return entropy_list, retrun_prob_list

def EntropyForSOHProb_withCurrent_oneNasaBattery(list_A, current_list, beta, mode, time, SOH):
    '''
    기존의 EntropyForSOHProb_withCurrent는 전체 battery data를 한번에 처리한다.
    이 Function은 1개의 battery data만 처리한다.
    :param list_A: 전압 정보, [cycle][voltage]
    :param current_list: 전류정보, [cycle][curent]
    :param beta: lenght coefficient
    :param mode: discharge/charge mode
    :return entropy_list: entropy [cycle][entropy]
    :return retrun_prob_list : probability [cycle][probability]
    '''
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''

    prob_list = []
    list_len = []
    retrun_prob_list = []

    #Cal_Ri = []
    start_idx_list = []
    end_idx_ilst = []

    for idx2 in range(len(list_A)): #cycle
        if len(current_list[idx2]) <= 1:
            start_idx_list.append(0)
            end_idx_ilst.append(-1)
            continue

        # tmp_Cal_Ri = []
        # for i in range(1, len(list_A[idx2]-1)):
        #     vol_diff = list_A[idx2][i-1] - list_A[idx2][i]
        #     vol_diff = np.abs(vol_diff)
        #     tmp_curr = np.abs(current_list[idx2][i-1])
        #     if tmp_curr > 0.01:
        #         tmp_Cal_Ri.append(vol_diff / tmp_curr)
        #         #print(" i : ", i, "voltage1 : ", list_A[idx2][i], "voltage2 : ", list_A[idx2][i - 1], "current : ",
        #          #     -current_list[idx2][i], " vol_diff : ", vol_diff, "Ri : ", tmp_Cal_Ri[-1])
        #
        # Cal = np.average(tmp_Cal_Ri)
        # if Cal > 0.15:
        #     Cal = Cal_Ri[-1]

            # Cal_Ri.append(Cal)

        try:
            Ri = (-0.082 * SOH[idx2]) + 0.21
        except:
            print("SOH length mismatch")

        #Ri = (-0.082 * SOH[idx2]) + 0.22

        # V = IR

        # for i in range(len(list_A[idx2])):
        #     try:
        #         list_A[idx2][i] = list_A[idx2][i] + (np.multiply(Ri, np.abs(current_list[idx2][i])))
        #     except:
        #         print(" i : ", i, "voltage : ", list_A[idx2][i], "current : ", -current_list[idx2][i], "resistance : ", Ri)


        idx_list = np.where(np.array(list_A[idx2])<3.1)

        if np.size(idx_list[0]) > 0:
            end_idx = idx_list[0][0]-1
        else :
            end_idx = len(list_A[idx2])-1

        idx_list = np.where(np.array(list_A[idx2]) > 4)

        if np.size(idx_list[0]) > 0:
            start_idx = idx_list[0][-1]+1
        else:
            start_idx = 0

        start_idx_list.append(start_idx)
        end_idx_ilst.append(end_idx)

        list_A2 = np.array(list_A[idx2][start_idx:end_idx])

        # if idx2 == 100:
        #     plt.figure()
        #     plt.plot(list_A2)
        #     plt.show()


        #print(list_A2[:])
        # tmp = np.multiply(list_A[idx2], 10)
        # tmp = np.trunc(tmp)
        # tmp = tmp/10
        #
        # try:
        #     vol_end_idx = list(tmp).index(3.1)
        # except:
        #     try:
        #         vol_end_idx = list(tmp).index(3.0)
        #     except:
        #         vol_end_idx = -1
        #         print("last volatge value : ",tmp[-1], "origin value : ", list_A[idx2][-1])
        hist, bin_edges = np.histogram(list_A2, bins=np.linspace(2.5, 4.5, 17)) # 16 bins
        list_size = len(list_A[idx2])
        list_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
        hist = hist + np.ones(16)
        prob_list.append(hist/(list_size+16))

        #hist, _ = np.histogram(list_A2, bins=np.linspace(2, 5, 7))
        #retrun_prob_list.append(hist/list_size)

    entropy_list = []
    curremt_sum_list = []

    # min_len_value = min(len_list[idx]) # 길이의 최소값

    for idx2 in range(1, len(list_A)-1):
        if len(list_A[idx2]) <= 1:
            print("list size is zero, idx : ", idx2)
            entropy_list.append(entropy_list[-1])
            continue

        try:
            entropy_cycle = Entropy0(prob_list[idx2])
        except:
            entropy_list.append(entropy_list[-1])
            print("CHECK ERROR : prob_list ", np.shape(prob_list), " list_A : ", np.shape(list_A))


        # print("prob_list shape : ", np.shape(prob_list))
        # print("current_list shape : ", np.shape(current_list))
        # print("time shape : ", np.shape(time))

        #entropy_cycle, sum_cur = Entropy1(prob_list[idx2], current_list[idx2], time[idx2], beta)
        #entropy_cycle, min_current = Entropy2(prob_list[idx2], time[idx2], current_list[idx2], beta)

        #entropy_cycle, min_current = Entropy2_2(prob_list[idx2], current_list[idx2], beta, time[idx2])

        # #entropy_cycle = Entropy3(prob_list[idx2], list_len[idx2], beta)

        # if mode is 'discharge':
        #     entropy_cycle, min_current = Entropy4_DC(prob_list[idx2], time[idx2], current_list[idx2], beta, ratedCap)
        # else:
        #     entropy_cycle, min_current = Entropy4_C(prob_list[idx2], time[idx2],
        #                                              current_list[idx2], beta, ratedCap)

        entropy_list.append(entropy_cycle)
        #curremt_sum_list.append(sum_cur)

    #entropy_list = MF(entropy_list, 3)

    #entropy_list = Moving_Avg_Filter(entropy_list, 5)
    #entropy_list = Moving_Avg_Filter(entropy_list, 10)

    #curremt_sum_list = MF(curremt_sum_list, 3)


    if mode is 'discharge':
        return entropy_list, prob_list, list_A, curremt_sum_list, start_idx_list, end_idx_ilst
    elif mode is 'charge':
        return entropy_list, retrun_prob_list

def EntropyForSOHProb_withCurrent_oneNasaBattery_0311(list_A, current_list, beta, mode, time, ratedCap, rate):
    '''
    기존의 EntropyForSOHProb_withCurrent는 전체 battery data를 한번에 처리한다.
    이 Function은 1개의 battery data만 처리한다.
    :param list_A: 전압 정보, [cycle][voltage]
    :param current_list: 전류정보, [cycle][curent]
    :param beta: lenght coefficient
    :param mode: discharge/charge mode
    :return entropy_list: entropy [cycle][entropy]
    :return retrun_prob_list : probability [cycle][probability]
    '''
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''

    prob_list = []
    list_len = []
    retrun_prob_list = []

    for idx2 in range(len(list_A)): #cycle

        #print("Discharge Voltage : ", np.shape(list_A[idx2]), " i : ", idx2)

        if len(list_A[idx2]) == 0:
            continue

        # list_A[idx2] = Moving_Avg_Filter(list_A[idx2], 10)
        #print("MF Voltage : ", np.shape(list_A[idx2]))


        hist, bin_edges = np.histogram(list_A[idx2], bins=np.linspace(2,5,17)) # 16 bins
        list_size = len(list_A[idx2])
        list_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
        hist = hist + np.ones(16)
        prob_list.append(hist/(list_size+16))

        hist, _ = np.histogram(list_A[idx2], bins=np.linspace(2, 5, 7))
        #retrun_prob_list.append(hist/list_size)

    entropy_list = []

    # min_len_value = min(len_list[idx]) # 길이의 최소값

    for idx2 in range(1, len(list_A)-1):
        # try:
        if len(list_A[idx2]) == 0:
            continue
        entropy_cycle = Entropy0(prob_list[idx2])

        #entropy_cycle = Entropy1(prob_list[idx2], rate[idx2], current_list[idx2], beta)

        #entropy_cycle, min_current = Entropy2(prob_list[idx2], rate[idx2], current_list[idx2], beta)

        #entropy_cycle, min_current = Entropy2_2(prob_list[idx2], time[idx2], current_list[idx2], beta, rate[idx2])

        # #entropy_cycle = Entropy3(prob_list[idx2], list_len[idx2], beta)

        # if mode is 'discharge':
        #     entropy_cycle, min_current = Entropy4_DC(prob_list[idx2], time[idx2], current_list[idx2], beta, ratedCap)
        # else:
        #     entropy_cycle, min_current = Entropy4_C(prob_list[idx2], time[idx2],
        #                                              current_list[idx2], beta, ratedCap)

        entropy_list.append(entropy_cycle)
        # except:
        #     print("Exception idx : ", idx2)
    entropy_list = MF(entropy_list, 3)

    if mode is 'discharge':
        return entropy_list, prob_list
    elif mode is 'charge':
        return entropy_list, retrun_prob_list

def EntropyForSOHProb_withCurrent_oneNasaBattery_0305(list_A, current_list, beta, mode, time, ratedCap):
    '''
    기존의 EntropyForSOHProb_withCurrent는 전체 battery data를 한번에 처리한다.
    이 Function은 1개의 battery data만 처리한다.
    :param list_A: 전압 정보, [cycle][voltage]
    :param current_list: 전류정보, [cycle][curent]
    :param beta: lenght coefficient
    :param mode: discharge/charge mode
    :return entropy_list: entropy [cycle][entropy]
    :return retrun_prob_list : probability [cycle][probability]
    '''
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''

    prob_list = []
    list_len = []
    retrun_prob_list = []

    for idx2 in range(len(list_A)): #cycle

        list_A[idx2] = Moving_Avg_Filter(list_A[idx2], 10)

        hist, bin_edges = np.histogram(list_A[idx2], bins=np.linspace(2,5,17)) # 16 bins
        list_size = len(list_A[idx2])
        list_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
        hist = hist + np.ones(16)
        prob_list.append(hist/(list_size+16))

        hist, _ = np.histogram(list_A[idx2], bins=np.linspace(2, 5, 7))
        #retrun_prob_list.append(hist/list_size)

    entropy_list = []

    # min_len_value = min(len_list[idx]) # 길이의 최소값

    for idx2 in range(1, len(list_A)-1):
        try:
            #entropy_cycle = Entropy0(prob_list[idx2])

            #entropy_cycle = Entropy1(prob_list[idx2], time[idx2], current_list[idx2], beta)

            entropy_cycle, min_current = Entropy2(prob_list[idx2], time[idx2], current_list[idx2], beta)
            #entropy_cycle, min_current = Entropy2_2(prob_list[idx2], time[idx2], current_list[idx2], beta, rate[idx2])
            # #entropy_cycle = Entropy3(prob_list[idx2], list_len[idx2], beta)

            # if mode is 'discharge':
            #     entropy_cycle, min_current = Entropy4_DC(prob_list[idx2], time[idx2], current_list[idx2], beta, ratedCap)
            # else:
            #     entropy_cycle, min_current = Entropy4_C(prob_list[idx2], time[idx2],
            #                                              current_list[idx2], beta, ratedCap)

            entropy_list.append(entropy_cycle)
        except:
            print("Exception idx : ", idx2)
    entropy_list = MF(entropy_list, 3)

    if mode is 'discharge':
        return entropy_list, prob_list
    elif mode is 'charge':
        return entropy_list, retrun_prob_list

def EntropyForSOHProb_withCurrentCap(list_A, current_list, beta, mode, ratedCap, time):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''
    all_prob = []
    return_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list_A)):
        prob_list = []
        temp_len = []
        retrun_prob_list = []
        for idx2 in range(len(list_A[idx])): #cycle
            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = int(time[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))

            hist, _ = np.histogram(list_A[idx][idx2], bins=np.linspace(2, 5, 7))
            retrun_prob_list.append(hist/list_size)
        len_list.append(temp_len)
        all_prob.append(prob_list)
        return_prob.append(retrun_prob_list)

    all_entropy_list = [] # 모든 배터리에 대한 엔트로피 리스트
    # all_discharge_current_list = [] # 모든 배터리에 대한 방전전류 리스트
    #all_concat_list = []
    for idx in range(len(list_A)):
        entropy_list = []
        min_current_list = []
        concat_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list_A[idx])):
            if mode is 'discharge':
                entropy_cycle, min_current = Entropy4_DC(all_prob[idx][idx2], time[idx][idx2], current_list[idx][idx2], beta, ratedCap)
            else:
                entropy_cycle, min_current = Entropy4_C(all_prob[idx][idx2], time[idx][idx2], current_list[idx][idx2], beta, ratedCap)

            #entropy_cycle, min_current = Entropy4(all_prob[idx][idx2], len_list[idx][idx2], current_list[idx][idx2], beta)

            entropy_list.append(entropy_cycle)
            min_current_list.append(min_current)
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)
        #all_concat_list.append(list(np.concatenate((np.array(entropy_list)[:, np.newaxis], np.array(min_current_list)[:, np.newaxis]), axis=1)))
        # all_discharge_current_list.append(min_current_list)

    if mode is 'discharge':
        return all_entropy_list, return_prob
    elif mode is 'charge':
        return all_entropy_list

def EntropyForSOHProb_withCurrentCap_prev(list_A, current_list, beta, mode, ratedCap):
    '''히스토그램 생성후 pdf 만들어서 엔트로피 구하는 방법'''
    '''entropy4와 함께 사용, 전류를 반영한 엔트로피 인덱스 계산
       return prob는 히스토그램의 bin 수를 엔트로피 계산 시 bin 보다 작게(예를 들어 6)하여 
       각 bin에 해당하는 확률값 리스트이다.'''
    all_prob = []
    return_prob = []
    len_list = [] # 각 데이터의 길이. minimum length 를 구하기위함. (Length Normalization)[][]
    for idx in range(len(list_A)):
        prob_list = []
        temp_len = []
        retrun_prob_list = []
        for idx2 in range(len(list_A[idx])): #cycle
            hist, bin_edges = np.histogram(list_A[idx][idx2], bins=np.linspace(2,5,17)) # 16 bins
            # hist, bin_edges = np.histogram(list[idx][idx2], bins='scott')
            list_size = len(list_A[idx][idx2])
            temp_len.append(list_size) # 각 cycle 마다의 데이터 길이값이 추가된다.
            hist = hist + np.ones(16)
            prob_list.append(hist/(list_size+16))

            hist, _ = np.histogram(list_A[idx][idx2], bins=np.linspace(2, 5, 7))
            retrun_prob_list.append(hist/list_size)
        len_list.append(temp_len)
        all_prob.append(prob_list)
        return_prob.append(retrun_prob_list)

    all_entropy_list = [] # 모든 배터리에 대한 엔트로피 리스트
    # all_discharge_current_list = [] # 모든 배터리에 대한 방전전류 리스트
    #all_concat_list = []
    for idx in range(len(list_A)):
        entropy_list = []
        min_current_list = []
        concat_list = []
        # min_len_value = min(len_list[idx]) # 길이의 최소값
        for idx2 in range(len(list_A[idx])):
            if mode is 'discharge':
                entropy_cycle, min_current = Entropy4_DC(all_prob[idx][idx2], len_list[idx][idx2], current_list[idx][idx2], beta, ratedCap)
            else:
                entropy_cycle, min_current = Entropy4_C(all_prob[idx][idx2], len_list[idx][idx2],
                                                         current_list[idx][idx2], beta, ratedCap)

            #entropy_cycle, min_current = Entropy4(all_prob[idx][idx2], len_list[idx][idx2], current_list[idx][idx2], beta)

            entropy_list.append(entropy_cycle)
            min_current_list.append(min_current)
        entropy_list = MF(entropy_list, 3)
        all_entropy_list.append(entropy_list)
        #all_concat_list.append(list(np.concatenate((np.array(entropy_list)[:, np.newaxis], np.array(min_current_list)[:, np.newaxis]), axis=1)))
        # all_discharge_current_list.append(min_current_list)

    if mode is 'discharge':
        return all_entropy_list, return_prob
    elif mode is 'charge':
        return all_entropy_list



#######################################################################
'''Distribution Entropy구하는 방법'''
def UpperDM(U):
    N = len(U)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _UpperDM(m):
        x = [[U[j] for j in range(i, i+m-1+1)] for i in range(N-m+1)]
        D = [[_maxdist(x[i], x[j]) for j in range(N-m+1)] for i in range(N-m+1)]
        D = np.array(D)

        return D[np.triu_indices(N-m, 1)]

    UD = _UpperDM(10)

    return UD

def DistEN(M, list):
    all_entropy =[]
    for idx in range(len(list)):
        UDM = UpperDM(list[idx])
        hist, bin_edges = np.histogram(UDM, bins=np.linspace(0,2, M+1))
        hist = hist + np.ones(M)
        list_size = len(UDM) + M
        prob = hist/list_size
        all_entropy.append(-1*np.sum(prob*np.log2(prob))/np.log2(M))

    return all_entropy
##########################################################################
def concatenateEntropy(DCEnt, CEnt):
    '''
    :param DCEnt: DisCharge [# of train battery test, Entropy list]
    :param CEnt: charge [# of train battery test, Entropy list]
    :return ConcEntropy: [# of train battery test, ?, 2]

    '''
    ConcEntropy =[]
    for idx in range(len(DCEnt)):
        if len(DCEnt[idx]) == len(CEnt[idx]):
            DC_newaxis = np.array(DCEnt[idx])[:, np.newaxis]
            C_newaxis = np.array(CEnt[idx])[:, np.newaxis]
            ConcEntropy.append(list(np.concatenate((DC_newaxis,C_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            break

    return ConcEntropy

def concatenateEntropy_withdischarge(DCEnt, CEnt):
    '''
    discharge ent와 charge ent를 이어줌.
    :param DCEnt: DisCharge [# of train battery test, [Entropy list, discharge_current]]
    :param CEnt: charge [# of train battery test, Entropy list]
    :return ConcEntropy: [# of train battery test, ?, 2]
    '''
    ConcEntropy =[]
    for idx in range(len(DCEnt)):
        if len(DCEnt[idx]) == len(CEnt[idx]):
            # DC_newaxis = np.array(DCEnt[idx])[:, np.newaxis]
            try:
                np.shape(DCEnt[idx])[1]
                DC_newaxis = DCEnt[idx]
            except IndexError:
                DC_newaxis = np.array(DCEnt[idx])[:, np.newaxis]
            try:
                np.shape(CEnt[idx])[1]
                C_newaxis = CEnt[idx]
            except IndexError:
                C_newaxis = np.array(CEnt[idx])[:, np.newaxis]

            ConcEntropy.append(list(np.concatenate((DC_newaxis,C_newaxis), axis=1)))
        else:
            print("Discharge Ent and Charge Ent length are different")
            break

    return ConcEntropy

def concatenateEntropy_withdischarge2(DCEnt, CEnt):
    '''
    discharge ent와 charge ent를 이어줌.
    :param DCEnt: DisCharge [# of train battery test, [Entropy list, discharge_current]]
    :param CEnt: charge [# of train battery test, Entropy list]
    :return ConcEntropy: [# of train battery test, ?, 2]
    '''
    ConcEntropy =[]
    if len(DCEnt) == len(CEnt):
        # DC_newaxis = np.array(DCEnt[idx])[:, np.newaxis]
        try:
            np.shape(DCEnt)[1]
            DC_newaxis = DCEnt
        except IndexError:
            DC_newaxis = np.array(DCEnt)[:, np.newaxis]
        try:
            np.shape(CEnt)[1]
            C_newaxis = CEnt
        except IndexError:
            C_newaxis = np.array(CEnt)[:, np.newaxis]

        ConcEntropy.append(list(np.concatenate((DC_newaxis,C_newaxis), axis=1)))
    else:
        print("Discharge Ent and Charge Ent length are different")

    return ConcEntropy

if __name__ == "__main__":
    """
    battery data의 entropy를 구해서 npy로 저장한다.
    """

    dir_path_CS2 = "../data/dis_current_constant/CS2_XX_0"
    dir_path_CS2_change = "../data/dis_current_changing"
    dir_path_CX2 = "../data/dis_current_constant/CX2_XX_0"
    dir_path_K2 = "../data/dis_current_constant/K2_XX_0"
    dir_path_B0 = "../data/Nasa_data/BatteryAgingARC_change"

    #battery_list = ['B0005','B0006','B0018']

    #battery_list = ['B0025','B0026','B0028']
    #battery_list = ['B0005','B0006','B0018','B0046', 'B0047','B0048','B0033']

    #battery_list = ['CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    #battery_list = ['CS2_36']
    battery_list = ['CS2_3/total']

    for battery in battery_list:
        if battery.__contains__('CS2_') and battery.__contains__('/'):
            battery_tmp = battery.split('/')
            battery_path = os.path.join(dir_path_CS2_change, battery_tmp[0])
            battery_path = os.path.join(battery_path, battery_tmp[1])
        elif battery.__contains__('CS2_'):
            battery_path = os.path.join(dir_path_CS2, battery)
        elif battery.__contains__('CX2_'):
            battery_path = os.path.join(dir_path_CX2, battery)
        elif battery.__contains__('K2_'):
            battery_path = os.path.join(dir_path_K2, battery)
        elif battery.__contains__('B0'):
            battery_path = os.path.join(dir_path_B0, battery)
        else:
            continue

        DC_list = np.load(battery_path + '/discharge_data.npy', allow_pickle=True)
        DC_list_current = np.load(battery_path + '/discharge_current.npy', allow_pickle=True)
        capacity = np.load(battery_path + '/capacity.npy', allow_pickle=True)
        All_DC_time = np.load(battery_path + '/discharge_time_all.npy', allow_pickle=True)

        print("DC_list shape : ", np.shape(DC_list))
        print("DC_list_current shape : ", np.shape(DC_list_current))
        print("capacity shape : ", np.shape(capacity))

        if battery.__contains__('CS2_'):
            rated_cap = 1.1
        elif battery.__contains__('CX2_'):
            rated_cap = 1.35
        elif battery.__contains__('K2_'):
            rated_cap = 2.6
        elif battery.__contains__('B0'):
            rated_cap = 2
        else:
            rated_cap = 1.1

        SOH = capacity/rated_cap

        filter_num = 5
        # time Filter
        # for j in range(filter_num, len(All_DC_time)):
        #     sum = 0
        #     for k in range(filter_num):
        #         sum = sum + All_DC_time[j - k][-1]
        #     sum = sum / filter_num
        #     All_DC_time[j][-1] = sum
        #
        # np.save(battery_path + '/check_time_all', All_DC_time)

        DC_Entropy, DC_prob, list_A, sum_cur, start_idx, end_idx = EntropyForSOHProb_withCurrent_oneNasaBattery(DC_list,
                                                                                            DC_list_current, 1500,
                                                                                            'discharge', [],
                                                                                            SOH)



        alpha = 1500
        epsilon = 50

        #Entropy Filter
        for j in range(filter_num, len(DC_Entropy)):
            sum = 0
            for k in range(filter_num):
                sum = sum + DC_Entropy[j - k]
            sum = sum / filter_num
            DC_Entropy[j] = sum


        plt.figure()
        plt.plot(DC_Entropy[11:])
        plt.show()

        # for j in range(len(DC_Entropy)):
        #     DC_Entropy[j] = DC_Entropy[j] * (alpha / (All_DC_time[j][-1]))

        #np.save(battery_path + '/discharge_Entropy.npy', DC_Entropy)

        check_curremt_sum = []

        print(np.abs(DC_list_current[1][10]))

        for j in range(len(list_A)):

            try:
                #check_curremt_sum.append(((All_DC_time[j][-1]) * 1.1) * (
                 #           (0.02) * np.exp((1.627) * 1.1) + 0.883))  # / ((-0.062) * np.exp(0.997 * current)+1.174))
                check_curremt_sum.append(((All_DC_time[j][end_idx[j]]-All_DC_time[j][start_idx[j]]) *np.abs(DC_list_current[1][10])))
                # check_curremt_sum.append((All_DC_time[i][j][-1])*(0.021 * np.exp(1.656 * current) + 0.867) * current)
            except:
                print("out of range start_idx : ")

        np.save(battery_path + '/discharge_Current_Sum.npy', check_curremt_sum)

        # for j in range(len(DC_Entropy)):
        #     # DC_Entropy[j] = DC_Entropy[j]*(alpha/(All_DC_time[i][j][-1]*current*(0.072 * np.exp(1.278 * current) + 0.706))) #*(0.428 * np.exp(1.037 * current) - 0.371)
        #     # current_sum.append(All_DC_time[i][j][-1]*current)
        #     # DC_Entropy[j] = DC_Entropy[j] * (alpha / ((All_DC_time[i][j][-1]+epsilon) * current*(0.072 * np.exp(1.278 * current) + 0.706)))
        #
        #     DC_Entropy[j] = DC_Entropy[j] * ((1.178) * np.exp((-0.144) * DC_list_current[1][10]))  # ((1.193) * np.exp(-0.159 * current))  #check_curremt_sum[i] / 1500
        #     # DC_Entropy[j] = DC_Entropy[j]*(alpha/(All_DC_time[i][j][-1]*current*(0.072 * np.exp(1.278 * current) + 0.706)))

        np.save(battery_path + '/discharge_Entropy.npy', DC_Entropy[11:])

