import numpy as np
from scipy import signal
from scipy.signal import iirfilter,sosfiltfilt
import mne
# sos IIR 的 filter coeff
def SOS_IIR_coeff(  fre_cutoff = [0.5, 50],   # 截止頻率 (0.5~50 Hz)
                    fs = 1000,                # 採樣頻率
                    pass_type = 'bandpass',   # 帶通模式
                    ftype = 'butter',         # 應用 Butterworth Filter
                    filter_order = 10):       # IIR_filter 階數
                    
  sos = iirfilter(  filter_order,             # IIR_filter 階數
                    Wn = fre_cutoff,          # 截止頻率 (0.5~50 Hz)
                    btype = pass_type,        # 帶通模式
                    analog = False,           # False代表離散訊號模式
                    ftype = ftype,            # 應用 Filter 類型 (Butterworth)
                    output = 'sos',           # 使用 Second-Order Section Filtering
                    fs = fs)                
  return sos

# SOS IIR Filter
def SOS_IIR_filter(filter_coeff, EEG):          
  # input 二維 EEG 訊號 (ch,t)
  return sosfiltfilt(filter_coeff, EEG, axis=1)

# 單次試驗 EEG data 預處理
def EEG_epoch_preprocess(EEG_epoch, PS):# input 二維 EEG 訊號 (ch,t)
  fs = PS.fs                          # 採樣頻率
  Tolerable_bad_ch_num = PS.ch * PS.ch_th # 可容忍異常通道數
  # 濾波
  EEG_epoch_ini = SOS_IIR_filter(PS.filter_coeff , EEG_epoch)
  
  # 檢測不良(判斷是否有電壓非常高|非常低)
  bad_ch, _ = np.where((EEG_epoch_ini < -PS.vol_th) | (PS.vol_th < EEG_epoch_ini))
  if len(np.unique(bad_ch)) > Tolerable_bad_ch_num:
    return None
  
  # 初始化異常通道
  PS.EEG_mne_info['bads'] = []
  for ch_i in range(PS.ch):
    ch_max = np.abs(EEG_epoch_ini[ch_i, :]).max()
    if ch_max > PS.vol_th:
      ch_name = PS.EEG_ch_label[ch_i]
      # print(f'{ch_name} - amp {ch_max}')
      PS.EEG_mne_info['bads'].append(ch_name)
  if len(PS.EEG_mne_info['bads']) > 0:
    # mne 基於附近的好通道，利用插值方法填補 bad_ch 的數據
    EEG_epoch_mne = mne.EvokedArray(EEG_epoch_ini, PS.EEG_mne_info)
    EEG_epoch_mne.interpolate_bads()
    EEG_epoch_ini = EEG_epoch_mne.data

  return EEG_epoch_ini

# 多次試驗的 EEG data 預處理 (含label)
def EEG_trial_preprocess_label(PS,EEG_trial_data,EEG_trial_label):
  X = []  # 預處理完成的
  Y = [] 
  
  # 建立濾波器參數 32ch [0.5,30] fs=1000
  PS.filter_coeff = SOS_IIR_coeff(fre_cutoff = PS.bandpass, fs = PS.fs)
  for EEG_epoch, label_epoch in zip(EEG_trial_data, EEG_trial_label) :
    X_epoch = EEG_epoch_preprocess(EEG_epoch, PS)
    if X_epoch is None: # 異常訊號直接丟了
      continue
    else: # 正常並且已處理完成
      X.append(X_epoch)
      Y.append(label_epoch)
  # list -> numpy.array
  print(f'已移除{len(EEG_trial_data)-len(X)}筆資料')
  return np.asarray(X), np.asarray(Y)

# 多次試驗的 EEG data 預處理 (不含label)
def EEG_trial_preprocess_nonlabel(PS,EEG_trial_data):
  X = []  # 預處理完成的
  
  # 建立濾波器參數 32ch [0.5,30] fs=1000
  PS.filter_coeff = SOS_IIR_coeff(fre_cutoff = PS.bandpass, fs = PS.fs)
  for EEG_epoch in EEG_trial_data :
    X_epoch = EEG_epoch_preprocess(EEG_epoch, PS)
    if X_epoch is None: # 異常訊號直接丟了
      continue
    else: # 正常並且已處理完成
      X.append(X_epoch)
  # list -> numpy.array
  print(f'已移除{len(EEG_trial_data)-len(X)}筆資料')
  return np.asarray(X)
  
# ===========================================================================================================
# 專注力相關 
# ===========================================================================================================
def get_bandpower_db(PS,X_epoch):
  if X_epoch.ndim == 2:
    X_epoch = X_epoch[np.newaxis,:,:] # 新增軸在第一軸
  f, Pxx = signal.welch(  X_epoch ,                     # EEG訊號 (trial,ch,time)
                          fs = PS.fs ,                  # 採樣頻率
                          window = 'hamming',           # 使用 hamming 窗
                          nperseg = PS.fs//2,           # 每段的樣本數 
                          noverlap = PS.fs//4,          # 重疊區間
                          nfft = PS.nfft,               # FFT 計算時的點數，決定了頻譜的解析度
                          axis = 2                      # (trial,ch,time) 沿時間軸處理  
                        )
  bandpower = {}
  bands = {'theta': (4, 8),'alpha': (8, 13),'beta': (13, 30)}
  for band_name, (f_low, f_high) in bands.items():
    idx, = np.where((f >= f_low) & (f <= f_high))
    bp = np.trapz(Pxx[:, :, idx], x=f[idx], axis=2)  # (trial, ch)
    bandpower[band_name] = 20 * np.log10(np.clip(np.abs(bp), 1e-20, 1e100))  # 轉成 dB
  return bandpower
  
# 計算當前專注度 (參考 五角圖 專注力部分的code)
def calc_attention_from_theta_alpha(curr_bandpower, base_bandpower, show_msg=True):
  # 挑選感興趣的通道
  curr_theta = np.mean(curr_bandpower['theta'][[0,1,2]]) # FP1、FP2、FZ
  curr_alpha = np.mean(curr_bandpower['alpha'][[0,1,2]]) # FP1、FP2、FZ
  base_theta = np.mean(base_bandpower['theta'][[0,1,2]]) # FP1、FP2、FZ
  base_alpha = np.mean(base_bandpower['alpha'][[0,1,2]]) # FP1、FP2、FZ
  # 計算 attentionlevel
  diff_attention = (curr_theta + curr_alpha) - (base_theta + base_alpha)
  if show_msg:
    print(f'attention baseline\t: {(base_theta + base_alpha):.2f}')
    print(f'attention bandpower\t: {(curr_theta + curr_alpha):.2f}')
    print(f'attention level\t\t: {diff_attention:.2f}')
  return diff_attention
  
# 計算 attention 的 baseline
def cal_attention_baseline(PS, attention_data):
  fs = PS.fs
  #(trials, ch, 8秒) [3秒 rest+5秒 attention]
  rest = attention_data[:, :, :3*fs]              # (trial, ch, 3秒) 非專注時
  atte = attention_data[:, :, -3*fs:]             # (trial, ch, 3秒) 專注時
  # 計算其 bandpower db
  rest_bandpower = get_bandpower_db(PS, rest)     # (trial, ch)_dict{theta,alpha,beta} 非專注時
  atte_bandpower = get_bandpower_db(PS, atte)     # (trial, ch)_dict{theta,alpha,beta} 專注時
  # 取平均
  rest_bandpower_m = {label: np.mean(data, axis=0) for label, data in rest_bandpower.items()} # np.mean(rest_bandpower['theta'], axis=0)
  atte_bandpower_m = {label: np.mean(data, axis=0) for label, data in atte_bandpower.items()}
  # 計算專注力
  attentionlevel = calc_attention_from_theta_alpha(atte_bandpower_m, rest_bandpower_m) # 計算當前專注度 (參考 五角圖 專注力部分的code)
  # 輸出的 baseline 及 Threshold
  PS.attention_baseline = rest_bandpower_m
  PS.attention_th = attentionlevel

# ===========================================================================================================
# 閉眼相關
# ===========================================================================================================
def calc_close_eyes_from_alpha(curr_bandpower, base_bandpower, show_msg=True):
  # 挑選感興趣的通道
  curr_alpha = np.mean(curr_bandpower['alpha'][[6,7]]) # O1,O2
  base_alpha = np.mean(base_bandpower['alpha'][[6,7]])
  if show_msg:
    print(f'close eyes baseline\t: {curr_alpha:.2f}')
    print(f'close eyes bandpower\t: {base_alpha:.2f}')
    print(f'close eyes level\t: {(curr_alpha-base_alpha):.2f}')
  return (curr_alpha-base_alpha)
  
# 計算 close eyes 的 baseline
def cal_close_eyes_baseline(PS, close_eyes_data):
  fs = PS.fs
  #(trials, ch, 8秒) [3秒 rest+5秒 close_eyes]
  open  = close_eyes_data[:, :, :3*fs]             # (trial, ch, 3秒) 張眼時
  close = close_eyes_data[:, :, -3*fs:]            # (trial, ch, 3秒) 閉眼時
  # 計算其 bandpower db
  open_bandpower  = get_bandpower_db(PS, open)     # (trial, ch)_dict{theta,alpha,beta} 張眼時
  close_bandpower = get_bandpower_db(PS, close)    # (trial, ch)_dict{theta,alpha,beta} 閉眼時
  # 取平均
  open_bandpower_m  = {label: np.mean(data, axis=0) for label, data in open_bandpower.items()}
  close_bandpower_m = {label: np.mean(data, axis=0) for label, data in close_bandpower.items()}
  # 計算 close eyes 的 baseline
  diff_bandpower = calc_close_eyes_from_alpha(close_bandpower_m, open_bandpower_m)
  # 輸出的 baseline 及 Threshold
  PS.close_eyes_baseline = open_bandpower_m
  PS.close_eyes_th = diff_bandpower
