from function.algorithm import cal_attention_baseline,cal_close_eyes_baseline
from function.algorithm import EEG_trial_preprocess_nonlabel,EEG_trial_preprocess_label
from function.display import ExperimentFigures
from function.bar import progress_bar
import numpy as np
import os
'''
  Cue:
    - Attention 取後 8秒 [3秒 rest+5秒 attention]
    - 閉眼 取後 8秒 [3秒 rest+5秒 close_eyes]
'''
class LSL_Process(object):
  def __init__(self,EEG_inlet):
    self.EEG_inlet = EEG_inlet
    self.EEG_buffer = []
  # 拉取 所有 LSL資料
  def pull_sample(self):
    chunk, _ = self.EEG_inlet.pull_chunk(max_samples=99999999)
    return np.asarray(chunk).T
  # 將已分割完的資料 append 進 EEG_buffer，等一個階段結束後輸出
  def add_buffer(self, eeg_data):
    self.EEG_buffer.append(eeg_data)
  # 將 EEG_buffer 資料回傳並清除
  def get_EEG_buffer(self):
    EEG_trial_data = np.stack(self.EEG_buffer)
    self.EEG_buffer = []# 
    return EEG_trial_data
    
def train_phase(PS,EPOCH):
  LSL = LSL_Process(PS.EEG_inlet)
  exp_fig = ExperimentFigures(PS)
  # Attention Stage
  bar = progress_bar(EPOCH,Name="Attention Stage")
  for trial in range(EPOCH):
    exp_fig.fixation(f'Attention Stage Training - Trial {trial+1}', delay=2000)
    exp_fig.countdown()                   # 3秒倒計時
    exp_fig.attention(delay=5000)         # 顯示 5次，一次 1秒
    eeg_data = LSL.pull_sample()          # 拉取 所有 LSL資料
    LSL.add_buffer(eeg_data[:,:int(8*PS.fs)])  # (ch, 8秒)
    bar.up(1)
  # 保存 Attention 階段的腦波 (trial,ch,time)
  PS.attention_data = LSL.get_EEG_buffer()
  # ===========================================================================
  # Close Eyes Stage
  bar = progress_bar(EPOCH, Name="Close Eyes Stage")
  for trial in range(EPOCH):
    exp_fig.open_eyes(f'Close Eyes Stage Training - Trial {trial+1}', delay=2000)
    exp_fig.countdown()
    exp_fig.close_eyes(f'Close Eyes Stage Training - Trial {trial+1}', delay=5000)
    eeg_data = LSL.pull_sample()          # 拉取 所有 LSL資料
    LSL.add_buffer(eeg_data[:,:int(8*PS.fs)])  # (ch, 8秒)
    bar.up(1)
  # 保存 Close Eyes 階段的腦波 (trial,ch,time)
  PS.close_eyes_data = LSL.get_EEG_buffer()
  exp_fig.close_windows()
  # ===========================================================================
  folder_path = 'EEG_data'
  out_data_path = f'EEG_data/{PS.user_name}_data.npz'
  if not os.path.exists('EEG_data'):
    os.makedirs('EEG_data') # 資料夾不存在時，建立資料夾
  # 檢查是否有已經保存的檔案，若有則將新錄的資料加入
  if os.path.exists(out_data_path):
    load_data = np.load(out_data_path)
    PS.attention_data = np.concatenate([load_data['attention_data'],PS.attention_data], axis=0)
    PS.close_eyes_data = np.concatenate([load_data['close_eyes_data'],PS.close_eyes_data], axis=0)
  # ===========================================================================
  # 對資料做訓練 及 找 base line
  # Data preprocess (濾波 + 修復異常通道)
  print('\n' + ' model Training '.center(60, '='))
  pre_attention_data = EEG_trial_preprocess_nonlabel(PS,PS.attention_data)
  pre_close_eyes_data = EEG_trial_preprocess_nonlabel(PS,PS.close_eyes_data)
  # 計算 attention 的 baseline
  cal_attention_baseline(PS, pre_attention_data)
  # 計算 close eyes 的 baseline
  cal_close_eyes_baseline(PS, pre_close_eyes_data)
  # ===========================================================================
  # 保存資料
  np.savez( out_data_path,
            attention_data = PS.attention_data,           # attention 階段 (trials, ch, 8秒)
            close_eyes_data = PS.close_eyes_data,         # close_eyes 階段 (trials, ch, 8秒)
            attention_baseline = PS.attention_baseline,   # attention 的 baseline -> (ch)_dict{theta,alpha,beta}
            attention_th = PS.attention_th,               # attention 的 門檻值 -> float
            close_eyes_baseline = PS.close_eyes_baseline, # close_eyes 的 baseline -> (ch)_dict{theta,alpha,beta}
            close_eyes_th = PS.close_eyes_th              # close_eyes 的門檻值 -> float
            )
  print(f'累計資料筆數:{PS.attention_data.shape[0]}')
