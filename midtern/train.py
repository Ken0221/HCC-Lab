from function.train_phase import train_phase
from function.test_phase import test_phase
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet
import socket
import mne
import time
import numpy as np

import sys
import os
os.chdir(sys.path[0])#將當前環境位置設為當前"檔案位置"

MSG = '''[Brain-Controlled Drone]
  1. Train stage 
  2. Test stage
'''

class ParameterStorage(object):#方便存放回傳參數用
  def __init__(self):
    # 視窗預設尺寸
    self.WIDTH = 1000
    self.HEIGHT = 750
    
    # BCI 參數 
    self.fs = 1000              # 採樣頻率
    self.ch = 8                 # 通道數
    self.EEG_ch_label = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2']
    self.nfft = 1024            # FFT 窗口大小  
    self.bandpass = [0.5, 30]   # 帶通濾波器帶通
    self.vol_th = 2000          # vol 門檻值
    self.ch_th = 0.5            # 異常通道比例，超過該比例直接丟棄該trials
    
    # test phase
    self.ctrl_comd = 0          # 0:上下模式 1:左右模式
    
    # 存放 Train_phase 收到的腦波數據
    self.attention_data = None        # attention 階段 (trials, ch, 8秒) [3秒 rest+5秒 attention]
    self.close_eyes_data = None       # close_eyes 階段 (trials, ch, 8秒) [3秒 rest+5秒 close_eyes]
    
    # mne 用 (處理異常訊號時用)
    mne.set_log_level("CRITICAL")
    self.EEG_mne_info = mne.create_info(ch_names=self.EEG_ch_label, sfreq=self.fs, ch_types='eeg', verbose=None)
    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    self.EEG_mne_info.set_montage(biosemi_montage)
    
    # LSL 用
    EEG_streams = resolve_stream('type', 'EEG')
    EEG_inlet = StreamInlet(EEG_streams[0])
    _ , _ = EEG_inlet.pull_sample()
    self.EEG_inlet = EEG_inlet
    time.sleep(0.1)
    eeg_chunk, _ = self.EEG_inlet.pull_chunk(max_samples=99999999)
    ch,t = (np.asarray(eeg_chunk).T).shape
    if ch == 8 and t>0:
      os.system('cls')
    else:
      print('\033[31;1m There may be a problem with LSL, and the data is not pulled correctly\033[0m')
      sys.exit()


def get_trial_count(msg):
  while True:
    try:
      trial_num = int(input(msg))
      return trial_num
    except:
      print('Please enter an integer')

if __name__ == '__main__':
  PS = ParameterStorage()
  # 
  while True:
    print('='*60)
    print(MSG)
    mode = input('Select mode: ')
    if mode == '1': # 包含訓練模型
      print('\n' + ' Train stage '.center(60, '='))
      PS.user_name = input('Please enter your name: ')
      epoch = get_trial_count('Please enter the number of training trials: ')
      train_phase(PS,epoch) 
      print('\n' + ' done! '.center(60, '='))
      
    elif mode == '2':
      print('\n' + ' Testing Mode '.center(60, '='))
      user_name = input('Please enter your name: ')
      PS.user_name = PS.user_name if user_name=='' else user_name # 省得在輸入一次
      # 避免輸入異常 (未先訓練模型時)
      if not(os.path.exists(f'EEG_data/{PS.user_name}_data.npz')):
        print('Please train first')
        continue
      # 避免輸入異常
      trial_num = get_trial_count('Please enter the number of testing trials: ')
      test_phase(PS, trial_num)
      print('\n' + ' done! '.center(60, '='))
      

'''
                  ______________
         ,===:'.,            `-._
              `:.`---.__         `-._
                `:.     `--.         `.
                  \.        `.         `.
          (,,(,    \.         `.   ____,-`.,
       (,'     `/   \.   ,--.___`.'
   ,  ,'  ,--.  `,   \.;'         `
    `{D, {    \  :    \;
      V,,'    /  /    //
      j;;    /  ,' ,-//.    ,---.      ,
      \;'   /  ,' /  _  \  /  _  \   ,'/
            \   `'  / \  `'  / \  `.' /
             `.___,'   `.__,'   `.__,'  

      🗡️ 神獸：龍神・Draconis Pyrelite
      ☁️ 屬性：異世界放空系・壓力解除・記憶釋放

      ☁️ 奧義：
          def release_pressure(code):
              try:
                  time.sleep(∞)
              except:
                  pass

      🧘‍♂️ 龍語低吟：
      「 不要害怕 commit，
        也不要執著 syntax。
      　你寫的每一行，都在飛翔的路上。」

      🐉「bug 不會殺死你，但焦慮會。」
'''