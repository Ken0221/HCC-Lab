from function.train_phase import train_phase
from function.test_phase_2 import test_phase, stop_control
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet
import socket
import mne
import time
import numpy as np

import sys
import os

import subprocess
import threading
import cv2
import msvcrt
import psutil

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
    global mode
    mode = '-1'
    #     
    
    def kill_existing_yolo():
      global yolo_process
      if yolo_process is not None:
        print("Killing existing YOLO process...")
        try:
          yolo_process.terminate()
          yolo_process.wait(timeout=3)  # 等待進程結束
        except Exception as e:
          print(f"Error terminating YOLO process: {e}")
        yolo_process = None
    
    def run_script():
        try:
            print("run detect_mid.py")
            cmd = [
                "python", "C:/Users/Ken/yolov5/detect_mid.py",
                "--weights", "yolov5s.pt",
                "--source", "udp://192.168.10.1:11111",
                "--view-img"
            ]
            global yolo_process, mode
            yolo_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in yolo_process.stdout:
                line = line.strip()
                # print("偵測輸出：", line)
                if mode == '2':
                  if "on the left" in line:
                      print("on the left")
                      stop_control("left")  
                  if "on the right" in line:
                      print("on the right")
                      stop_control("right")
                    
        except subprocess.CalledProcessError as e:
            print("執行 detect.py 時發生錯誤：", e)
            
    t = threading.Thread(target=run_script)
    t.start()
          
    def brain_control():
      while True:
          print('='*60)
          print(MSG)
          global mode
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
          
          elif mode == '3':
            print('\n' + ' Capture Mode '.center(60, '='))
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
            
    recvThread = threading.Thread(target=brain_control)
    recvThread.start()

    # def capture():
    #   global mode
    #   while True:
    #     if mode == '3':
    #       if msvcrt.kbhit():  # 有按鍵被按下
    #         key = msvcrt.getch().decode('utf-8').lower()
    #         if key == 'c':
    #           # print("capture")
    #           cap = cv2.VideoCapture("udp://192.168.10.1:11111")
    #           isFrame, frame = cap.read()
    #           if isFrame:
    #             fname="photo.jpg"
    #             cv2.imwrite(fname,frame)
    #             print("capture done")
    #           else:
    #             print("capture failed")
    #           cap.release()
    #           cv2.destroyAllWindows()
    #         time.sleep(0.1)
    # recvThread = threading.Thread(target=capture)
    # recvThread.daemon = True  # 設定為守護線程，主線程結束時自動結束
    # recvThread.start()
    
    def recv2():
      while True:
        if mode == '2':
          while True:
            if msvcrt.kbhit():
              key = msvcrt.getch().decode('utf-8').lower()
              if key == 'q':
                stop_control("land")
                break
        if mode == '3':
          kill_existing_yolo()
          time.sleep(3)
          host = ''
          port = 9002
          locaddr = (host,port) 
          # Create a UDP socket
          sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
          sock.bind(locaddr)
          #please fill UAV IP address
          tello_address1 = ('192.168.10.1', 8889)
          # print("command")
          sock.sendto("command".encode(encoding='utf-8'), tello_address1)
          time.sleep(2)
          # print("streamon")
          sock.sendto("streamon".encode(encoding='utf-8'), tello_address1)
          time.sleep(2)
          cap=cv2.VideoCapture("udp://192.168.10.1:11111")
          # print("cap.isOpened():", cap.isOpened())
          while True:
              isFrame, frame=cap.read()
              if isFrame:
                  cv2.namedWindow("UAV video",0)
                  cv2.resizeWindow("UAV video",640,480)
                  cv2.imshow("UAV video",frame)
              if cv2.waitKey(1) & 0xFF == ord('o'):
                  break
              if msvcrt.kbhit():  # 有按鍵被按下
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'c':
                  # print("capture")
                  # cap = cv2.VideoCapture("udp://192.168.10.1:11111")
                  isFrame, frame = cap.read()
                  if isFrame:
                    fname="photo.jpg"
                    cv2.imwrite(fname,frame)
                    print("capture done")
                  else:
                    print("capture failed")
                  # cap.release()
                  # cv2.destroyAllWindows()
                if key == 'q':
                  stop_control("land")
                time.sleep(0.1)
          # cap.release()
          # cv2.destroyAllWindows()
    recvThread = threading.Thread(target=recv2)
    recvThread.start()