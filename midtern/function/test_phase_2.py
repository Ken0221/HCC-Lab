from function.algorithm import EEG_epoch_preprocess, SOS_IIR_coeff
from function.algorithm import get_bandpower_db
from function.algorithm import calc_close_eyes_from_alpha, calc_attention_from_theta_alpha
from scipy import signal
import numpy as np
import socket
import pygame
import time
import json
import sys
# Set UDP for sending command

try:
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind(('', 9000))
  tello_address = ('192.168.10.1', 8889)
except:
  print('\033[31;1mPlease do not run 2 programs simultaneously\033[0m')
  sys.exit()
'''
  眨眼 控制上下
    determine_updown
  專注 控制前進
    determine_forward
  咬牙 切換模式
    determine_sw
  閉眼 控制降落
    determine_land
'''
color_dict = {
    'r': '31',  # 紅色字
    'g': '32',  # 綠色字
    'b': '34',  # 藍色字
    'y': '33',  # 黃色字
    'lc': '96'  # 淺青色
}
# 彩色字
def C_text(text,color):
  if color in color_dict:
    return f'\033[{color_dict[color]}m{text}\033[0m'
  return text
  
class LSL_Process(object):
  def __init__(self,EEG_inlet):
    self.EEG_inlet = EEG_inlet
  # 拉取 所有 LSL資料
  def pull_sample(self):
    chunk, _ = self.EEG_inlet.pull_chunk(max_samples=99999999)
    return np.asarray(chunk).T

def test_init(PS):
  # 載入 baseline 及 Threshold
  load_data = np.load(f'EEG_data/{PS.user_name}_data.npz',allow_pickle=True)
  PS.attention_baseline = load_data['attention_baseline'].item() # 轉回 dict
  PS.close_eyes_baseline = load_data['close_eyes_baseline'].item() # 轉回 dict
  PS.attention_th = load_data['attention_th']
  PS.close_eyes_th = load_data['close_eyes_th']
  # 初始化 音效
  pygame.init()   # pygame initialized
  PS.sound = pygame.mixer.Sound("exp_img/done.wav")
  PS.sound2 = pygame.mixer.Sound("exp_img/donedone.wav")
  # Drone control
  sock.sendto(("command").encode(encoding='utf-8'), tello_address)
  sock.sendto(("takeoff").encode(encoding='utf-8'), tello_address)
  # Bandpass filter setting
  PS.filter_coeff = SOS_IIR_coeff(fre_cutoff = PS.bandpass, fs = PS.fs)
  # tunable params
  with open('tunable_params.json', 'r', encoding='utf-8') as f:
    try:
      PS.TP = json.load(f)
    except:
      print("tunable_params.json is not in the correct format")

# 眨眼 控制上下 or 左右
def determine_updown(EEG, PS):
  up_threshold = PS.TP["up_threshold"]
  down_threshold = PS.TP["down_threshold"]
  peaks_Fp1, _ = signal.find_peaks(EEG[0, :], prominence=150, width=50)
  peaks_Fp2, _ = signal.find_peaks(EEG[1, :], prominence=150, width=50) 
  thres_ud = np.min([len(peaks_Fp1), len(peaks_Fp2)])

  thres_ud_ct = C_text(f'{thres_ud}','lc')
  comd = {1:'UP',-1:'Down',0:'None'} if PS.ctrl_comd==0 else {1:'Left',-1:'right',0:'None'}
  state = 1 if (thres_ud >= up_threshold) else -1 if (thres_ud >= down_threshold) else 0
  print(f'    updown : Eye Blink:{thres_ud_ct},[{down_threshold}~{up_threshold}]:{comd[state]}')
  return state

# 閉眼 控制降落
def determine_land(EEG, PS):
  # 計算這 3秒腦波的 bandpower -> (1,ch)_dict{theta,alpha,beta}
  curr_bp = get_bandpower_db(PS, EEG) 
  # 去掉第 trial 軸，只保留 ch 軸
  curr_bp_m = {label: data[0,:] for label, data in curr_bp.items()}
  # 計算 diff_bandpower
  diff_bandpower = calc_close_eyes_from_alpha(curr_bp_m,PS.close_eyes_baseline, show_msg=False)
  diff_bandpower_ct = C_text(f'{diff_bandpower:.2f}','lc')
  state = True if diff_bandpower > PS.close_eyes_th*PS.TP["land_params"] else False
  print(f'    land : close eyes level:{diff_bandpower_ct},[{PS.close_eyes_th:.2f}*{PS.TP["land_params"]:.2f}]:{state}')
  return state
# 咬牙 切換模式
def determine_sw(EEG, PS):
  # 計算這 3秒腦波的 bandpower -> (1,ch)_dict{theta,alpha,beta}
  curr_bp = get_bandpower_db(PS, EEG) 
  # 取最高頻的 beta 其 FP1、FP2 通道 做分析
  Grit_teeth_beta = np.mean(curr_bp['beta'][0,:2])
  # base 用專注力的平均值，省的在抓一筆
  base_beta = np.mean(PS.attention_baseline['beta'][:2])
  # 
  Grit_teeth_beta_ct = C_text(f'{Grit_teeth_beta:.2f}','lc')
  state = True if Grit_teeth_beta > base_beta*PS.TP["sw_params"] else False
  print(f'    switch mode : Grit teeth beta power: {Grit_teeth_beta_ct},[{base_beta:.2f}*{PS.TP["sw_params"]:.2f}]:{state}')
  return state

# 專注 控制前進
def determine_forward(EEG, PS):
  # 計算這 3秒腦波的 bandpower -> (1,ch)_dict{theta,alpha,beta}
  curr_bp = get_bandpower_db(PS, EEG) 
  # 去掉第 trial 軸，只保留 ch 軸
  curr_bp_m = {label: data[0,:] for label, data in curr_bp.items()}
  # 計算 diff_bandpower
  attentionlevel = calc_attention_from_theta_alpha(curr_bp_m, PS.attention_baseline, show_msg=False)
  attentionlevel_ct = C_text(f'{attentionlevel:.2f}','lc')
  # 
  state = True if attentionlevel < PS.attention_th*PS.TP["forward_params"] else False
  print(f'    forward : Attention level:{attentionlevel_ct},[{PS.attention_th:.2f}*{PS.TP["forward_params"]:.2f}]:{state}')
  return state

global forward_dist

# 控制指令
def control_command(PS, comm_ud, comm_land, comm_sw, comm_forward):
  command_dict = PS.TP["command"]
  green_text = '\033[32;1m  output : {0}\033[0m'
  if comm_sw:
    print(green_text.format('switch mode'))
    PS.ctrl_comd = 1-PS.ctrl_comd # 1->0 / 0->1
    return None
  if comm_ud == 1 and PS.ctrl_comd == 0:
    print(green_text.format('up'))
    return command_dict['up']    
  if comm_ud == -1 and PS.ctrl_comd ==0:
    print(green_text.format('down'))
    return command_dict['down']
  if comm_ud == 1 and PS.ctrl_comd == 1:
    print(green_text.format('left'))
    return command_dict['left']    
  if comm_ud == -1 and PS.ctrl_comd ==1:
    print(green_text.format('right'))
    return command_dict['right']
  if comm_forward:
    global forward_dist
    forward_dist += 100
    print(green_text.format('forward'))
    return command_dict['forward']
  if comm_land:
    print(green_text.format('land'))
    return command_dict['land']
  print(green_text.format('rest'))
  return None

global IsDetectMode
global Direction

def stop_control(direct):
  global IsDetectMode
  IsDetectMode = True
  global Direction
  Direction = direct
  # print("stop:", IsDetectMode)

def test_phase(PS, trial_num):
  # 初始化
  global IsDetectMode
  global Direction
  global forward_dist
  forward_dist = -180 # 最後向前調整
  IsDetectMode = False
  LSL = LSL_Process(PS.EEG_inlet)
  test_init(PS)
  sock.sendto("cw 0".encode(encoding='utf-8'), tello_address)
  for i in range(trial_num):

    time.sleep(2)
    print('\n' + f' Trial No.{i+1} '.center(60, '=') + '\n')
    if PS.ctrl_comd==0:
      PS.sound.play() # 音效提示 叮一次
    else:
      PS.sound2.play()  # 叮兩次
    time.sleep(3)
    EEG = LSL.pull_sample()
    EEG = EEG_epoch_preprocess(EEG[:,-PS.fs*3:], PS)
    if EEG is None:
      print(C_text('Abnormal brain wave signals','r'))
      continue
    # command function
    print("  state : ")
    comm_ud = determine_updown(EEG, PS)           # 眨眼 控制上下 or 左右
    comm_land = determine_land(EEG, PS)           # 閉眼 控制降落
    comm_sw = determine_sw(EEG, PS)               # 咬牙 切換模式
    comm_forward = determine_forward(EEG, PS)     # 專注 控制前進
    # control
    print("IsDetectMode:", IsDetectMode)
    if not IsDetectMode:
      command = control_command(PS, comm_ud, comm_land, comm_sw, comm_forward)
    else:
      print("stop control!!!")
      command = None
      if Direction == "land":
        break
      if Direction == "left":
        sock.sendto("left 190".encode(encoding='utf-8'), tello_address)
      elif Direction == "right":
        sock.sendto("right 190".encode(encoding='utf-8'), tello_address)
      time.sleep(4)
      sock.sendto(("forward "+str(forward_dist)).encode(encoding='utf-8'), tello_address)
      print(("forward "+str(forward_dist)))
      time.sleep(4)
      sock.sendto("land".encode(encoding='utf-8'), tello_address)
      time.sleep(4)
      break
    if not(command is None):
      sock.sendto(command.encode(encoding='utf-8'), tello_address)
    else:
      sock.sendto("cw 0".encode(encoding='utf-8'), tello_address)
    if comm_land and not(comm_sw) and not(comm_ud) and not(comm_forward):
      break
  pygame.quit()
  sock.sendto(("land").encode(encoding='utf-8'), tello_address)
  # sock.close()