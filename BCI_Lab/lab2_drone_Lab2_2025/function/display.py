import numpy as np
import pygame
from pygame.locals import *
import random
import sys



# 實驗過程中的畫面
class ExperimentFigures:
  def __init__(self, PS):
    pygame.init()   # pygame initialized
    self.WIDTH, self.HEIGHT = PS.WIDTH, PS.HEIGHT
    self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
    pygame.display.set_caption('Mind-Controlled Drone Training')
    self.clock = pygame.time.Clock()
    # 載入實驗所需的 圖片及音檔
    self.done_sound = pygame.mixer.Sound("exp_img/done.mp3")
    self.open_eyes_img = self.load_img('exp_img/open eyes.png') 
    self.close_eyes_img = self.load_img('exp_img/close eyes.png')
    self.left_arrow_img = self.load_img('exp_img/left arrow.png')
    self.right_arrow_img = self.load_img('exp_img/right arrow.png')
  
  # 載入圖片
  def load_img(self, filename):
    image = pygame.image.load(filename)
    image = pygame.transform.scale(image, (400, 300))
    return image
    
  # 顯示圖片
  def show_img(self, img):
    self.window.blit(img, (300, 225))
    
  # 顯示文字
  def draw_text(self, text, size, x, y):
    font_name = pygame.font.match_font('arial')
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    self.window.blit(text_surface, text_rect)
    
  # 用來取代 pygame.time.delay(1000) ，避免畫面卡死
  def safe_delay(self, ms):
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < ms:
      for event in pygame.event.get():
        if event.type == QUIT:
          pygame.quit()
          sys.exit()        # 時間有點趕 一時想不到更好的寫法，就直接結束程式了 
      pygame.time.wait(10)  # 每 10毫秒檢查一次事件

  # 顯示倒計時
  def countdown(self):
    x, y = 450, 150
    for i in range(3):  # 在(450, 150) 顯示大小為 15 的白點
      pygame.draw.circle(self.window, (255, 255, 255), (x + i * 50, y), 15)
      pygame.display.update()
      self.safe_delay(1000)

  # Fixation cue 
  def fixation(self, text, delay=2000):
    self.window.fill((0, 0, 0))
    self.draw_text(text, 50, self.WIDTH // 2, 50)               # size=50, x=500, y=50
    self.draw_text('+', 200, self.WIDTH // 2, self.HEIGHT // 2)
    pygame.display.update()
    self.safe_delay(delay)

  # Attention Stage (Follow Red Ball)
  def attention(self, delay=5000):
    for _ in range(delay//1000):
      self.window.fill((0, 0, 0))
      # 生成 10 個白點和 1 個紅點
      dots = [(random.randint(25, 976), random.randint(25, 726)) for _ in range(11)]
      for dot in dots[:-1]:
        pygame.draw.circle(self.window, (255, 255, 255), dot, 25)
      pygame.draw.circle(self.window, (255, 0, 0), dots[-1], 25)
      pygame.display.update()
      self.safe_delay(1000)
  # open Eyes Stage
  def open_eyes(self, text, delay=2000):
    self.window.fill((0,0,0))
    self.draw_text(text, 50, self.WIDTH // 2, 50)
    self.show_img(self.open_eyes_img) # 顯示張開眼睛
    pygame.display.update()
    self.safe_delay(delay)
  # close Eyes Stage
  def close_eyes(self, text, delay=5000):
    self.window.fill((0,0,0))
    self.draw_text(text, 50, self.WIDTH // 2, 50)
    self.show_img(self.close_eyes_img) # 顯示閉眼
    pygame.display.update()
    self.safe_delay(delay)
    self.done_sound.play()
  # Left & Right Stage
  def lr_stage(self, text, label=0, delay=2000):
    self.window.fill((0,0,0))
    self.draw_text(text, 50, self.WIDTH // 2, 50)
    # label=0 不顯示/ label=1 顯示左 / label=2 顯示右 
    if label == 1: 
      self.show_img(self.left_arrow_img) # 顯示左箭頭
    elif label == 2:
      self.show_img(self.right_arrow_img) # 顯示右箭頭
    pygame.display.update()
    self.safe_delay(delay)
    self.done_sound.play()
  # 關閉視窗
  def close_windows(self):
    pygame.quit() # 關閉視窗
  
class ParameterStorage(object):
  def __init__(self):
    self.WIDTH = 1000   # 視窗預設尺寸
    self.HEIGHT = 750
    self.STAGE = 3      # 3 training stages
    self.EPOCH = 10     # trials per training stage

def test(PS, exp_fig):
  # # Attention Stage
  # print('\n=====Attention Training=====')
  # # (錄製)
  # for trial in range(PS.EPOCH):
    # print(f'Trial {trial+1}')
    # exp_fig.fixation(f'Attention Stage Training - Trial {trial+1}', delay=2000)
    # exp_fig.countdown() # 3秒倒計時
    # exp_fig.attention(delay=5000) # 顯示 5次，一次 1秒
  # # (結束錄製) training_stage += 1
  
  # # Close Eyes Stage
  # print('\n=====Close Eyes Training=====')
  # # (錄製)
  # for trial in range(PS.EPOCH):
    # print(f'Trial {trial+1}')
    # exp_fig.open_eyes(f'Close Eyes Stage Training - Trial {trial+1}', delay=2000)
    # exp_fig.countdown()
    # exp_fig.close_eyes(f'Close Eyes Stage Training - Trial {trial+1}', delay=5000)
  # # (結束錄製) training_stage += 1
  
  # Left & Right Stage
  print('\n=====L/R Training=====')
  # (錄製)
  label_list = [0,1,2]*PS.EPOCH
  for trial in range(PS.EPOCH*3):
    label = label_list[trial] # 刪
    print(f'Trial {trial+1}')
    exp_fig.fixation(f'L/R Stage Training - Trial {trial+1}', delay=3000)
    exp_fig.countdown() # 3秒倒計時
    exp_fig.lr_stage(f'L/R Stage Training - Trial {trial+1}', label=label, delay=2000)
  
  pygame.quit() # 關閉視窗
    
if __name__ == "__main__":
  import os
  os.chdir(sys.path[0])  # 將當前環境位置設為當前檔案位置
  PS = ParameterStorage()
  exp_fig = ExperimentFigures(PS)
  
  test(PS, exp_fig)
