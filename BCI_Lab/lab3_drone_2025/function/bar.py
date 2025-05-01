import time
class progress_bar(object):
  def __init__(self,total,Length=30,Name=None):#初始化進度條
    self.total = total  #總長度
    self.schedule = 0   #紀錄當前進度
    self.Length = 30    #進度條長度
    self.Name = '進度' if Name is None else Name    #進度條名字
    self.symbol = '■' #進度條符號(#、▉、■)
    self.show_progress_bar()
  def show_progress_bar(self):#顯示進度條
    Proportion = (self.schedule/self.total)#計算進度條比例
    Proportion = Proportion if Proportion<=1 else 1
    P = round(Proportion*self.Length)
    N = self.Length-P
    Bar = f"{self.symbol}"*P + " "*N
    percentage = f'{round(Proportion*100,2)}%'
    print(f'{self.Name}',
          f'|{Bar}|',
          f'{percentage} ({self.schedule}/{self.total})',
          end='\r')
    if Proportion>=1:
      print()
  def up(self,num):
    self.schedule+=num
    self.show_progress_bar()

if __name__=='__main__':
  X = 10
  bar = progress_bar(X,Name="attention stage")
  for i in range(X):
    time.sleep(10)
    bar.up(1)