def my_and(x,y):
  w0=1.5
  if(x+y>w0):
    return 1
  else:
    return 0

def my_or(x,y):
  w1=0.5
  if(x+y>w1):
    return 1
  else:
    return 0
  # (work1) ORを分類する関数を完成させよう

def my_nand(x,y):
    if my_and(x,y):
        return 0
    else:
        return 1
  # (work2) NANDを分類する関数を完成させよう

def my_xor(x,y):
    return my_and(my_or(x,y),my_nand(x,y))
  # (work3) my_and(x,y), my_or(x,y), my_nand(x,y)を組み合わせて，XORを分類する関数を完成させよう！

#正しく関数が作成できたか以下で確認しよう！
xo1= my_xor(0,0)
xo2= my_xor(0,1)
xo3= my_xor(1,0)
xo4= my_xor(1,1)
print('0 XOR 0= ',xo1)
print('0 XOR 1= ',xo2)
print('1 XOR 0= ',xo3)
print('1 XOR 1= ',xo4)