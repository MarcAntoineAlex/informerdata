from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# from numpy import linspace
#
from data.data_loader import Dataset_ETT_hour
import time
import pandas as pd
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy as np
#
#
# ##### CCD
# data = pd.read_csv('/Users/marc-antoine/Documents/S7/物理实验/gch/lgp.txt', sep='; ')
# print(data.columns)
#
# x = data['location'].to_numpy()
# y = data['intensity'].to_numpy()
# for i in range(len(x)):
#     index = x[i].find(',')
#     x[i] = x[i][:index] + '.' + x[i][index+1:]
# for i in range(len(y)):
#     index = y[i].find(',')
#     y[i] = y[i][:index] + '.' + y[i][index+1:]
# x = x.astype(np.float)
# y = y.astype(np.float)-150
#
# print(x[340 + np.argmax(y[340:620])])
# print(x[0 + np.argmax(y[0:380])])
# print(x[1020+ np.argmax(y[1020:1500])])
# print(x[1500 + np.argmax(y[1500:2000])])
# print(x[np.argmax(y)])
#
# for i in range(x.shape[0]):
#     if y.max()/2-10 <= y[i] <= y.max()/2+10:
#         print(x[i])
#
# plt.plot(x[:], y[:])
# plt.xlabel('location/mm')
# plt.ylabel('Radiance')
# plt.ylim(0, 4000)
# plt.grid()
# plt.xlim(0)
# plt.ylim(0)
#
# plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/lgp.jpg')
# plt.show()
# sys.exit()

#### FOURRIER
# # preprocessing
data = pd.read_csv('/Users/marc-antoine/Documents/S7/物理实验/gch/erhao.csv', sep=';')
temps = data['Temps'].to_numpy()[20:]
volt = data['EA1'].rolling(10).mean().to_numpy()[20:]
volt -= volt.mean()
# temps = np.concatenate((temps, np.linspace(80, 90, 10000)))
# volt = np.concatenate((volt, volt[74980:79980]/2))
# volt = np.concatenate((volt, volt[74980:79980]/2))
font2 = {'weight': 'normal',
         'size': 14,
         }
# I(t)
plt.plot(temps, volt+0.6, linewidth=0.5, color='black')
plt.xlabel('T/s', font2)
plt.ylabel('U/V', font2)
plt.grid()
plt.xlim(0, 100)
plt.ylim(0, 1.2)
plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(t).jpg')
plt.show()

# I(Delta)
xnew = temps * 0.558 * 2
xnew -= xnew[np.argmax(volt)]
# xnew = - np.flipud(xnew[:np.argmax(volt)])
# xnew -= xnew[0]
# volt = volt[:np.argmax(volt)]
plt.plot(xnew, volt, linewidth=0.5, color='black')
print("DeltaM", xnew[-1]-xnew[0])

plt.xlabel('{}/{}m'.format(chr(916), chr(956)), font2)
plt.ylabel('U/V', font2)
plt.grid()
plt.xlim(xnew[0], xnew[-1])
plt.ylim(-0.6, 0.6)
plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(delta).jpg')
plt.show()

start = np.argmax(volt)
xnew = xnew[start:]
xnew -= xnew[0]
volt = volt[start:]


after_idct = idct(volt, norm='ortho')
after_x = xnew / xnew[-1] * xnew.shape[0] / xnew[-1] / 2
x = np.reciprocal(after_x[80:500])*1000
y = after_idct[80:500]
m, M = np.argmin(y), np.argmax(y)
x *= 580.5/584.5
print(x[M], y[M])
y[m] += 3
y[m-1] += 3
y = y * 2 * 2**0.5 * xnew[-1] / xnew.shape[0]**0.5
plt.plot(x, y, color='black')
plt.grid()
plt.xlim(200, 1000)
plt.xlabel('{}/nm'.format(chr(955)), font2)
plt.ylabel('U/V', font2)
# plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(lambda).jpg')
plt.show()
for i in range(len(x)-1):
    if (y[i] - y[M]/2)*(y[i+1] - y[M]/2) <= 0:
        print(x[i] + (x[i+1]-x[i])*(y[M]/2-y[i])/(y[i+1]-y[i]))
print(x[3]-x[2])

# I(sigma)
x = np.reciprocal(x) * 1000
plt.plot(x, y, color='black')
plt.grid()
plt.xlim(1, 5)
plt.xlabel('{}*{}m'.format(chr(963), chr(956)), font2)
plt.ylabel('U/V', font2)
plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(sigma).jpg')
plt.show()

# I(lambda_positive)
x = np.reciprocal(x/1000)
x = np.flipud(x[y>0])
y = np.flipud(y[y>0])
y1 = np.zeros(y.shape)
for i in range(y.shape[0]-4):
    y1[i+2] = y[i] + y[i+1] + y[i+2] + y[i+3] + y[i+4] / 5
plt.plot(x, y1, color='black')
plt.grid()
plt.xlim(200, 1000)
plt.xlabel('{}/nm'.format(chr(955)), font2)
plt.ylabel('U/V', font2)
plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(lambda)positive.jpg')
plt.show()






# plt.hist(y, bins=100)
# plt.show()

# plt.plot(x[m-5:m+5], y[m-5:m+5], '*--')
# plt.show()
# x_final, y_final = [], []
# for i in range(len(x)-2):
#     if y[i] <= y[i+1] and y[i+1] >= y[i+2] and y[i+1] > 0:
#         x_final.append(x[i+1])
#         y_final.append(y[i+1])
# plt.plot(x_final, y_final)
# plt.xlabel('{}/nm'.format(chr(955)))
# plt.ylabel('U/V')
# plt.grid()
# plt.xlim(300, 900)
#
# print(x_final[np.argmax(y_final)])
# plt.savefig('/Users/marc-antoine/Documents/bao.jpg')
# plt.show()

#
## speed
# data = pd.read_csv('/Users/marc-antoine/Documents/S7/物理实验/Nouveau dossier/speed.csv', sep=';')
# temps = data['Temps'].to_numpy()
# temps = temps * 535/558
# volt = data['EA1'].to_numpy()
# plt.figure(figsize=(40, 25))
# plt.tick_params(labelsize=70)
# plt.xlim(0, 12)
# font2 = {'weight': 'normal',
#          'size': 92,
#          }
# plt.xlabel('T/s', font2)
# plt.ylabel('U/V', font2)
# plt.grid()
# bwith = 3
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(bwith)
# ax.spines['left'].set_linewidth(bwith)
# ax.spines['top'].set_linewidth(bwith)
# ax.spines['right'].set_linewidth(bwith)
#
# plt.plot(temps, volt, color='black',linewidth=2.5)
# plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/Nouveau dossier/speed.jpg')
# plt.show()
# print(temps.shape)
# print(temps[np.argmax(volt[11700:])+11700])
