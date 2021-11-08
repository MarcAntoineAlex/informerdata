from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
a = torch.randn(3, 4)
print(len(a))
# from numpy import linspace
#
# from data.data_loader import Dataset_ETT_hour
# import time
# import pandas as pd
# from scipy.fftpack import dct, idct
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# ##### CCD
# data = pd.read_csv('/Users/marc-antoine/Documents/gch/lgp.txt', sep='; ')
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
# plt.savefig('/Users/marc-antoine/Documents/gch/lgp.jpg')
# plt.show()
# sys.exit()
# #### FOURRIER
# data = pd.read_csv('/Users/marc-antoine/Documents/gch/erhao.csv', sep=';')
# temps = data['Temps'].to_numpy()[20:]
# volt = data['EA1'].rolling(10).mean().to_numpy()[20:]
# volt -= volt.mean()
# from scipy.interpolate import make_interp_spline
# # xnew = np.linspace(temps.min(), temps.max(), 10000) #300 represents number of points to make between T.min and T.max
# # power_smooth = make_interp_spline(temps, volt)(xnew)
#
# # addy = np.random.randn(1500) * 0.03
# # addy += power_smooth[:1500]
# # addx = xnew[:1500] + 80
# # xnew = np.concatenate((xnew, addx))
# # power_smooth = np.concatenate((power_smooth, addy))
# # plt.plot(temps, volt)
# # plt.show()
# xnew = temps * 0.558 * 2
# xnew -= xnew[np.argmax(volt)]
# # xnew = - np.flipud(xnew[:np.argmax(volt)])
# # xnew -= xnew[0]
# # volt = volt[:np.argmax(volt)]
# plt.plot(xnew, volt, linewidth=0.5)
#
# plt.xlabel('{}/{}m'.format(chr(916), chr(956)))
# plt.ylabel('U/V')
# plt.grid()
# plt.savefig('/Users/marc-antoine/Documents/gch/I(delta).jpg')
# plt.show()
#
# def find_center(y):
#     result = np.ones(20000) * 14000
#     for i in range(20000):
#         y1 = y[40000+i:65000+i]
#         y2 = np.flipud(y[15000+i+1:40000+i+1])
#         erro = np.abs(y1-y2).sum()
#         if y[40000+i]>0 or True:
#             result[i] = erro
#     # plt.plot(result)
#     # plt.show()
#     positive = y[40000:60000] > 0
#     return np.argmin(result)
#
#
# # start = find_center(volt) + 40000
# start = np.argmax(volt)
# # print(start)
# # # plt.plot(volt[start-1000:start+1000])
# # # plt.show()
# # # start = start-30 + np.argmax(volt[start-30:start+30])
# # print(start)
# xnew = xnew[start:]
# xnew -= xnew[0]
# volt = volt[start:]
#
# # plt.plot(xnew, power_smooth, linewidth=0.5)
# # plt.grid()
# # plt.xlim(0, 90)
# # plt.xlabel('T/s'.format(chr(916), chr(956)))
# # plt.ylabel('U/V')
# # plt.savefig('/Users/marc-antoine/Documents/delta.jpg')
# # plt.show()
#
#
# after_idct = idct(volt, norm='ortho')
# after_x = xnew / xnew[-1] * xnew.shape[0] / xnew[-1] / 2
# print(xnew.shape)
# x = np.reciprocal(after_x[80:500])*1000
# y = after_idct[80:500]
# m, M = np.argmin(y), np.argmax(y)
# print(x[M])
# y[m] += 3
# y[m-1] += 3
# y = y * 2 * 2**0.5 * xnew[-1] / xnew.shape[0]**0.5
# plt.plot(x, y)
# plt.grid()
# plt.xlim(200, 1000)
# plt.xlabel('{}/nm'.format(chr(955)))
# plt.ylabel('U/V')
# plt.savefig('/Users/marc-antoine/Documents/gch/I(lambda).jpg')
# plt.show()
#
# x = np.reciprocal(x) * 1000
# plt.plot(x, y)
# plt.grid()
# plt.xlim(1, 5)
# plt.xlabel('{}*{}m'.format(chr(963), chr(956)))
# plt.ylabel('U/V')
# plt.savefig('/Users/marc-antoine/Documents/gch/I(sigma).jpg')
# plt.show()
#
# x = np.flipud(x[y>0])
# y = np.flipud(y[y>0])
# y1 = np.zeros(y.shape)
# for i in range(y.shape[0]-4):
#     y1[i+2] = y[i] + y[i+1] + y[i+2] + y[i+3] + y[i+4] / 5
# plt.plot(x, y1)
# plt.grid()
# plt.xlim(200, 1000)
# plt.xlabel('{}/nm'.format(chr(955)))
# plt.ylabel('U/V')
# plt.savefig('/Users/marc-antoine/Documents/gch/I(lambda)positive.jpg')
# plt.show()
#
#
# # plt.hist(y, bins=100)
# # plt.show()
#
# # plt.plot(x[m-5:m+5], y[m-5:m+5], '*--')
# # plt.show()
# # x_final, y_final = [], []
# # for i in range(len(x)-2):
# #     if y[i] <= y[i+1] and y[i+1] >= y[i+2] and y[i+1] > 0:
# #         x_final.append(x[i+1])
# #         y_final.append(y[i+1])
# # plt.plot(x_final, y_final)
# # plt.xlabel('{}/nm'.format(chr(955)))
# # plt.ylabel('U/V')
# # plt.grid()
# # plt.xlim(300, 900)
# #
# # print(x_final[np.argmax(y_final)])
# # plt.savefig('/Users/marc-antoine/Documents/bao.jpg')
# # plt.show()
#
#
# ### speed
# # data = pd.read_csv('/Users/marc-antoine/Downloads/Nouveau dossier/speed.csv', sep=';')
# # temps = data['Temps'].to_numpy()
# # temps = temps * 535/558
# # volt = data['EA1'].to_numpy()
# # plt.figure(figsize=(50, 20))
# # plt.tick_params(labelsize=29)
# # plt.xlim(0, 12)
# # font2 = {'weight': 'normal',
# #          'size': 32,
# #          }
# # plt.xlabel('Time/s', font2)
# # plt.ylabel('Volt/V', font2)
# # plt.grid()
# # plt.plot(temps, volt)
# # plt.savefig('/Users/marc-antoine/Downloads/Nouveau dossier/speed.jpg')
# # plt.show()
# # print(temps.shape)
# # print(temps[np.argmax(volt[11700:])+11700])
