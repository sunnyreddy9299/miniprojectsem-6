# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:56:04 2019

@author: Yadav
"""

import serial
import sys
#import keyboard

try:
    ser = serial.Serial('COM3', baudrate=115200)
    ser.flushInput()
    x=1
    while x<10:
            ser_bytes = ser.readline().decode()
            t=str(ser_bytes)
            t.strip('\r\n')
            #k=t[3:9]
            #m=t[15:20]
            print(t)
            '''
            file=open("open.csv","a")
            file.write((ser_bytes))
            file.close()
            '''
            x=x+1
        
 #       if keyboard.is_pressed('esc'):
 #           break;
    ser.close
except:
    print("Unexpected error:", sys.exc_info()[0])
    print("Unexpected error:", sys.exc_info()[1])