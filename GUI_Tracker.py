# This code uses 3D reconstruction for tracking; thereore, the DLT coeffients are needed for tracking folder 1

#!/usr/bin/env python
import os, sys, math, cv2, argparse, pdb
import Global_Var, Keyboard_Fun, CSV_RW, Kalman_DLT_LQR
import numpy as np
from glob import glob
from copy import deepcopy
from numpy import genfromtxt
from termcolor import colored
import pandas as pd
import matplotlib
if sys.platform == 'linux' or sys.platform == 'linux2':
    matplotlib.use("TkAgg")
else:
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from scipy.optimize import minimize, minimize_scalar
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename

import tempfile, atexit, shutil
from guidata.dataset.datatypes import DataSet, BeginGroup, EndGroup
from guidata.dataset.dataitems import (FloatItem, IntItem, FileOpenItem, DirectoryItem, FloatArrayItem)




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Inputs
parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")

##### SLIC parameters
parser.add_argument("-ss", "--SigSLIC", type=int, help = "Sigma for SLIC, the default is 1; it smooth the image", default = 1)
parser.add_argument("-cs", "--ComSLIC", type=int, help = "Compactness of SLIC, the default is 5; the compactness plays an important role on how the superpixels are made. Dont change it unless you know what it should be", default = 8)
parser.add_argument("-tc", "--Threshold_Changes", type=int, help = "How much the Collision_Threshold will be added after first collision", default = 50)
parser.add_argument("-ws", "--WindowSize", type=int, help = "Window Size Landmarkers", default = 35)
parser.add_argument("-wss", "--WindowSize_Second", type=int, help = "Window Size Landmarkers, the default is 100. This number shows 100 pixels more than tracked markers locations.", default = 100)

##### Help
parser.add_argument('Short Key "q"', nargs = '?', help = "Closes the opened window and continue by frame number has been set previously; if no key has been pushed before, it goes forward by one frame.")
parser.add_argument('Short Key "e"', nargs = '?', help = "It lets you to edit the tracker work when you see a problem; it just edit paws. After clicking on point you need to use 'q' to close the page.")
parser.add_argument('Short Key "w"', nargs = '?', help = "It lets you to edit the tracker work when you see a problem; it edits paw, ear, and tail. After clicking on point you need to use 'q' to close the page.")

parser.add_argument('Short Key "z"', nargs = '?', help = "It demonstartes the results 5 frame by 5")
parser.add_argument('Short Key "x"', nargs = '?', help = "It demonstartes the results 10 frame by 10")
parser.add_argument('Short Key "c"', nargs = '?', help = "It demonstartes the results 25 frame by 25")
parser.add_argument('Short Key "v"', nargs = '?', help = "It demonstartes the results 50 frame by 50")
parser.add_argument('Short Key "b"', nargs = '?', help = "It demonstartes the results 150 frame by 150")
parser.add_argument('Short Key "n"', nargs = '?', help = "It demonstartes the results 250 frame by 250")
parser.add_argument('Short Key "m"', nargs = '?', help = "It does not demonstarte till the final frame!")

parser.add_argument('Short Key "1"', nargs = '?', help = 'It come back to the 1 frames before and then you can use "e" to modify the tracking or other short keys to go forward; if you use come back keys, you need to use other keys to stop going backward')
parser.add_argument('Short Key "2"', nargs = '?', help = 'It come back to the 5 frames before and then you can use "e" to modify the tracking or other short keys to go forward; if you use come back keys, you need to use other keys to stop going backward')
parser.add_argument('Short Key "3"', nargs = '?', help = 'It come back to the 10 frames before and then you can use "e" to modify the tracking or other short keys to go forward')
parser.add_argument('Short Key "4"', nargs = '?', help = 'It come back to the 25 frames before and then you can use "e" to modify the tracking or other short keys to go forward')
parser.add_argument('Short Key "5"', nargs = '?', help = 'It come back to the 50 frames before and then you can use "e" to modify the tracking or other short keys to go forward')
parser.add_argument('Short Key "6"', nargs = '?', help = 'It come back to the 150 frames before and then you can use "e" to modify the tracking or other short keys to go forward')
parser.add_argument('Short Key "7"', nargs = '?', help = 'It come back to the 250 frames before and then you can use "e" to modify the tracking or other short keys to go forward')

parser.add_argument('Short Key "a"', nargs = '?', help = "It pauses Visulization mode to Tracking and you need to enter a number through the termianl command line and pressing enter")
parser.add_argument('Short Key "d"', nargs = '?', help = "It change the mode from the Tracking to Visulization")
parser.add_argument('Short Key "f"', nargs = '?', help = "Whenever it is pushed the data will be saved with the current condition")
parser.add_argument('Short Key "g"', nargs = '?', help = "It lets you to jump to any number of frame in the preious tarcked frames if tracking mode is active or any frame number on Visulization mode")

args = parser.parse_args(); ARGS = vars(args)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### GUI
class TestParameters(DataSet):
    """
    DataSet test
    This is a Graphical User Interface for getting the initial values and genral paths.
    """
    _bg1 = BeginGroup("General Parameters")
    Current_Path = os.getcwd()
    cam1_path = DirectoryItem("cam1_path", Current_Path)
    dlt_path = FileOpenItem("dlt_path", "csv", os.path.join(Current_Path, "DLT.csv"))
    NumSLIC = IntItem("NumSLIC", default=15000, slider=False)                      
    Tracking_Mode = IntItem("Tracking_Mode", default=1, min=0, max=1, slider=False).set_pos(col=1)
    _eg1 = EndGroup("General Parameters")

    _bg2 = BeginGroup("Collision Parameters")
    Collision_Threshold = IntItem("Collision_Threshold", default=50, slider=False)
    Start_Frame_Collision = IntItem("Start_Frame_Collision", default=20, slider=False).set_pos(col=1)
    _eg2 = EndGroup("Collision Parameters")

    _bg3 = BeginGroup("Sub_image Parameters")
    PawXWindowSize = IntItem("PawXWindowSize", default=70, slider=False)
    PawYWindowSize = IntItem("PawYWindowSize", default=40, slider=False).set_pos(col=1)
    _eg3 = EndGroup("Sub_image Parameters")

    _bg4 = BeginGroup("visualization Parameters")
    Demonstration_Flag = IntItem("Demonstration_Flag", default=1, min=0, max=1, slider=False)
    Demonstration_Delay = FloatItem("Demonstration_Delay", default=0.6, slider=False).set_pos(col=1)
    Jump_Frame_Number = IntItem("Jump_Frame_Number", default=-1, slider=False)
    Collison_Demonstartion = IntItem("Collison_Demonstartion", default=0, min=0, max=1, slider=False).set_pos(col=1)
    _eg4 = EndGroup("visualization Parameters")

    _bg5 = BeginGroup("Probabilistic Parameters")
    Weights_F = FloatArrayItem("Weights_F", default=np.array([2, 0, 4, 2, 2, 0, 1, 4]))
    Weights_H = FloatArrayItem("Weights_H", default=np.array([2, 0, 4, 1, 2, 0, 2, 4])).set_pos(col=1)
    _eg5 = EndGroup("Probabilistic Parameters")
    

def Set_Variable(ARGS, GUI_Data):
    ARGS['cam1_path'] = GUI_Data.cam1_path
    ARGS['dlt_path'] = GUI_Data.dlt_path
    ARGS['NumSLIC'] = GUI_Data.NumSLIC         
    ARGS['Tracking_Mode'] = GUI_Data.Tracking_Mode

    ARGS['Collision_Threshold'] = GUI_Data.Collision_Threshold
    ARGS['Start_Frame_Collision'] = GUI_Data.Start_Frame_Collision

    ARGS['PawXWindowSize'] = GUI_Data.PawXWindowSize
    ARGS['PawYWindowSize'] = GUI_Data.PawYWindowSize

    ARGS['Demonstration_Flag'] = GUI_Data.Demonstration_Flag
    ARGS['Demonstration_Delay'] = GUI_Data.Demonstration_Delay
    ARGS['Jump_Frame_Number'] = GUI_Data.Jump_Frame_Number
    ARGS['Collison_Demonstartion'] = GUI_Data.Collison_Demonstartion

    ARGS['Weights_F'] = GUI_Data.Weights_F
    ARGS['Weights_H'] = GUI_Data.Weights_H
    return(ARGS)



###############################################################################
############################################################################### Initializer Tracking
def Object_Finder(Object, Segments, Frame, Frame_H, Frame_G, Frame_S, flag): # This function correlates coordinates with object after slic segmentation
    SegNum_object = Segments[int(Object[1]), int(Object[0])]
    mask = np.zeros(Frame.shape[:2], dtype="uint8")
    mask[Segments == int(SegNum_object)] = 255
    Indices = np.where(np.asarray(mask) > 0)
    OBJECT = np.zeros([2, 1])
    if flag == 'W':
        OBJECT[0] = Indices[0].max()
    else:
        OBJECT[0] = Indices[0].mean()
    OBJECT[1] = Indices[1].mean()
    OBJECT_H = Frame_H[np.where(mask > 0)].mean()
    OBJECT_G = Frame_G[np.where(mask > 0)].mean()
    OBJECT_S = Frame_S[np.where(mask > 0)].mean()
    return(OBJECT, OBJECT_H, OBJECT_G, OBJECT_S)



def Initial_Tracker(self): # This function provides a pop_up window to select the slic segments
    Global_Var.Coords = []; Counter = 0
    Number_Cameras = len(self.List_Switch)
    for label in self.List_Switch:
        Image = getattr(getattr(self, 'FPAW'+label), 'Image')
        Frame = deepcopy(Image)
        Frame_G = getattr(getattr(self, 'FPAW'+label), 'Green')
        Frame_H = getattr(getattr(self, 'FPAW'+label), 'Hue')
        Frame_S = getattr(getattr(self, 'FPAW'+label), 'Sat')

        (Segments, Fused_Frame) = Keyboard_Fun.FSLIC(Frame, Frame, self.NumSLIC, self.ComSLIC, self.SigSLIC, True)
        fig = plt.figure(self.Image_Name);
        fig.canvas.mpl_connect('key_press_event', lambda event: Keyboard_Fun.press(event,plt));
        fig.canvas.mpl_connect('button_press_event', lambda event: Keyboard_Fun.onclick(event,plt))
        ax = fig.add_subplot(1, 1, 1);
        ax.imshow(Fused_Frame);plt.axis("off");
        plt.show()

        for L0, L1 in zip(Global_Var.Coords[1:self.Number_Marker+1], self.Coor_Labels[0+Counter:self.Number_Marker*Number_Cameras+Counter:Number_Cameras]): # First click is just for zooming
            OUTPUT = Object_Finder(L0, Segments, Frame, Frame_H, Frame_G, Frame_S, L1[-1])
            for (L2, L3) in  zip(self.Labels_Channels, OUTPUT): setattr(getattr(self, L1), L2, L3)
        Global_Var.Coords = []; Counter += 1



def Object_Finder_Second(Object, Segments, Frame, Frame_H, Frame_G, Frame_S, Limits): # This function correlates coordinates with object after slic segmentation
    SegNum_object = Segments[int(Object[1]), int(Object[0])]
    mask = np.zeros(Frame.shape[:2], dtype="uint8")
    mask[Segments == int(SegNum_object)] = 255
    Indices = np.where(np.asarray(mask) > 0)
    OBJECT = np.zeros([2, 1])
    OBJECT[0] = Indices[0].mean() + Limits[0]
    OBJECT[1] = Indices[1].mean() + Limits[1]
    OBJECT_H = Frame_H[np.where(mask > 0)].mean()
    OBJECT_G = Frame_G[np.where(mask > 0)].mean()
    OBJECT_S = Frame_S[np.where(mask > 0)].mean()
    return(OBJECT, OBJECT_H, OBJECT_G, OBJECT_S)



def second_Tracker(self): # This function provides a pop_up window to select the slic segments
    Global_Var.Coords = []; Counter = 0
    Number_Cameras = len(self.List_Switch)
    for label in self.List_Switch:
        Minx = Miny = 10000
        Maxx = Maxy = 0
        if label == '':
            for L0 in self.Coor_Labels[0::4]: 
                Miny = int(min(Miny,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Maxy = int(max(Maxy,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Minx = int(min(Minx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
                Maxx = int(max(Maxx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
        elif label == '2':
            for L0 in self.Coor_Labels[1::4]:
                Miny = int(min(Miny,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Maxy = int(max(Maxy,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Minx = int(min(Minx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
                Maxx = int(max(Maxx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
        elif label == '3':
            for L0 in self.Coor_Labels[2::4]:
                Miny = int(min(Miny,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Maxy = int(max(Maxy,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Minx = int(min(Minx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
                Maxx = int(max(Maxx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
        elif label == '4':
            for L0 in self.Coor_Labels[3::4]:
                Miny = int(min(Miny,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Maxy = int(max(Maxy,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]))
                Minx = int(min(Minx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))
                Maxx = int(max(Maxx,getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]))

        
        Image = getattr(getattr(self, 'FPAW'+label), 'Image')
        Frame = deepcopy(Image)
        Frame_G = getattr(getattr(self, 'FPAW'+label), 'Green')
        Frame_H = getattr(getattr(self, 'FPAW'+label), 'Hue')
        Frame_S = getattr(getattr(self, 'FPAW'+label), 'Sat')

        Minx -= self.second_window_size; Miny -= self.second_window_size; Maxx += self.second_window_size; Maxy += self.second_window_size
        if Minx < 0: Minx = 0
        if Miny < 0: Miny = 0
        if Maxx > Image.shape[1]: Maxx = Image.shape[1]
        if Maxy > Image.shape[0]: Maxy = Image.shape[0]

        Frame = Image[Miny:Maxy, Minx:Maxx, :]
        Frame_G = Frame_G[Miny:Maxy, Minx:Maxx]
        Frame_H = Frame_H[Miny:Maxy, Minx:Maxx]
        Frame_S = Frame_S[Miny:Maxy, Minx:Maxx]

        Frame_NSLIC = int(round(self.NumSLIC / ((Image.shape[0] / (Maxy - Miny))*(Image.shape[1] / (Maxx - Minx))), 0))
        (Segments, Fused_Frame) = Keyboard_Fun.FSLIC(Frame, Frame, Frame_NSLIC, self.ComSLIC, self.SigSLIC, True)
        fig = plt.figure(self.Image_Name);
        fig.canvas.mpl_connect('key_press_event', lambda event: Keyboard_Fun.press(event,plt));
        fig.canvas.mpl_connect('button_press_event', lambda event: Keyboard_Fun.onclick(event,plt))
        ax = fig.add_subplot(1, 1, 1);
        ax.imshow(Fused_Frame);plt.axis("off");
        plt.show()

        if Global_Var.Edition_Flag == 1 and Global_Var.Just_Paws_Init == 1:
            for L0, L1 in zip(Global_Var.Coords, self.Coor_Labels[4+Counter::Number_Cameras]):
                OUTPUT = Object_Finder_Second(L0, Segments, Frame, Frame_H, Frame_G, Frame_S, [Miny, Minx])
                for (L2, L3) in  zip(self.Labels_Channels, OUTPUT): setattr(getattr(self, L1), L2, L3)
        else:
            for L0, L1 in zip(Global_Var.Coords, self.Coor_Labels[0+Counter::Number_Cameras]):
                OUTPUT = Object_Finder_Second(L0, Segments, Frame, Frame_H, Frame_G, Frame_S, [Miny, Minx])
                for (L2, L3) in  zip(self.Labels_Channels, OUTPUT): setattr(getattr(self, L1), L2, L3)
        Global_Var.Coords = []; Counter += 1



###############################################################################
############################################################################### Marker Tracker
def Limits_Maker_Top(PastPos, WindowSize, Size):
    if PastPos + WindowSize + 20 > 0:
        if PastPos + WindowSize + 20 < Size:
            Limits = PastPos + WindowSize + 20
        else:
            Limits = Size
    else:
        Limits = 0
    return (Limits)



def Limits_Maker_Down(PastPos, WindowSize, Size):
    if PastPos - WindowSize - 20 > 0:
        if PastPos - WindowSize - 20 < Size:
            Limits = PastPos - WindowSize - 20
        else:
            Limits = Size
    else:
        Limits = 0
    return (Limits)




###############################################################################
############################################################################### Paw Tracking functions
def Paw_Subimage(self, Im, Prediceted_Position, GREEN, Sat, Hue):
    Ob_Limits = (int(min(Im.shape[1], Prediceted_Position[self.counter-1, 0] + self.PawXWindowSize)), int(max(0, Prediceted_Position[self.counter-1, 0] - self.PawXWindowSize)),
        int(min(Im.shape[0], Prediceted_Position[self.counter-1, 1] + self.PawYWindowSize)), int(max(0, Prediceted_Position[self.counter-1, 1] - self.PawYWindowSize)))
    
    Ob_image = Im[Ob_Limits[3]:Ob_Limits[2], Ob_Limits[1]:Ob_Limits[0]]
    Ob_NSLIC = int(round(1.2* self.NumSLIC / ((Im.shape[0] / (Ob_Limits[2] - Ob_Limits[3])) *
        (Im.shape[1] / (Ob_Limits[0] - Ob_Limits[1])) - 3), 0))
    Ob_Segments = Keyboard_Fun.FSLIC(Ob_image, Ob_image, Ob_NSLIC, self.ComSLIC, self.SigSLIC, False)
    Ob_Gray = GREEN[Ob_Limits[3]:Ob_Limits[2], Ob_Limits[1]:Ob_Limits[0]]
    Ob_Sat = Sat[Ob_Limits[3]:Ob_Limits[2], Ob_Limits[1]:Ob_Limits[0]]
    Ob_Hue = Hue[Ob_Limits[3]:Ob_Limits[2], Ob_Limits[1]:Ob_Limits[0]]
    return(Ob_Limits, Ob_image, Ob_NSLIC, Ob_Segments, Ob_Gray, Ob_Sat, Ob_Hue)



def Paw_Features(self, Ob_Segments, Ob_Hue, Ob_Gray, Ob_Sat, Ob_Limits, OB, OB_S, OB_S0, OB_H, OB_H0, OB_G, OB_G0, Collision_Flag, Label):
    Loop = range(0, Ob_Segments.max())
    size = Ob_Segments.shape
    Ob = np.zeros([Ob_Segments.max(), 14])
    weight = np.zeros([1, Ob_Segments.max()])

    
    for counter in Loop:
        Positions = np.where(Ob_Segments == counter)
        Ob[counter][13] = Positions[0].max() # y image coordinates
        Ob[counter][12] = Positions[1].mean() # x image coordinates
        Ob[counter][11] = Ob_Hue[np.where(Ob_Segments == counter)].mean()
        Ob[counter][10] = Ob_Gray[np.where(Ob_Segments == counter)].mean()
        Ob[counter][9] = Ob_Sat[np.where(Ob_Segments == counter)].mean()
        
        Ob[counter][7] = math.sqrt(abs(Ob[counter][13] - size[0]/2) ** 2 + abs(Ob[counter][12] - size[1]/2) ** 2)
        Ob[counter][6] = math.sqrt(abs(Ob[counter][13] - size[0]) ** 2 + abs(Ob[counter][12]) ** 2)
        Ob[counter][8] = Ob[counter][11] - size[1]/2
        Ob[counter][5] = abs(Ob[counter][9] - OB_S)
        Ob[counter][4] = abs(Ob[counter][9] - OB_S0)
        Ob[counter][3] = abs(Ob[counter][11] - OB_H)
        Ob[counter][2] = abs(Ob[counter][11] - OB_H0)
        Ob[counter][1] = abs(Ob[counter][10] - OB_G)
        Ob[counter][0] = abs(Ob[counter][10] - OB_G0)
       
        Ob[counter][13] = Ob[counter][13] + Ob_Limits[3]
        Ob[counter][12] = Ob[counter][12] + Ob_Limits[1]

        
    temp_Ob = deepcopy(Ob)

    for counter in range(0,8):
        Ob[...,counter] = (Ob[...,counter] - min(Ob[...,counter]))/(max(Ob[...,counter]) - min(Ob[...,counter]))
    Ob[...,:8] = 1-Ob[...,:8]


    if Collision_Flag == 0:
        if Label[0] == 'F':
            factors = self.Weights_F
        else:
            factors = self.Weights_H
        #                  [G0,G,H0, H,S0, S,D1,D2]
    else:
        if Label[0] == 'F':
            self.Weights_F[7] -= 1; self.Weights_F[3] -= 1
        else:
            self.Weights_H[7] -= 2

    weight = np.sum(Ob[...,:8]*factors, axis = 1)

    Weight = weight.transpose()
    positions = np.where(Weight == Weight.max())
    OB[0] = Ob[positions[0][0]][13]
    OB[1] = Ob[positions[0][0]][12]
    OB_H = Ob[positions[0][0]][3]
    OB_G = Ob[positions[0][0]][1]
    OB_S = Ob[positions[0][0]][5]


    # factors = np.array([1, 0, 2, 0, 1, 0, 0, 0, 0])
    # weight = np.sum(Ob[...,:9]*factors, axis = 1)
    weight = np.sum(Ob[...,:8]*factors, axis = 1)
    Index_Last_Five = weight.argsort()[-5:][::-1]
    Best_Segment_list = []
    for L0 in Index_Last_Five:
        Features = [temp_Ob[L0, 13], temp_Ob[L0, 12], temp_Ob[L0, 11], temp_Ob[L0, 10], temp_Ob[L0, 9], OB_H0, OB_G0, OB_S0, self.counter]
        #          [Ycoor, Xcoor, Hue, Gray, Sat, Hue0, Gray0, Sat0, Frame]
        Best_Segment_list.append(Features)


    return(OB, OB_H, OB_G, OB_S, Best_Segment_list)



def Circular_Correlation(X,Y):
    if X.shape[0] > Y.shape[0]:
        temp = deepcopy(Y)
        Y = np.zeros([X.shape[0], ])
        Y[:temp.shape[0]] = temp

    elif Y.shape[0] > X.shape[0]:
        temp = deepcopy(X)
        X = np.zeros([Y.shape[0], ])
        X[:temp.shape[0]] = temp
        
    Z = np.zeros([X.shape[0], ])
    for n, xx in enumerate(X):
        Z[n] = sum(X*np.roll(Y,n))

    return(Z, X, Y)



def Minimize_Cost_Function(x, args):
    Spline_Function = args['SF']
    Signal_Tracked = args['ST']
    Frequency = x[0]
    Amplitude_Gain = x[1]
    # Amplitude_Gain = 1

    Spline_x = Spline_Function(np.linspace(0, 50, int(50*Frequency))) * Amplitude_Gain
    Spline_x -= (min(Spline_x)+max(Spline_x))/2

    (Correlation, Signal_Tracked, Spline_x) = Circular_Correlation(Signal_Tracked, Spline_x)
    Needed_Rolling = np.argmax(Correlation) # this is what needed for catching the signal
    Spline_x = np.roll(Spline_x, Needed_Rolling)
    Spline_x[np.where(Signal_Tracked==0)] = 0
    
    return sum(abs(Spline_x - Signal_Tracked))



def Running_Optimizer(Signal, Spline_f, Spline):
    b, a = signal.butter(4, 0.075, 'low')
    Signal_Tracked = signal.filtfilt(b, a, Signal)
    Signal_Tracked -= (min(Signal_Tracked)+max(Signal_Tracked))/2
    Length_Signal = len(Signal_Tracked)

    additional = {'SF': Spline_f, 'ST': Signal_Tracked}
    Res = minimize(Minimize_Cost_Function, x0=np.array([1.5, 1]), args=additional, method="Nelder-Mead")

    temp_Spline = Spline - (min(Spline)+max(Spline))/2
    Spline = Spline_f(np.linspace(0, 50, int(50*Res.x[0]))) * Res.x[1]
    Length_Spline = len(Spline)

    SP = Spline - (min(Spline)+max(Spline))/2
    Spline -= (min(Spline)+max(Spline))/2

    Correlation, _, _ = Circular_Correlation(Signal_Tracked, Spline)
    Needed_Rolling = np.argmax(Correlation) # this is what needed for catching the signal
    SP1 = np.roll(SP, (Needed_Rolling)%Length_Spline)
    Spline_1 = np.append(SP1, SP1)
    Spline_d = np.diff(Spline_1)

    return (Spline_d, Spline_1, Length_Signal, temp_Spline, SP, Signal_Tracked, Res, Needed_Rolling)




def Fixing_Y(Signal, Spline_f, Spline, Res, Needed_Rolling):
    b, a = signal.butter(4, 0.075, 'low')
    Signal_Tracked = signal.filtfilt(b, a, Signal)
    Signal_Tracked -= (min(Signal_Tracked)+max(Signal_Tracked))/2
    Length_Signal = len(Signal_Tracked)

    temp_Spline = Spline - (min(Spline)+max(Spline))/2
    Spline = Spline_f(np.linspace(0, 50, int(50*Res.x[0])))
    Length_Spline = len(Spline)

    SP = Spline - (min(Spline)+max(Spline))/2
    Spline -= (min(Spline)+max(Spline))/2

    SP1 = np.roll(SP, (Needed_Rolling)%Length_Spline)
    Spline_1 = np.append(SP1, SP1)
    Spline_d = np.diff(Spline_1)

    return (Spline_d, Spline_1, Length_Signal, temp_Spline, SP, Signal_Tracked)




def Tracker_Paw(self, OBJ, Label, Collision_Flag, On_Each_Other_Flag):
    OB = OBJ.Tr; OB_H = OBJ.H; OB_G = OBJ.G; OB_S = OBJ.S
    OB_H0 = OBJ.H0; OB_G0 = OBJ.G0; OB_S0 = OBJ.S0
    IMAGE = OBJ.Image; Green = OBJ.Green; Hue = OBJ.Hue; Sat = OBJ.Sat

    Prediceted_Position = OBJ.DLT_Inv_Predict_Pos2D

    if Collision_Flag == 1:
        Step = 2
        Temp_Prediceted_Position = deepcopy(Prediceted_Position)
        if OBJ.First_Collision_Flag == 1:
            Range_Spline = min(50, self.counter-2)

            Signal_x = OBJ.Tr_All[self.counter-Range_Spline:self.counter-1,1]
            (OBJ.Spline_xd, OBJ.Spline_x1, Length_Signal_x, temp_Spline_x, SPx, Signal_Tracked_x, Res_x, Needed_Rolling_x) = Running_Optimizer(Signal_x, OBJ.Spline_xf, OBJ.Spline_x)

            # This is just for collision
            Peaks_x = np.append(signal.argrelextrema(OBJ.Spline_x1,np.greater), signal.argrelextrema(OBJ.Spline_x1,np.less))
            if (Peaks_x>Length_Signal_x).any():
                Peaks_x = Peaks_x[Peaks_x > Length_Signal_x]
            Peaks_x = Peaks_x[np.argmin(Peaks_x)]

            Signal_y = OBJ.Tr_All[self.counter-Range_Spline:self.counter-1,0]

            (OBJ.Spline_yd, OBJ.Spline_y1, Length_Signal_y, temp_Spline_y, SPy, Signal_Tracked_y) = Fixing_Y(Signal_y, OBJ.Spline_yf, OBJ.Spline_y, Res_x, Needed_Rolling_x)
            Peaks_y = Peaks_x


            setattr(getattr(self, Label), 'Frequency', Res_x.x[0])

            setattr(getattr(self, Label), 'Spline_x', OBJ.Spline_x1)
            setattr(getattr(self, Label), 'Spline_y', OBJ.Spline_y1)
            setattr(getattr(self, Label), 'Spline_xd', OBJ.Spline_xd)
            setattr(getattr(self, Label), 'Spline_yd', OBJ.Spline_yd)

            setattr(getattr(self, Label), 'Starting_Collision_Frame_x', Length_Signal_x)
            setattr(getattr(self, Label), 'Starting_Collision_Frame_y', Length_Signal_y)
            setattr(getattr(self, Label), 'Counter_Collision_Frame', 0)

            if On_Each_Other_Flag == 1:
                setattr(getattr(self, Label), 'Starting_Predicted_Location', np.roll(OBJ.Tr_All[self.counter-2,:],1))
            else:
                setattr(getattr(self, Label), 'Starting_Predicted_Location', np.roll(OBJ.Tr_All[self.counter-1,:],1))

            setattr(getattr(self, Label), 'Peak_x', Peaks_x)
            setattr(getattr(self, Label), 'Peak_y', Peaks_y)




        Step = int(np.round(Step * getattr(getattr(self, Label), 'Frequency')))
        Starting_Predicted_Location = getattr(getattr(self, Label), 'Starting_Predicted_Location')

        Counter_Collision = getattr(getattr(self, Label), 'Counter_Collision_Frame') 
        Start_x = getattr(getattr(self, Label), 'Starting_Collision_Frame_x')
        Counter_Collision += Step
        Prediceted_Position[self.counter-1, 0] = Starting_Predicted_Location[0] + sum(OBJ.Spline_xd[Start_x-1 : Start_x+Counter_Collision])
        

        Start_y = getattr(getattr(self, Label), 'Starting_Collision_Frame_x')
        Prediceted_Position[self.counter-1, 1] = Starting_Predicted_Location[1] + sum(OBJ.Spline_yd[Start_y-1 : Start_y+Counter_Collision])

        
        Prediceted_Position[self.counter-1, 1] = Prediceted_Position[self.counter-1, 1]
        Prediceted_Position[self.counter-1, 0] = Prediceted_Position[self.counter-1, 0]
        
        setattr(getattr(self, Label), 'Counter_Collision_Frame', Counter_Collision)


        if self.Collison_Demonstartion == 1:
            fontweight = 'bold'
            fontsize = 12

            Counter_Collision = getattr(getattr(self, Label), 'Counter_Collision_Frame') 
            II = 15; XX = []; YY = []
            for ii in range(II): 
                XX = np.append(XX, Starting_Predicted_Location[0] + sum(OBJ.Spline_xd[Start_x-1 : Start_x+Counter_Collision]))
                Counter_Collision += Step

            Counter_Collision = getattr(getattr(self, Label), 'Counter_Collision_Frame')
            for ii in range(II): 
                YY = np.append(YY, Starting_Predicted_Location[1] + sum(OBJ.Spline_yd[Start_y-1 : Start_y+Counter_Collision]))
                Counter_Collision += Step
            fig, ax = plt.subplots(2,1); ax[0].imshow(IMAGE, extent=[0, 2048, 0, 700]); ax[0].plot(XX[:II], 700-YY[:II], '*', linewidth=3, color='w'); 
            # ax[0].plot(XX[:II], 700-YY[:II], '*', linewidth=3, color='firebrick'); 
            ax[1].plot(Signal_Tracked_x, 'b', label="Tracked U Points"); ax[1].plot(Signal_Tracked_y, 'xkcd:neon blue', label="Tracked V Points");
            ax[1].plot(temp_Spline_x, 'g', label="Spline U Before Fitting"); ax[1].plot(temp_Spline_y, 'lime', label="Spline V Before Fitting");
            ax[1].plot(SPx, 'r', label="Spline U After Fitting"); ax[1].plot(SPy, 'xkcd:red pink', label="Spline V After Fitting"); 
            ax[1].plot(OBJ.Spline_x1, 'xkcd:mahogany', label="Spline U After Fitting and Shifting"); ax[1].plot(OBJ.Spline_y1, 'xkcd:medium brown', label="Spline V After Fitting and Shifting"); 
            box = ax[1].get_position(); ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height]); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=dict(weight=fontweight, size= fontsize+2))
            Counter_Collision = Counter_Collision

            ax[1].set_xlabel('Time Points', fontweight = fontweight, fontsize = fontsize+2)
            ax[1].set_ylabel('Displacement (Pixels)', fontweight = fontweight, fontsize = fontsize+2)
            for tick in ax[1].xaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
                tick.label1.set_fontweight(fontweight)
            for tick in ax[1].yaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
                tick.label1.set_fontweight(fontweight)

            plt.show()


    (Object_Limits, Object_image, Object_NSLIC, Object_Segments, Object_Gray, Object_Sat, Object_Hue) = Paw_Subimage(self, IMAGE, Prediceted_Position, Green, Sat, Hue)

    (OB, OB_H, OB_G, OB_S, OB_Segment_list) = Paw_Features(self, Object_Segments, Object_Hue, Object_Gray, Object_Sat, Object_Limits, OB, OB_S, OB_S0, OB_H, OB_H0, OB_G, OB_G0, Collision_Flag, Label)

    return(OB, OB_H, OB_G, OB_S, OB_Segment_list)



###############################################################################
###############################################################################
############################################################################### Otherside Checking
def Find_Sign_Otherside(self, Coord_3D, Cameras, Transferred_Cam, Tracked_Object):
    Current_Coordinates_2D_Other_Side = Kalman_DLT_LQR.DLT_Inverse(self, Coord_3D[0:3,0], self.Coef, Cameras)[2*Transferred_Cam:2*Transferred_Cam+2]
    Current_Coordinates_2D_Other_Side = np.roll(Current_Coordinates_2D_Other_Side, 1)
    Past_Coordinates_2D_Other_Side = Kalman_DLT_LQR.DLT_Inverse(self, Coord_3D[6:9,0], self.Coef, Cameras)[2*Transferred_Cam:2*Transferred_Cam+2]
    Past_Coordinates_2D_Other_Side = np.roll(Past_Coordinates_2D_Other_Side, 1)

    if np.sqrt((Tracked_Object[0]-Current_Coordinates_2D_Other_Side[0])**2+(Tracked_Object[1]-Current_Coordinates_2D_Other_Side[1])**2) < 20:
        Sign = -np.sign(Current_Coordinates_2D_Other_Side[1] - Past_Coordinates_2D_Other_Side[1])
    else:
        Sign = 2
    return(Sign)



def Coordinates_3D_Updater(self, OBJECT, OBJECT2, OBJECT3D, Cameras):
    Temp = Kalman_DLT_LQR.DLT(self, np.append(np.array([float(OBJECT[1]),float(OBJECT[0])]), np.array([float(OBJECT2[1]),float(OBJECT2[0])])), self.Coef, Cameras)
    temp = deepcopy(OBJECT3D)
    temp[0:3,0] = Temp[...,0]
    temp[3:6,0] = temp[0:3,0] - temp[6:9,0]
    return(temp)



def Reassign_Best(New_Object, Best_Objects, Last_Tracked_Object, Flag):
    # Normal colllision of other side
    if Flag ==1:
        Sign = -1 # it should go forward
        for Num, Object in enumerate(Best_Objects):
            if Sign == np.sign(Object[1] - Last_Tracked_Object[1]):
                New_Object = Object[:2]
                break

    # Otherside collision after same side
    elif Flag == 0:
        for Num, Object in enumerate(Best_Objects[1:]):
            if Object[0] < Last_Tracked_Object[0] and Object[0] > New_Object[0]:
                New_Object = Object[:2]
                break


    return(New_Object)



def Checking_Otherside_Collision(self):
    for L0 in self.Coor_Labels: 
        Label_Paw = L0[:L0.find('W')+1]
        Transferred_Cam = 0

        if L0[-1] == '3' or L0[-1] == '4':
            Cameras = self.Cameras[1]
            Other_L0 = ['', '2']
            Other_Cameras = self.Cameras[0]
            if L0[-1] == '4': Transferred_Cam = 1
        else:
            Cameras = self.Cameras[0]
            Other_L0 = ['3', '4']
            Other_Cameras = self.Cameras[1]
            if L0[-1] == '2': Transferred_Cam = 1

        
        Best_Objects = np.array(getattr(getattr(self, L0), 'Best_SPs'))
        Tracked_Object = getattr(getattr(self, L0), 'Tr_All')[self.counter,:]
        Last_Tracked_Object= getattr(getattr(self, L0), 'Tr_All')[self.counter-1,:]

        Object_3D_Other_Side = Coordinates_3D_Updater(self, getattr(getattr(self, Label_Paw+Other_L0[0]), 'Tr'), getattr(getattr(self, Label_Paw+Other_L0[1]), 'Tr'), getattr(getattr(self, Label_Paw+Other_L0[0]), 'Coord_3D'), Other_Cameras)

        Sign = Find_Sign_Otherside(self, Object_3D_Other_Side, Cameras, Transferred_Cam, Tracked_Object)

        # if the previous frame had collision and the current frame has the collision of other side
        if getattr(getattr(self, L0), 'Temp_Collision_Flag') == 1:
            OUTPUT = Reassign_Best(Tracked_Object, Best_Objects, Last_Tracked_Object, 0)

        # if they are close and moving toward each other
        elif Sign != 2 and Sign != 0: 
            OUTPUT = Reassign_Best(Tracked_Object, Best_Objects, Last_Tracked_Object, 1)

            getattr(getattr(self, L0), 'Tr_All')[self.counter, 0:2] = OUTPUT
            getattr(getattr(self, L0), 'Tr')[0] = OUTPUT[0]
            getattr(getattr(self, L0), 'Tr')[1] = OUTPUT[1]         
    return(self)



###############################################################################
###############################################################################
############################################################################### Subfunctions to simplify the function "Running_Tracker"
def First_Time_Initilization_Variables(self, Flag):
    for (Object_Name, L0) in zip(self.Coor_Labels, self.Switching): 
        setattr(self, Object_Name, Objects(getattr(self, 'Image'+L0)))
    
    Kalman_DLT_LQR.Kalman_Spline(self)
    
    for L0 in self.Coor_Labels[0::2]: 
        setattr(getattr(self, L0), 'Kalman_Predict_Pos3D', [])
        setattr(getattr(self, L0), 'Coord_3D', np.zeros([12,1]))

    for L0 in self.Coor_Labels:   
        setattr(getattr(self, L0), 'DLT_Inv_Predict_Pos2D', np.zeros([min(len(self.Image_Bank), len(self.Image_Bank2)), 2]))
        if Flag == 0:
            setattr(getattr(self, L0), 'Tr_All', np.zeros([min(len(self.Image_Bank), len(self.Image_Bank2)), 2]))

        for L1 in self.Labels_Channels[1:]:
            setattr(getattr(self, L0), L1, 0)
            setattr(getattr(self, L0), L1+'_All', np.zeros([min(len(self.Image_Bank), len(self.Image_Bank2)), 1]))

    for S0, L0 in zip(self.spline_Labels, self.Coor_Labels):
        setattr(getattr(self, L0), 'Collision_Flag', 0)
        setattr(getattr(self, L0), 'Collision_Threshold', self.Collision_Threshold)
        setattr(getattr(self, L0), 'Spline_x', self.xspline[S0])
        setattr(getattr(self, L0), 'Spline_y', self.yspline[S0])
        setattr(getattr(self, L0), 'Spline_xd', self.xspline_diff[S0])
        setattr(getattr(self, L0), 'Spline_yd', self.yspline_diff[S0])
        setattr(getattr(self, L0), 'Spline_xf', self.xspline_function[S0])
        setattr(getattr(self, L0), 'Spline_yf', self.yspline_function[S0])

    if Flag ==1:
        for label in self.Coor_Labels: # self.M.H0/self.TAIL.G0/...
            for (l0, l1) in zip(self.Labels_Channels0, self.Labels_Channels[1:]): setattr(getattr(self, label), l0, 0)

        for L0 in self.Coor_Labels: # update self.M.H, ... after come back to previous frames
            setattr(getattr(self, L0), 'Tr', getattr(getattr(TEMP_OBJECT, L0), 'Tr_All')[self.counter, 0:2].transpose())
            for L1 in self.Labels_Channels[1:]:
                setattr(getattr(self, L0), L1 , 0)
    return(self)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running tracker
def Running_Tracker(self):
    if self.counter == 0: # First time initialization of important parameters
        self = First_Time_Initilization_Variables(self, 0)

    else: # Updating class images
        if self.temp_counter == 1:
            TEMP_OBJECT = self
            Kalman_DLT_LQR.Kalman_Spline(self)

        for (Object_Name, L0) in zip(self.Coor_Labels, self.Switching):
            OUTPUT = Update_Object_Class(getattr(self, 'Image'+L0))
            for (L1, L2) in zip(('Image', 'Green', 'Hue', 'Sat'), OUTPUT): 
                setattr(getattr(self, Object_Name), L1, L2)


    ###############################################################################
    ############################################################################### Process starting from here
    if self.type_of_come_back == 2 and self.temp_counter == 2:
        second_Tracker(self)
        if Global_Var.Just_Paws_Init == 1:
            for label in self.Coor_Labels: 
                for (l0, l1) in zip(self.Labels_Channels0, self.Labels_Channels[1:]): setattr(getattr(self, label), l0, getattr(getattr(self, label),l1))
        else:
            for label in self.Coor_Labels: 
                for (l0, l1) in zip(self.Labels_Channels0, self.Labels_Channels[1:]): setattr(getattr(self, label), l0, getattr(getattr(self, label),l1))
        Global_Var.Edition_Flag = 0
        Global_Var.Edition_Flag_Plot = 1


    elif self.counter<2: # First two times tracking
        ############################################################################### First two manual tracking
        if self.counter == 0:
            Initial_Tracker(self) # Function for manual tracking
        else:
            second_Tracker(self) # Function for manual tracking
            

        for label in self.Coor_Labels: # self.M.H0/self.TAIL.G0/...
            for (l0, l1) in zip(self.Labels_Channels0, self.Labels_Channels[1:]): setattr(getattr(self, label), l0, getattr(getattr(self, label),l1))


    elif self.temp_counter == 1:
        if self.type_of_come_back == 3:
            self = First_Time_Initilization_Variables(self, 1)

        else:
            for label in self.Coor_Labels: # self.M.H0/self.TAIL.G0/...
                for (l0, l1) in zip(self.Labels_Channels0, self.Labels_Channels[1:]): setattr(getattr(self, label), l0, getattr(getattr(TEMP_OBJECT, label),l1))

            for L0, Cam in zip(self.Coor_Labels[0::2], self.Cameras): # Update variables like self.M.Kalman/self.M.Coord_3D/self.M.Kalman_predict_Pos3D/self.M(M2).DLT_Inv_Predict_Pos2D
                if L0[-1] == '3':
                    L3 = '4'; L4 = L0[:-1]
                else:
                    L3 = '2'; L4 = L0

                (OUTPUT) = Kalman_DLT_LQR.DLT_Based_Checking(self, getattr(getattr(self, L0), 'Tr_All')[self.counter,:], getattr(getattr(self, L4+L3), 'Tr_All')[self.counter-1,:], getattr(getattr(self, L0), 'Coord_3D'), getattr(getattr(self, L0), 'Kalman'), Cam)
                for (L1, L2) in zip(['Coord_3D', 'Kalman', 'Kalman_Predict_Pos3D', 'DLT_Inv_Predict_Pos2D_Temp'], OUTPUT): setattr(getattr(self, L0), L1, L2)
        
            for L0 in self.Coor_Labels: # update self.M.H, ... after come back to previous frames
                setattr(getattr(self, L0), 'Tr', getattr(getattr(TEMP_OBJECT, L0), 'Tr_All')[self.counter, 0:2].transpose())
                for L1 in self.Labels_Channels[1:]:
                    setattr(getattr(self, L0), L1 , getattr(getattr(TEMP_OBJECT, L0), L1+'_All')[self.counter, 0])



    else:
        for L0 in self.Coor_Labels:
            On_Each_Other_Flag = 0
            if L0[0] == 'F':
                Fpaw = getattr(getattr(self, L0),'Tr_All')
                Hpaw = getattr(getattr(self, 'H'+L0[1:]),'Tr_All')
            else:
                Fpaw = getattr(getattr(self, 'F'+L0[1:]),'Tr_All')
                Hpaw = getattr(getattr(self, L0),'Tr_All')

            
            # checking if the collision of same side happening
            if self.counter > self.Start_Frame_Collision and abs(Fpaw[self.counter-1, 1] - Hpaw[self.counter-1, 1]) <= getattr(getattr(self, L0), 'Collision_Threshold'):
                setattr(getattr(self, L0), 'First_Collision_Flag', 0)
                
                # check if this is the first time that collision happening
                if getattr(getattr(self, L0), 'Collision_Flag') == 0 and ((abs(Fpaw[self.counter-2, 1] - Hpaw[self.counter-2, 1]) > getattr(getattr(self, L0), 'Collision_Threshold')) or self.counter == self.Start_Frame_Collision+1): 
                    setattr(getattr(self, L0), 'First_Collision_Flag', 1)
                    setattr(getattr(self, L0), 'Collision_Threshold', getattr(getattr(self, L0), 'Collision_Threshold')+self.Threshold_Changes)
                    
                    # this flag shows when the paws or on each other for the first time to get two frames back instead of the previous one
                    # if abs(Fpaw[self.counter-1, 1] - Hpaw[self.counter-1, 1]) <= getattr(getattr(self, L0), 'Collision_Threshold')/2 - self.Threshold_Changes:
                    On_Each_Other_Flag = 1

                setattr(getattr(self, L0), 'Collision_Flag', 1)

                if L0[0] == 'F':
                    Temp_N_SLIC = self.NumSLIC
                    self.NumSLIC = int(self.NumSLIC * 1.5)

                OUTPUT = Tracker_Paw(self, getattr(self, L0), L0, getattr(getattr(self, L0), 'Collision_Flag'), On_Each_Other_Flag)

                if L0[0] == 'F':
                    self.NumSLIC = Temp_N_SLIC


            else:
                # jsut a temprerary flag to know if the previous fram had collision
                setattr(getattr(self, L0), 'Temp_Collision_Flag', getattr(getattr(self, L0), 'Collision_Flag'))
                # Bring back the threshold to what it was
                if getattr(getattr(self, L0), 'Collision_Flag') == 1:
                    setattr(getattr(self, L0), 'Collision_Threshold', getattr(getattr(self, L0), 'Collision_Threshold')-self.Threshold_Changes)
                # set collisions flags to zero
                setattr(getattr(self, L0), 'First_Collision_Flag', 0)
                setattr(getattr(self, L0), 'Collision_Flag', 0)

                OUTPUT = Tracker_Paw(self, getattr(self, L0), L0, getattr(getattr(self, L0), 'Collision_Flag'), On_Each_Other_Flag)


            # Assign the variables after tracking
            for (L1, value) in zip(self.Labels_Channels, OUTPUT): 
                setattr(getattr(self, L0), L1, value) # updating self.FPAW/self.FPAW_H/... after tracking
            setattr(getattr(self, L0), 'Best_SPs', OUTPUT[-1])


            # print(L0, On_Each_Other_Flag, getattr(getattr(self, L0), 'Collision_Flag'), getattr(getattr(self, L0), 'First_Collision_Flag'), getattr(getattr(self, L0), 'Collision_Threshold'))


                

    ############################################################################### Checking otherside collision
    for L0 in self.Coor_Labels: 
        getattr(getattr(self, L0), 'Tr_All')[self.counter, 0:2] = getattr(getattr(self, L0), 'Tr').transpose()
        for L1 in self.Labels_Channels[1:]:
            getattr(getattr(self, L0), L1+'_All')[self.counter, 0] =  getattr(getattr(self, L0), L1)


    if self.counter>2:
        self = Checking_Otherside_Collision(self)


    ############################################################################### Checking the paws are not swtiched
    if getattr(getattr(self, L0), 'Collision_Flag') == 0:
        for L0, L1 in zip(self.Coor_Labels[:4], self.Coor_Labels[4:]): 
            FPAW = getattr(getattr(self, L0),'Tr_All')
            HPAW = getattr(getattr(self, L1),'Tr_All')
            if FPAW[self.counter, 1]-2*self.Threshold_Changes > HPAW[self.counter, 1]:
                # print(FPAW[self.counter, 1], HPAW[self.counter, 1])
                getattr(getattr(self, L0), 'Tr_All')[self.counter, 0:2] = HPAW[self.counter, 0:2]
                getattr(getattr(self, L1), 'Tr_All')[self.counter, 0:2] = FPAW[self.counter, 0:2]

        


    ############################################################################### Keeping 3D Position and Tracked Coordinates Updated
    for L0, Cam in zip(self.Coor_Labels[0::2], self.Cameras_Repeated): # Update variables like self.M.Kalman/self.M.Coord_3D/self.M.Kalman_predict_Pos3D/self.M(M2).DLT_Inv_Predict_Pos2D
        if L0[-1] == '3':
            L3 = '4'; L4 = L0[:-1]
        else:
            L3 = '2'; L4 = L0

        (OUTPUT) = Kalman_DLT_LQR.DLT_Based_Checking(self, getattr(getattr(self, L0), 'Tr'), getattr(getattr(self, L4+L3), 'Tr'), getattr(getattr(self, L0), 'Coord_3D'), getattr(getattr(self, L0), 'Kalman'), Cam)
        for (L1, L2) in zip(['Coord_3D', 'Kalman', 'Kalman_Predict_Pos3D', 'DLT_Inv_Predict_Pos2D_Temp'], OUTPUT): 
            setattr(getattr(self, L0), L1, L2)

        # Coordinates_2D = Kalman_DLT_LQR.DLT_Inverse(self, getattr(getattr(self, L0), 'Coord_3D')[0:3,0], self.Coef, Cam)
        # P1 = getattr(getattr(self, L0), 'Tr')
        # P1 = np.roll(np.reshape(P1,(2,)),1)
        # P2 = getattr(getattr(self, L4+L3), 'Tr')
        # P2 = np.roll(np.reshape(P2,(2,)),1)
        # PP = np.append(P1, P2)
        # DD = getattr(getattr(self, L0), 'Coord_3D')
        # print(L0, sum(abs(DD.flatten())[3:6]), sum(abs(Coordinates_2D-PP)))
    
    for (L0, L1) in  zip(self.Coor_Labels[0::2], self.Coor_Labels[1::2]):
        getattr(getattr(self, L0), "DLT_Inv_Predict_Pos2D")[self.counter, 0:2] = getattr(getattr(self, L0), "DLT_Inv_Predict_Pos2D_Temp")[0:2]
        getattr(getattr(self, L1), "DLT_Inv_Predict_Pos2D")[self.counter, 0:2] = getattr(getattr(self, L0), "DLT_Inv_Predict_Pos2D_Temp")[2:]


    if self.type_of_come_back == 3:
        self.type_of_come_back = 2


    # Check for far mistakes
    # if self.counter > 0:
    #     for L0 in self.Coor_Labels:
    #         # print([getattr(getattr(self, L0), "DLT_Inv_Predict_Pos2D")[self.counter, 0:2], getattr(getattr(self, L0), "Tr_All")[self.counter, 0:2]])
    #         if np.abs(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0]-getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 0]) > 130 or np.abs(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]-getattr(getattr(self, L0), 'Tr_All')[self.counter-1, 1]) >130:
    #             pdb.set_trace()


    if (self.counter > 1 and ((self.Demonstration_Flag==1 and np.mod(self.counter, Global_Var.Jump_Step_demons)==0) or (self.type_of_come_back>0) or (self.counter == len(self.Image_Bank)-1))) or self.temp_counter:
        PLOT(self)

    return(self)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### # Demonstration
def PLOT(self):
    Image_Labeleds = deepcopy(self.Image); Image_Labeleds2 = deepcopy(self.Image2);
    Color = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for (L0, L1) in zip(self.Coor_Labels[0::4], Color):
        Image_Labeleds = cv2.circle(Image_Labeleds, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    for (L0, L1) in zip(self.Coor_Labels[1::4], Color):
        Image_Labeleds2 = cv2.circle(Image_Labeleds2, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    
    Image_Labeleds3 = deepcopy(self.Image3); Image_Labeleds4 = deepcopy(self.Image4);
    for (L0, L1) in zip(self.Coor_Labels[2::4], Color):
        Image_Labeleds3 = cv2.circle(Image_Labeleds3, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    for (L0, L1) in zip(self.Coor_Labels[3::4], Color):
        Image_Labeleds4 = cv2.circle(Image_Labeleds4, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    
    fig = plt.figure(self.Image_Name); fig.canvas.mpl_connect('key_press_event', lambda event: Keyboard_Fun.press(event,plt));
    ax = fig.add_subplot(2, 2, 1);ax.imshow(Image_Labeleds);plt.axis("off"); ax.set_title('Cam {}'.format(1))
    ax = fig.add_subplot(2, 2, 2);ax.imshow(Image_Labeleds2);plt.axis("off"); ax.set_title('Cam {}'.format(2)) 
    ax = fig.add_subplot(2, 2, 3);ax.imshow(Image_Labeleds3);plt.axis("off"); ax.set_title('Cam {}'.format(3))
    ax = fig.add_subplot(2, 2, 4);ax.imshow(Image_Labeleds4);plt.axis("off"); ax.set_title('Cam {}'.format(4))

    plt.show()


def PLOT_Close_Fast(self):
    Image_Labeleds = deepcopy(self.Image); Image_Labeleds2 = deepcopy(self.Image2);
    Color = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for (L0, L1) in zip(self.Coor_Labels[0::4], Color):
        Image_Labeleds = cv2.circle(Image_Labeleds, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    for (L0, L1) in zip(self.Coor_Labels[1::4], Color):
        Image_Labeleds2 = cv2.circle(Image_Labeleds2, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    
    Image_Labeleds3 = deepcopy(self.Image3); Image_Labeleds4 = deepcopy(self.Image4);
    for (L0, L1) in zip(self.Coor_Labels[2::4], Color):
        Image_Labeleds3 = cv2.circle(Image_Labeleds3, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    for (L0, L1) in zip(self.Coor_Labels[3::4], Color):
        Image_Labeleds4 = cv2.circle(Image_Labeleds4, (int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 1]), int(getattr(getattr(self, L0), 'Tr_All')[self.counter, 0])), 10, L1, -1)
    
    fig = plt.figure(self.Image_Name); fig.canvas.mpl_connect('key_press_event', lambda event: Keyboard_Fun.press(event,plt));
    ax = fig.add_subplot(2, 2, 1);ax.imshow(Image_Labeleds);plt.axis("off"); ax.set_title('Cam {}'.format(1))
    ax = fig.add_subplot(2, 2, 2);ax.imshow(Image_Labeleds2);plt.axis("off"); ax.set_title('Cam {}'.format(2))
    ax = fig.add_subplot(2, 2, 3);ax.imshow(Image_Labeleds3);plt.axis("off"); ax.set_title('Cam {}'.format(3))
    ax = fig.add_subplot(2, 2, 4);ax.imshow(Image_Labeleds4);plt.axis("off"); ax.set_title('Cam {}'.format(4))

    if Global_Var.Return_to_Tracking == 1:
        plt.ioff()
    else:
        plt.ion()
        
    plt.show()
    plt.pause(self.Demonstration_Delay)

    if Global_Var.Return_to_Tracking == 1:
        plt.ioff()

    plt.close('all')



############################################################################### Classes:
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Object Class
class Objects:
    def __init__(self, Image):
        self.Image = deepcopy(Image)

        # self.Green = cv2.medianBlur(cv2.equalizeHist(Image[...,1]), 1)
        self.Green = cv2.medianBlur(Image[...,1], 1)
        self.HSV = cv2.cvtColor(Image, cv2.COLOR_RGB2HSV)
        self.Hue = cv2.medianBlur(self.HSV[..., 0], 1)
        # self.Sat = cv2.medianBlur(cv2.equalizeHist(Image[...,0]), 1) # Red Channel
        self.Sat = cv2.medianBlur(Image[...,0], 1)
        self.Image = deepcopy(Image)


def Update_Object_Class(Image):
    IMAGE = deepcopy(Image) 

    # Green = cv2.medianBlur(cv2.equalizeHist(Image[...,1]), 1)
    Green = cv2.medianBlur(Image[...,1], 1)
    HSV = cv2.cvtColor(Image, cv2.COLOR_RGB2HSV)
    Hue = cv2.medianBlur(HSV[..., 0], 1)
    # Sat = cv2.medianBlur(cv2.equalizeHist(Image[...,0]), 1) # Red Channel
    Sat = cv2.medianBlur(Image[...,0], 1)
    IMAGE = deepcopy(Image) 
    return(IMAGE, Green, Hue, Sat)



def Load_Frames(Bank):
    Image_Name = Bank
    Image = cv2.imread(Image_Name)
    b, g, r = cv2.split(Image)
    Image = cv2.merge((r, g, b))

    return(Image_Name, Image)



############################################################################### 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Main Class
class Marker_Tracker(object): # The main class
    # def __init__(self, CurrentPath, DLT_Path): 
    def __init__(self, GUI_Data): 
        ####################################################################### Getting input and initial numbers for running the tracker
        self.NumSLIC = ARGS['NumSLIC']
        self.ComSLIC = ARGS['ComSLIC']
        self.SigSLIC = ARGS['SigSLIC']
        self.PawXWindowSize = ARGS['PawXWindowSize']
        self.PawYWindowSize = ARGS['PawYWindowSize']
        self.WindowSize = ARGS['WindowSize']
        self.Demonstration_Flag = ARGS['Demonstration_Flag']
        self.Collision_Threshold = ARGS['Collision_Threshold']
        self.second_window_size = ARGS['WindowSize_Second']
        self.Demonstration_Delay = ARGS['Demonstration_Delay']
        self.Jum_To_Frame = ARGS['Jump_Frame_Number']
        self.Threshold_Changes = ARGS['Threshold_Changes']
        self.Start_Frame_Collision = ARGS['Start_Frame_Collision']
        self.Weights_F = ARGS['Weights_F']
        self.Weights_H = ARGS['Weights_H']
        self.Collison_Demonstartion = ARGS["Collison_Demonstartion"]
        ####################################################################### General variables needed to be defined
        self.counter = 0

        self.CurrentPath = ARGS['cam1_path']
        self.Cam_Position_in_string = self.CurrentPath.find('cam')
        # self.spline_Labels = [['0','2','4','6'],['1','3','5','7']]
        self.spline_Labels = ['0', '0', '0', '0', '6', '6', '6', '6']
        if self.Cam_Position_in_string == -1:
            self.Cam_Position_in_string = self.CurrentPath.find('Cam')
            if self.Cam_Position_in_string == -1:
                self.Cam_Position_in_string = self.CurrentPath.find('CAM')
        self.cam2 = deepcopy(self.CurrentPath)
        self.cam3 = deepcopy(self.CurrentPath)
        self.cam4 = deepcopy(self.CurrentPath)

        self.list1 = list(self.cam2)
        self.list1[self.Cam_Position_in_string+3] = '2'
        self.cam2 = ''.join(self.list1)
        self.list1[self.Cam_Position_in_string+3] = '3'
        self.cam3 = ''.join(self.list1)
        self.list1[self.Cam_Position_in_string+3] = '4'
        self.cam4 = ''.join(self.list1)

        self.Image_Bank = sorted(glob(os.path.join(self.CurrentPath, "*.png")))
        self.Image_Bank2 = sorted(glob(os.path.join(self.cam2, "*.png")))
        self.Image_Bank3 = sorted(glob(os.path.join(self.cam3, "*.png")))
        self.Image_Bank4 = sorted(glob(os.path.join(self.cam4, "*.png")))

        self.DLT_Path = ARGS['dlt_path']
        self.Coef = genfromtxt(self.DLT_Path, delimiter=',')
        self.Cameras = np.array([[0 , 1], [2, 3]]) # Camera Numbers
        self.Cameras_Repeated = np.tile(self.Cameras, [4,1])
        self.Collision_Flag = 0
        ####################################################################### Labels
        self.List_Switch = ['','2','3','4']
        self.Switching = np.tile(self.List_Switch, 2)
        self.Coor_Labels = ("FPAW", "FPAW2", "FPAW3", "FPAW4", "HPAW", "HPAW2", "HPAW3", "HPAW4")
        self.Labels_Channels0 = ("H0", "G0", "S0")
        self.Labels_Channels = ("Tr","H", "G", "S")
        self.Loaded_Cordinates = []
        self.Number_Marker = 2
        self.temp_counter = 0
        self.type_of_come_back = 0

        self.List_Features = ['Ycoor', 'Xcoor', 'Hue', 'Gray', 'Sat', 'Hue0', 'Gray0', 'Sat0', 'Frame']



    def Main_Loop_Function(self): # Main loop to run the code for all frames in the folders
        While_Loop_Flag = 1; change_status = 0
        self.Tracking_Mode = ARGS['Tracking_Mode']



        while While_Loop_Flag == 1:
            if self.Jum_To_Frame>-1: # This is checking if user asked for specific number of frame from begining using terminal
                self.counter = deepcopy(self.Jum_To_Frame)
                self.Tracking_Mode = 0
                self.Jum_To_Frame = -1


            for L0, L1 in zip(self.List_Switch, [self.Image_Bank, self.Image_Bank2, self.Image_Bank3, self.Image_Bank4]):
                (Image_Name, Image) = Load_Frames(L1[self.counter])
                setattr(self, 'Image_Name'+L0, Image_Name)
                setattr(self, 'Image'+L0, Image)

            ############################################################################### Preprocessing: Loading Image, Checking camera, Fliping Images, and HSV

            print ("Frame number %d from %s is in processing" % (self.counter+1, self.Image_Name))

            self.Image3 = np.fliplr(self.Image3)
            self.Image4 = np.fliplr(self.Image4)


            if os.path.exists('./Tracked_Paw_Final_Results.csv') and self.counter == 0:
                for (Object_Name, L0) in zip(self.Coor_Labels, self.Switching): setattr(self, Object_Name, Objects(getattr(self, 'Image'+L0)))
                Loaded_Cordinates = CSV_RW.Read_CSV(self)
                counter = 0
                for L0 in self.Coor_Labels[0:int(np.round(Loaded_Cordinates.shape[1]/2))]: 
                    setattr(getattr(self, L0), 'Tr_All', Loaded_Cordinates[:,counter*2:counter*2+2])
                    counter += 1
                print(colored("WARNING: There is a CSV file showing the trial has been tracked, MAKE sure about the process doing",'yellow'))


            if self.Tracking_Mode == 1:
                self = Running_Tracker(self)
                change_status = 0
            else:
                if not(os.path.exists('./Tracked_Paw_Final_Results.csv')):
                    print(colored("ERORR: Loading the requested CSV file has been denied. Setting back to the tracking mode from the visulaization mode.", 'red'))
                    Global_Var.Return_to_Tracking = 1
                else:
                    PLOT_Close_Fast(self)


            ############################################################################### Counter update (General and for reviewing previous frames)
            if self.counter == 0:
                self.temp_counter_Previous_Frame = 0
            else:
                self.temp_counter_Previous_Frame = deepcopy(self.temp_counter)


            if Global_Var.Saving_csv == 1:
                CSV_RW.Save_CSV(self)
                Global_Var.Saving_csv = 0
                print(colored("The data were saved successfully in middle of process.", 'green'))


            if Global_Var.Frame_Number > 0:
                self.counter = Global_Var.Frame_Number-1
                self.temp_counter = 1 # counter for the first two times running the code afetr asking for edition
                self.type_of_come_back = 1 # coming back but not editing 'a' and 'g'; 'a' needs edition which is done later
                Global_Var.Frame_Number = 0

            elif Global_Var.Edition_Flag == 1:
                self.counter -= 1
                if self.counter <2: self.counter = 2
                self.temp_counter = 1
                self.type_of_come_back = 2 # when needing editing using 'e' and 'w'
                Global_Var.Edition_Flag = 0

            elif Global_Var.Come_Back_Flag == 1:
                self.counter -= (Global_Var.Come_Back_step + 1)
                if self.counter < 2: self.counter = 2
                self.temp_counter = 1
                self.type_of_come_back = 1 # coming back but not editing for the rest of keyboards
                Global_Var.Come_Back_Flag = 0


            else:
                self.counter += 1
                if self.type_of_come_back > 0:
                    self.temp_counter += 1
                    if self.temp_counter > 2: # check if twice the code has been running after request for edition
                        self.type_of_come_back = 0 # reset come back flag
                        self.temp_counter = 0

            if Global_Var.Return_to_Tracking == 1:
                self.counter -= 1
                if self.counter <2: self.counter = 2 # counter cannot be less than 2
                self.temp_counter = 1
                self.type_of_come_back = 3 # when needing editing for pressing 'a'
                Global_Var.Edition_Flag = 0 # reset edition flag
                self.Tracking_Mode = 1 # activate tracking mode
                Global_Var.Return_to_Tracking = -1 # reset tracking mode flag from keyboard

            if Global_Var.Return_to_Tracking == 0:
                self.Tracking_Mode = 0 # Switch to visualization mode
                Global_Var.Return_to_Tracking = -1 # reset tracking mode flag from keyboard


            if self.counter == min(len(self.Image_Bank), len(self.Image_Bank2)):
                While_Loop_Flag = 0
                

                
        CSV_RW.Save_CSV(self)
        
        print(colored("Tracking has been Completed and saved successfully.", 'green'))





    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    import guidata
    _app = guidata.qapplication()
    
    GUI_Data = TestParameters()
    GUI_Data.edit()
    print(GUI_Data)

    ARGS = Set_Variable(ARGS, GUI_Data)
    Info = Marker_Tracker(ARGS)
    Global_Var.Global_Var() 
    Info.Main_Loop_Function()
