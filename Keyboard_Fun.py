import Global_Var
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import threading
import pdb
import re


def press(event, plt): # This function checks the keyboard when a pop-up window is open
    if event.key == 'r': # reset the coordinates; it means start marking the landmarks
        Global_Var.Coords = []

    if event.key == 'q': # closes the window
        Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == 'e': # edition
        Global_Var.Coords = []; Global_Var.Come_Back_step = 2; Global_Var.Edition_Flag = 1; Just_Paws_Init = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == 'w':
        Global_Var.Coords = []; Global_Var.Come_Back_step = 2; Global_Var.Edition_Flag = 1; Just_Paws_Init = 1; Global_Var.Jump_Step_demons = 1; plt.close()



    if event.key == 'z': # Demonstrate 1 frame by 1 frame
        Global_Var.Jump_Step_demons = 5; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'x': # Demonstrate 5 frame by 5 frame
        Global_Var.Jump_Step_demons = 10; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'c': # Demonstrate 10 frame by 10 frame
        Global_Var.Jump_Step_demons = 25; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'v': # Demonstrate 50 frame by 50 frame
        Global_Var.Jump_Step_demons = 50; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'b': # Demonstrate 150 frame by 150 frame
        Global_Var.Jump_Step_demons = 150; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'n': # Demonstrate 300 frame by 300 frame
        Global_Var.Jump_Step_demons = 250; Global_Var.Forward_Flag = 1; Global_Var.Come_Back_Flag = 0; plt.close()

    if event.key == 'm': # Demonstrate till end
        Global_Var.Jump_Step_demons = 1000; Global_Var.Forward = 1; Global_Var.Come_Back_Flag = 0; plt.close()



    if event.key == '1': # Jump backward for 2 frame
        Global_Var.Come_Back_step = 2; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '2': # Jump backward for 5 frame
        Global_Var.Come_Back_step = 5; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '3': # Jump backward for 10 frame
        Global_Var.Come_Back_step = 10; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '4': # Jump backward for 50 frame
        Global_Var.Come_Back_step = 25; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '5': # Jump backward for 150 frame
        Global_Var.Come_Back_step = 50; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '6': # Jump backward for 300 frame
        Global_Var.Come_Back_step = 150; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()

    if event.key == '7': # Jump backward for 300 frame
        Global_Var.Come_Back_step = 250; Global_Var.Come_Back_Flag = 1; Global_Var.Forward_Flag = 0; Global_Var.Jump_Step_demons = 1; plt.close()



    if event.key == 'a': # For Pause of just visualization and entering to the process mode
        response = input("Please enter your the frame number: ")
        Global_Var.Frame_Number = int(re.findall(r'\d+', response)[0])
        Global_Var.Return_to_Tracking = 1;

    if event.key == 'd': # Return to Visualization mode
        Global_Var.Return_to_Tracking = 0; plt.close()


    if event.key == 'f': # Save
        Global_Var.Saving_csv = 1; Global_Var.Jump_Step_demons = 1; plt.close()


    if event.key == 'g': # For jumping to specific frame number
        response = input("Please enter your the frame number: ")
        Global_Var.Frame_Number = int(re.findall(r'\d+', response)[0]); plt.close()



def onclick(event, plt): # This function gets the mouse click coordinates
    ix, iy = round(event.xdata, 0), round(event.ydata, 0)
    Global_Var.Coords.append((ix, iy))



def FSLIC(IMAGE=None, IM=None, NumSLIC=None, ComSLIC=None, SigSLIC=None, Initial=None): # This is SLIC function
    IMAGE = img_as_float(IMAGE)
    Segments = slic(IMAGE, n_segments=NumSLIC, sigma=SigSLIC, compactness=ComSLIC)
    if Initial == True:  # if initial is true, it returns the fusedimage showing the SLIC segments
        Fusied_Image = mark_boundaries(IM, Segments, color = (0.7, 0.7, 0.7))
        return (Segments, Fusied_Image)
    else:
        return (Segments)