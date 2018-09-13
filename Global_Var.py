# from PIL import ImageGrab

############################################################################### Globals
def Global_Var():
    global Coords, Edition_Flag, Forward_Flag, Jump_Step_demons, Come_Back_step, Come_Back_Flag, Frame_Number, Return_to_Tracking, Saving_csv, Just_Paws_Init
    
    Coords = []
    Forward_Flag = 1
    Edition_Flag = 0
    Jump_Step_demons = 1
    Come_Back_step = 2
    Come_Back_Flag = 0
    Frame_Number = 0
    Return_to_Tracking = -1
    Saving_csv = 0
    Just_Paws_Init = 0

    # img = ImageGrab.grab()
    # Monitor_Size = img.size