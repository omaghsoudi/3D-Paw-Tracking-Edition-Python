import pandas as pd
import os
import numpy as np
from copy import deepcopy

############################################################################### Saving data :D
def Save_CSV(self):
    File_Name = '/Tracked_Paw_Final_Results.csv'

    Temp = ['The 3D marker tracked was used by the following settings:',
            'SLIC parameters as Number %d Sig %d Comp %d' %(self.NumSLIC, self.SigSLIC, self.ComSLIC),
            'Window size was %d' %(self.WindowSize),
            'Path for the first camera was %s' %(self.CurrentPath),
            'Frames', # Ends up being the header name for the index.
            ]
    header = '\n'.join([line for line in Temp])

    Final_Results = pd.DataFrame(self.Image_Bank, columns = ['First Camera Frames Name']) 
    Final_Results = pd.concat([Final_Results, pd.DataFrame(self.Image_Bank2, columns = ['Second Camera Frames Name'])], axis = 1)
    Final_Results = pd.concat([Final_Results, pd.DataFrame(self.Image_Bank3, columns = ['Second Camera Frames Name'])], axis = 1)
    Final_Results = pd.concat([Final_Results, pd.DataFrame(self.Image_Bank4, columns = ['Second Camera Frames Name'])], axis = 1)
    for L0 in self.Coor_Labels: 
        Temp = pd.DataFrame(getattr(getattr(self, L0), 'Tr_All'), columns = ['Y Coordinate '+L0,'X Coordinate '+L0]) 
        Final_Results = pd.concat([Final_Results,Temp], axis = 1)


    Result_File_Name = os.path.expanduser(self.CurrentPath) + File_Name
    try:
        os.remove(Result_File_Name)
    except OSError:
        pass

    with open(File_Name, 'wt') as CSV:
        for line in header:
            CSV.write(line)
        Final_Results.to_csv(CSV)



def Read_CSV(self):
    File_Name = '/Tracked_Paw_Final_Results.csv'
    
    data_temp = pd.read_csv(File_Name, skiprows=[0,1,2,3], sep=',', dtype={'ID': object})
    data = data_temp.iloc[:,3:].values
    Loaded_Cordinates = deepcopy(data)
    if np.isnan(data).any():
        counter = 0
        for count in range(data.shape[0]):
            if np.isnan(data[counter][:]).all():
                data = np.delete(data, counter, axis = 0)
                Loaded_Cordinates = deepcopy(data)
                counter -= 1
            counter += 1
    return (Loaded_Cordinates)
