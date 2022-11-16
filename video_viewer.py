# img_viewer.py

import PySimpleGUI as sg
import os.path

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Video Folder"),
        sg.In(size=(25, 1), enable_events=True, key="Videos"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an Video from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-Video-")],
]


# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Video Viewer", layout)



while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "Browse":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []
    
       # fnames = [
        #    f
         #   for f in file_list
          #  if os.path.isfile(os.path.join(folder, f))
           # and f.lower().endswith((".mp4", ".avi"))
        #]
        window["-FILE LIST-"].update(file_list)
        
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
     try:
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
        window["-TOUT-"].update(filename)
        window["-IMAGE-"].update(filename=filename)
     except:
        pass


window.close()


import os


for root, dirs, files in os.walk(r'F:'):
    # select file name
    for file in files:
        # check the extension of files
        if file.endswith('.png'):
            # print whole path of files
            print(os.path.join(root, file))











