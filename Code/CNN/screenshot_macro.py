import keyboard as kb
import time
import math
import sys

#it takes 50 frames to go through an entire "screen" of objects at 30fps, on lowest aspect ratio
#framesteppers: e goes forwards.

def run_macro():
    for i in range(n_screenshots):
        for j in range(frames_interval): 
            #forward step a set number of times
            kb.press_and_release(forward_step)
            time.sleep(0.05)
        # kb.press_and_release(screenshot)
        time.sleep(1)

def stop_macro():
    print("Stopping macro. ")
    sys.exit()

#KEYBINDINGS (adjust depending on what you are doing)
forward_step = "e"
backward_step = "q"
screenshot = "t"
 
#CAPTURE SETTINGS
frames_interval = 50 #number of frames between screenshots
level_length = 90 #level length in seconds, will be used to prevent the macro from running too long
fps = 30 #frames per second for the game

n_screenshots = math.floor(level_length*fps/frames_interval) #number of screenshots to be taken

print("Macro working. Press space to begin.")
kb.wait('space')
print("Macro activating in 5s. Tab into the level. Press space again to emergency stop.")
time.sleep(5)
kb.on_press_key("space", stop_macro())
run_macro()
