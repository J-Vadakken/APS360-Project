import numpy as np

# Values we want to keep constant throughout the project


imshape_ = (3, 120, 160)
            # Class names:      # Class color  # Channel Number in target matrix
class_colors_ = {'platform':     (000,000,187), # 0
               'spike':          (127,000,000), # 1
               'player':         (255,255,255), # 2
               'yellow jump orb':(255,255,68), # 3
               'blue jump orb':  (17,255,255), # 4
               'blue pad':       (7,102,102), # 5
               'yellow pad':     (102,102,27), # 6
               'portal':         (17, 170, 0) #7 
#              'background':     (000,000,000)  # 7 # Not needed as it is the default value
}

labels_ = class_colors_.keys()
n_classes_ = len(class_colors_) + 1 # +1 for background class

batch_size_ = 32  