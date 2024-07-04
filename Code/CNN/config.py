import numpy as np

# Values we want to keep constant throughout the project


imshape_ = (640, 480, 3)
            # Class names:      # Class color  # Channel Number in target matrix
class_colors_ = {'platform':     (000,000,187), # 0
               'spike':          (255,000,000), # 1
               'player':         (255,255,255), # 2
               'yellow jump orb':(000,255,000), # 3
               'blue jump orb':  (100,000,000), # 4
               'blue pad':       (000,000,100), # 5
               'yellow pad':     (000,000,200), # 6
#              'background':     (000,000,000)  # 7 # Not needed as it is the default value
}

labels_ = class_colors_.keys()
n_classes_ = len(class_colors_) + 1 # +1 for background class

batch_size_ = 32  