import cv2 
import numpy as np
from matplotlib import pyplot as plt
 
img_rgb = cv2.imread('../Sample_Data/map.png')

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('../Sample_Data/square.png', cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread('../Sample_Data/spike.png', cv2.IMREAD_GRAYSCALE)
template_player = cv2.imread('../Sample_Data/player.png', cv2.IMREAD_GRAYSCALE)
template_spike_small = cv2.imread('../Sample_Data/spike_small.png', cv2.IMREAD_GRAYSCALE)


w, h = template.shape[::-1]
w2, h2 = template2.shape[::-1] # spike
w_player, h_player = template_player.shape[::-1] # player
w_spike_small, h_spike_small = template_spike_small.shape[::-1] # spike_small

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
res_player = cv2.matchTemplate(img_gray,template_player,cv2.TM_CCOEFF_NORMED)
res_spike_small = cv2.matchTemplate(img_gray,template_spike_small,cv2.TM_CCOEFF_NORMED)

threshold = 0.7
threshold2 = 0.8
threshold_player = 0.7
threshold_spike_small = 0.87
loc = np.where( res >= threshold)
loc2 = np.where( res2 >= threshold2)
loc_player = np.where( res_player >= threshold)
loc_spike_small = np.where( res_spike_small >= threshold_spike_small)


for pt in zip(*loc2[::-1]):
 
 cv2.rectangle(img_rgb, (pt[0]+int(w2/3),pt[1]+int(h2/4)), (pt[0] + int(2*w2/3), int(pt[1] + 3*h2/4)), (0,255,0), -1)
 # fill the rectangle

for pt in zip(*loc[::-1]):
 cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), -1)

for pt in zip(*loc_player[::-1]):
 cv2.rectangle(img_rgb, pt, (pt[0] + w_player, pt[1] + h_player), (255,0,0), -1)

for pt in zip(*loc_spike_small[::-1]):
 cv2.rectangle(img_rgb, pt, (pt[0] + w_spike_small, pt[1] + h_spike_small), (255,255,0), -1)



cv2.imwrite('res.png',img_rgb)

# find the difference between the two images
map = cv2.imread('../Sample_Data/map.png')
res = cv2.imread('res.png')
diff = cv2.absdiff(map, res)
cv2.imwrite('diff.png', diff)
