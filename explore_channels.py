

import cv2
import numpy as np

# color = r"G:\kaikeba_dataset\live-detection\phase1\Training\real_part\CLKJ_AS0005\real.rssdk\color\1.jpg"
# ir = r"G:\kaikeba_dataset\live-detection\phase1\Training\real_part\CLKJ_AS0005\real.rssdk\depth\1.jpg"
# depth = r"G:\kaikeba_dataset\live-detection\phase1\Training\real_part\CLKJ_AS0005\real.rssdk\ir\1.jpg"



# color_img = cv2.imread(color)

# color_img_np = np.asarray(color_img)

# print("img_np.shape: ", color_img_np.shape)


# ir_img = cv2.imread(ir)

# ir_img_np = np.asarray(ir_img)

# print("img_np.shape: ", ir_img_np.shape)


# depth_img = cv2.imread(depth)

# depth_img_np = np.asarray(depth_img)

# print("img_np.shape: ", depth_img_np.shape)

# img_np.shape:  (326, 313, 3)
# img_np.shape:  (149, 121, 3)
# img_np.shape:  (149, 121, 3)

channel = 1024
group = 32
steps = [group*i for i in range(channel//group)]
print(steps)
steps = steps[1:]
print(steps)
print("*"*8)
print(steps[0])
print(channel)
print(steps[0]/channel)
print("&"*8)
for i in range(len(steps)):
    print(steps[i], steps[i]/channel)
    print("\n")
    print("\n")
