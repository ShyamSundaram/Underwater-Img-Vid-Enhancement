import numpy as np
from PIL import Image
 
# 24-bit to 8-bit grayscale
image1 = Image.open(r'F:\PyTorch\PDC\PyTorch-Underwater-Image-Enhancement\test_img\img.png')
print(image1.mode)
image1=image1.convert('RGB')
image1.save("out.png")