from PIL import Image 
import matplotlib.pyplot as plt 
left = 600
top = 320
width = 1050
height = 1050


# image = Image.open("./t"+str(0)+".png")
# plt.imshow(image.crop((left,top,left+width,top+height)))
# plt.show()

for i in range(6): 
    image = Image.open("./t"+str(i)+".png")
    # plt.imshow(image)
    # plt.show()
    cropeed = image.crop((left,top,left+width,top+height))
    file_name = f"./Taipei{i}.png"
    cropeed.save(file_name)