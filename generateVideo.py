import os 
import cv2  
from PIL import Image 

print(os.getcwd()) 

os.chdir("/home/sushant/Desktop/bhavam-code/Tracker/outputvid3/")   
path = "/home/sushant/Desktop/bhavam-code/Tracker/outputvid3/"

mean_height = 0
mean_width = 0
  
num_of_images = len(os.listdir('.')) 
print(num_of_images) 

for file in os.listdir('.'): 
    im = Image.open(os.path.join(path, file)) 
    width, height = im.size 
    mean_width += width 
    mean_height += height 

mean_width = int(mean_width / num_of_images) 
mean_height = int(mean_height / num_of_images)

for file in os.listdir('.'): 
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"): 
        
        im = Image.open(os.path.join(path, file))  
   
        
        width, height = im.size    
        print(width, height) 
  
          
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)  
        imResize.save( file, 'JPEG', quality = 95)
        
        print(im.filename.split('\\')[-1], " is resized")  

def generate_video(): 
    image_folder = '/home/sushant/Desktop/bhavam-code/Tracker/outputvid3/'
    video_name = 'mygeneratedvideo.avi'
    os.chdir("/home/sushant/Desktop/bhavam-code/Tracker/outputvid3/") 
      
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
     
   
    print(images)  
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))  
  
     
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
      
     
    cv2.destroyAllWindows()  
    video.release() 
  
  

generate_video()