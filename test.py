import cv2 as cv 
import numpy as np
from tkinter import *  
from tkinter import font as tkFont
from PIL import ImageTk,Image 
import cvlib 
from cvlib.object_detection import draw_bbox

#helv36 = tkFont.Font(family='Helvetica', size=36, weight='bold')

KNOWN_DISTANCE = 45 
CAR_WIDTH = 64

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)

FONTS = cv.FONT_HERSHEY_COMPLEX

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
  
    data_list = []
    count=0
    
    for (classid, score, box) in zip(classes, scores, boxes):
     
        if classid == 2:
            
            color= COLORS[int(classid) % len(COLORS)]        
            label = "%s : %f" % (class_names[classid[0]], score)
            
            cv.rectangle(image, box, color, 2)
            cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)  
            
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])            
        
            count+=1
        

    
 
       # if classid ==67: 
       #   data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        
    return data_list,count
    #return count
def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance
    
ref_car = cv.imread('ReferenceImages/image20.png')

car_data,x = object_detector(ref_car)
car_width_in_rf = car_data[0][1]   


focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280 )
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)



def Run():
    #ix = PhotoImage(file=r'C:\Users\rk474\Downloads\Yolov4-Detector-and-Distance-Estimator-master\Yolov4-Detector-and-Distance-Estimator-master\qq.png')
    #canvas.create_image(0,0,image = ix,anchor = NW)
   
    while 1:
        ret, frame = cap.read()
        im3 = frame[0 :500,   0:640]
        im4 = frame[0 :500, 640:1280]
        
        data3,kk = object_detector(im3)      
        for d in data3:         
            if d[0] =='car':
                distance = distance_finder (focal_car, CAR_WIDTH, d[1])
                x, y = d[2]
            cv.rectangle(im3, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(im3, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
            
        
        data4,kk1 = object_detector(im4) 
        for d in data4:
            if d[0] =='car':
                distance = distance_finder (focal_car, CAR_WIDTH, d[1])
                x, y = d[2]
            cv.rectangle(im4, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(im4, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)    

        cv.imwrite('imej1.png',im3)
        cv.imwrite('imej2.png',im4) 
        
        imj1 = PhotoImage(file=r'path of imej1.png')
        imj2 = PhotoImage(file=r'path of imej2.png')
        
        canvas.create_image(70,60, image = imj1,anchor = NW)
        canvas.image = imj1
        canvas.update_idletasks()
            
        canvas.create_image(780,60,image = imj2,anchor = NW)
        canvas.image = imj2
        canvas.update_idletasks()
                
        canvas.create_text(400,580,fill="yellow",font="Times 30 bold",text="#cars = "+str(kk),tag = "tag1")
        canvas.text = "#cars = "+str(kk)
        canvas.update_idletasks() 
        
        canvas.create_text(1100,580,fill="yellow",font="Times 30 bold",text="#cars = "+str(kk1),tag = "tag2")
        canvas.text = "#cars = "+str(kk1)
        canvas.update_idletasks()
        
        
        if(kk>0 and kk1>0):
            canvas.create_text(740,670,fill="red",font="Times 100 bold",text="🛑")
        else:
            canvas.create_text(740,670,fill="green",font="Times 100 bold",text="🛑")
        
        canvas.delete("tag1")
        canvas.delete("tag2")
            
        
        key = cv.waitKey(1)
        if key ==ord('q'):
            break

def Stop():
    root.destroy().pack()
    cap.release()
    cv.destroyAllWindows()


root=Tk()
root.title("Collision Warning System")
root.geometry('1920x1080')

canvas = Canvas(root,bg='black',bd=3,relief='groove')
canvas.pack(expand=1,fill=BOTH)

Startbutton = Button(root, width=10, height=2, text='Start',bg='light green', command=lambda: Run())
Startbutton.place(x=650, y=10)

StopButton = Button(root, width=10, height=2, text='Stop',bg='orange', command=lambda: Stop())
StopButton.place(x=740, y=10)

root.mainloop()
