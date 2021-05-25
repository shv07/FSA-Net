from logging import exception
import os
import cv2
import sys
from keras.backend.theano_backend import reset_uids

from numpy.core.fromnumeric import resize

# dir path to the root dir of FSANET 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from math import cos, sin
from lib.FSANET_model import *
from lib import image_reader_util
import numpy as np
from keras.layers import Average
import datetime
# from moviepy.editor import *
# from mtcnn.mtcnn import MTCNN



#*********************Global Initialisation for speed up*************************#
# face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
# detector = MTCNN()

# load model and weights
img_size = 64
stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1
img_idx = 0
detected = '' #make this not local variable
time_detection = 0
time_network = 0
time_plot = 0
ad = 0.6

#Parameters
num_capsule = 3
dim_capsule = 16
routings = 2
stage_num = [3,3,3]
lambda_d = 1
num_classes = 3
image_size = 64
num_primcaps = 7*3
m_dim = 5
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

num_primcaps = 8*8*3
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

print('Loading models ...')

weight_file1 = os.path.join(parent_dir,'pre-trained','300W_LP_models','fsanet_capsule_3_16_2_21_5','fsanet_capsule_3_16_2_21_5.h5')
model1.load_weights(weight_file1)
print('Finished loading model 1.')

weight_file2 = os.path.join(parent_dir, 'pre-trained','300W_LP_models','fsanet_var_capsule_3_16_2_21_5','fsanet_var_capsule_3_16_2_21_5.h5')
model2.load_weights(weight_file2)
print('Finished loading model 2.')

weight_file3 = os.path.join(parent_dir,'pre-trained','300W_LP_models','fsanet_noS_capsule_3_16_2_192_5','fsanet_noS_capsule_3_16_2_192_5.h5')
model3.load_weights(weight_file3)
print('Finished loading model 3.')

inputs = Input(shape=(64,64,3))
x1 = model1(inputs) #1x1
x2 = model2(inputs) #var
x3 = model3(inputs) #w/o
avg_model = Average()([x1,x2,x3])
model = Model(inputs=inputs, outputs=avg_model)



# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([parent_dir,"demo", "face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join([parent_dir,"demo","face_detector",
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)



print('Start detecting pose ...')
detected_pre = np.empty((1,1,1))

#****************************************************************************************************#

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    #print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img, [(int(tdx), int(tdy)), (int(x3),int(y3))]
    
def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):
    poses = []
    result_vectors = []
    # loop over the detections
    if detected.shape[2]>0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY
                
                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
                
                face = np.expand_dims(faces[i,:,:,:], axis=0)
                p_result = model.predict(face)
                
                face = face.squeeze()
                img, result_vector = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])
                
                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                
                yaw, pitch, roll = p_result[0][0], p_result[0][1], p_result[0][2]

                poses.append(tuple([yaw,pitch,roll]))
                result_vectors.append(result_vector)

                #print(f"yaw: {yaw}, roll: {roll}, pitch: {pitch}")
                pitch = pitch * np.pi / 180
                yaw = -(yaw * np.pi / 180)
                roll = roll * np.pi / 180
                #print(f"yaw: {yaw}, roll: {roll}, pitch: {pitch}")

    #cv2.imshow("result", input_img)
    
    return input_img, poses, result_vectors #,time_network,time_plot

def detect_head_poses(input_img, debug = False):        
    if input_img is None:
        print("Invalid Image")
        return None
    img_h, img_w, _ = np.shape(input_img)

        
    time_detection = 0
    time_network = 0
    time_plot = 0
    
    # detect faces using LBP detector
    gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    # detected = face_cascade.detectMultiScale(gray_img, 1.1)
    # detected = detector.detect_faces(input_img)
    # pass the blob through the network and obtain the detections and
    # predictions
    blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detected = net.forward()

    faces = np.empty((detected.shape[2], img_size, img_size, 3))

    input_img, poses, result_vectors = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
    
    if debug:
        cv2.imwrite('./img/'+str(img_idx)+'.png',input_img)
    return poses, result_vectors


def CalculateResultPose(origin:tuple, point:tuple)->str:
    if len(origin)!=2 and len(point)!=2:
        return ""
    x1, y1 = origin
    x2, y2 = point

    result = ""
    if x2>x1:
        result = result+"left"
    elif x2==x1:
        result = result+"straight"
    else:
        result = result+"right"

    result = result+"-"
    
    if y2>y1:
        result = result+"down"
    elif y2==y1:
        result = result+"straight"
    else:
        result = result+"up"
        
    return result

def DetectHeadPose(img:str, debug = False, tolerance=20)->dict:
    """
    params:
        img (str): Base64 image string
    returns:
        predictions of yaw, pitch, roll mapped as right, up, side respectively
    """
    result = {"Task":"HeadPoseEstimation",
              "Result":[]}
    if img is None or img=="":
        return result
    input_img = image_reader_util.base64_to_cv2_img(img)
    poses = []
    result_vectors = []

    try:
        poses, result_vectors = detect_head_poses(input_img, debug=debug)
    except exception as e:
        print(f"{datetime.datetime.utcnow()}  Exception at detect_head_poses \n", e)
        return result

    assert len(poses)==len(result_vectors)
    
    result["Result"] = [{"Angles":{"Right":pose[0],"Up":pose[1], "Side":pose[2]},
                         "Direction":CalculateResultPose(result_vectors[idx][0], result_vectors[idx][1])}
                          for idx,pose in enumerate(poses) if (max(pose)>=tolerance and len(pose)==3)]
    
    return result
