import numpy as np
import cv2
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from jetracer.nvidia_racecar import NvidiaRacecar
from marvelmind import MarvelmindHedge
import urllib3, json
import threading
from time import sleep

from object_detection.detect_objects import init, detect_objects, take_action


#--editable--#
mid = 199       #picture midpoint
midOffset = 65  #white mask slice offset
consoleInfo = False
gpsId = 0

frame = np.zeros((398, 224,3), dtype=np.uint8)

#fallback (kada ne nade b/z crtu n frameova)
whiteFB = 1
yellowFb = 2

#image resolution
width  = 398
height = 224

corrMask = cv2.imread("correctionMat.png") / 100.0
gpsData =  {"id": gpsId, "gpsX": -1, "gpsY": -1,}

#direction
left, right = (False, False)        #default direction -> straight
serverL, serverR = (False, False)
lastL, lastR = (False, False)
serverUpdated = False

forced = 0

#steering
lastSteer = 0.0

#throttle
serverSpeed = 0.0
lastSpeed = 0.0
carSpeed = -0.25
signAction = ""


#start camera
camera = CSICamera(width=398, height=224, framerate=30)
camera.running = True

#car
car = NvidiaRacecar()

#yolo
model = init()
n = 0

#car specific:
carLoff, carRoff = (158, 138)   #white line offset
caryL, caryR = (168, 146)       #yellow

src=np.float32([[42, 112], [0,146], [398, 146], [352, 112]])
dst=np.float32([[0,0],[0,224],[398, 224],[398,0]])

M = cv2.getPerspectiveTransform(src, dst)

#gps
http = urllib3.PoolManager()
server = 'http://192.168.1.249:8080/get_car_pos'

#node globals
whiteNot = 0
yellowNot = 0
yellowCount = 0
nodeFound = False
nodePassed = False

#image processing
def white_mask(image):
    lower = np.array([210, 217, 0])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    return mask

def yellow_mask(image):
    lower = np.array([0, 214, 0])
    upper = np.array([192, 255, 255])
    mask = cv2.inRange(image, lower, upper)    
    return mask

def get_frame(cam, M):
    """
        cam -> camera object
        M -> transformation matrix
    """
    global corrMask, frame, n
    frame = cam.value.copy()
    warp = cv2.warpPerspective(frame.copy(), M, (width,height), flags=cv2.INTER_LINEAR)
    
    #correct colors of image
    warp = (warp * corrMask)
    warp[warp>255]=255
    warp = warp.astype(np.uint8)
    n+=1
    return warp

def mask_slice(maskS, white=True):
    """
        white = False for yellow sliceing
    """
    global mid, midOffset
    
    if white:
        left = maskS[:, :int(mid-midOffset)]
        right = maskS[:, int(mid+midOffset):]
        midM = maskS[:, int((mid/2)):int(mid+(mid/2))]

        return left, midM, right

    else:
        right = maskS[:, 150:]
        left = maskS[:, :-150]

        return left, right    
    

def parse_steering(x, bools, mid=mid, l_offset=carLoff, r_offset=carRoff):
    """
        x -> list from sliding window
        mid -> line middle location
        l/r_offset -> offset from left/right line to center
        bools -> left, right, mid        
    """
    global carSpeed, lastSteer
    
    l, r, m = bools
    steer = 0
    
    if l or r:
        if l:
            lane_mid = np.median(x)+l_offset
        elif r:
            lane_mid = np.median(x)-r_offset
        steer = ((mid-lane_mid)/(mid))
    elif m:
        if (np.mean(x)<x[-1]):
            steer = -0.7
        else:
            steer = 0.7
    #clamp
    steer = min(0.72, max(-0.72, steer))
    
    if carSpeed < -0.25:
        steer = steer*0.65
    else:
        steer = (lastSteer + steer)/2
    lastSteer = steer   
    
    return steer


def line_check(img, n = 130, upThr = 32000, lowthr = 2500):
    """
        
        img -> image mask
        n -> minimum line height
    """
    height = 0
    y = list()
    hist_h = np.sum(img, axis=1)
    
    for i in range(len(hist_h)):
        if hist_h[i]>lowthr and hist_h[i]<upThr:
            height+=1
            y.append(i)
            
    l = False if height<n else True
    
    if l and len(y)>2:
        y1 = np.min(y)
        y2 = np.max(y)
    else:
        y1 = 0
        y2 = img.shape[0]
    
    return l, y1, y2, height


def sliding_one(img, nwindows=9, margin=150, minpix = 1, offset=0, path=(False, False)):
    """
        offset -> mask offset from x=0
        skretanje lijevo -> True, False
    """
    l, r = path
    fitted = False
    #calculate global histogram
    window_height = np.int(img.shape[0]/nwindows)
    histogram = np.sum(img, axis=0)
    
    #first window starting point
    base = np.argmax(histogram)
    x_current = base
    
    #nz
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Create empty lists to receive left and right lane pixel indices
    x_coords = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        x_coords.append(offset+x_current)        
            
        #next window starting position
        if l:   #force left steering
            x_h = x_current
            x_l = 0 if int(x_h-(margin*2)) < 0 else int(x_h-(margin*3))
        elif r:
            x_l = x_current
            x_h = img.shape[1] if int(x_l+(margin*2)) > img.shape[1] else int(x_l+(margin*3))
        else:
            x_l = 0 if int(win_x_low-margin) < 0 else int(win_x_low-margin)
            x_h = img.shape[1] if int(win_x_high+margin) > img.shape[1] else int(win_x_high+margin)
        
        
        x_h = int(x_h)
        x_l = int(x_l)
        next_hist = np.sum(img[(win_y_low-window_height):(win_y_low), x_l:x_h], axis=0)
        
        if np.sum(next_hist)>2550:
            hig = np.argmax(next_hist)
            x_current = x_l+hig
            fitted = True

    return x_coords, fitted


def get_steering():
    global server, gpsData, frame, model
    response, throttle, steering, action = (None, None, None, None)
    
    #read position from gps
    pos = hedge.position()
    if (pos[1]!=0 and pos[2]!=0):
        gpsData = {"id": gpsId, "gpsX": pos[1], "gpsY" :pos[2]}
    
    #send request to server
    try:
        response = http.request("GET",
                        server, 
                        headers={"Content-Type": "application/json"},
                        body= json.dumps(gpsData))
    except: 
        return None, None
        
    if response.status != 200:
        return None, None

    #process response
    try:
        carInfo = response.data.decode("utf-8")
        carInfo = json.loads(carInfo)
        throttle = float(carInfo["throttle"])    
        steering = int(carInfo["steering"][0])
    except:
        print("Server response error!")
    return (steering, throttle)


def get_direction():
    global serverL, serverR, serverUpdated, serverSpeed
    n = 0
    while True:
        s, t = get_steering()       
        #process steering
        if s is not None:
            if s==-1:
                serverL, serverR = (False, True)
            elif s==1:
                serverL, serverR = (True, False)
            else:
                serverL, serverR = (False, False)
            
            f = False
            serverUpdated = True
            
        if t is not None:
            serverSpeed = t


def yolo_thread():
    global n, carSpeed, lastSpeed, signAction, serverSpeed, model, frame
    fr = np.zeros((398, 224,3), dtype=np.uint8)
    while True:
        if n%2 == 0:
            fr = frame.copy()
            detect_objects(model, fr)           
            action, speed = take_action(fr)

            if speed is not None:
                if speed == "slow":
                    carSpeed = -0.25
                elif speed == "fast":
                    carSpeed = -0.34
                    
            if action is not None:
                if action == "ped":
                    lastSpeed = carSpeed
                    carSpeed = 0.0
                    
                elif action == "stop":
                    lastSpeed = carSpeed
                    carSpeed = 0.0
                    sleep(4)
                    carSpeed = lastSpeed

                elif action == "red":
                    lastSpeed = carSpeed
                    carSpeed = 0.0

                elif action == "green":
                    if signAction == "red":
                        carSpeed = lastSpeed
                        
                elif action == "stopCar":
                    lastSpeed = carSpeed
                    carSpeed = 0.0
                    
                elif action == "" and (signAction == "stopCar" or signAction == "ped"):
                    carSpeed = lastSpeed
                    
                signAction = action
        
        #stop on server route end
        if carSpeed == 0.0 and serverSpeed < 0.0 and signAction != "red" and signAction != "ped":
            carSpeed = -0.25
    
        if serverSpeed == 0.0 and not (signAction == "stopCar" or signAction == "ped"):
            carSpeed = 0.0
        
        sleep(0.016)


#gps
hedge = MarvelmindHedge(tty = "/dev/gps", adr=30, debug=False) # create MarvelmindHedge thread
hedge.start()

#server
thrGps = threading.Thread(target=get_direction)
thrGps.start()

#yolo
thrYolo = threading.Thread(target=yolo_thread)
thrYolo.start()


while True:
    #reset var
    winFitted = False
    
    #get new frame
    warp = get_frame(camera, M)
    
    #get masks
    yellowMask = yellow_mask(warp)
    whiteMask = white_mask(warp)
    
    #slice masks
    whiteLeft, whiteMid, whiteRight = mask_slice(whiteMask, True)
    yellowLeft, yellowRight = mask_slice(yellowMask, False)

    #check for white lines  n -> minimum line height
    wL, wlY1, wlY2, hL = line_check(whiteLeft, n=35)
    wR, wrY1 ,wrY2, hR = line_check(whiteRight, n=35) 
    wM, wmY1, wmY2, _ = line_check(whiteMid, n=30)   
    yL, ylY1 ,ylY2, yhL = line_check(yellowLeft, n=25)
    yR, yrY1 ,yrY2, yhR = line_check(yellowRight, n=25)
    
        
    if (not left and not right) or (nodeFound and yellowNot>1) or yellowNot > 10:
        #update last l/r
        if left or right and not stForced:
            lastL, lastR = left, right

        #update l/ for next node
        left, right = (serverL, serverR)
        
        #reset node vars
        nodeFound = False
        stForced = False
        yellowNot = 0
        yellowCount = 0
    
    #----yellow lines----#
    if (left or right) or (whiteNot > whiteFB):
        #fallback to yellow lines
        if whiteNot > whiteFB and not (left or right):
            right = True

        if right and yL:
            xl, winFitted = sliding_one(yellowLeft[ylY1:ylY2, :], nwindows=4, margin=25, minpix = 1, path=(left, right))
            if winFitted:
                steer = parse_steering(xl, (True, False, False), l_offset=caryL, r_offset=0)
        
        elif left and yR:
            xr, winFitted = sliding_one(yellowRight[yrY1:yrY2, :], nwindows=4, margin=25, minpix = 1, offset = 140, path=(left,right))
            if winFitted:
                steer = parse_steering(xr, (False, True, False), l_offset=0, r_offset=caryR)            

        #winFitted = found yellow lane on this frame (True/False)
        if winFitted:
            nodePassed=False
            yellowNot = 0
            yellowCount += 1
            
        else:
            yellowNot += 1
            #if node was found set nodePassed as true
            if nodeFound and yellowNot > 1:
                nodePassed = True
                forced = 0

        #set node as found and reset not found counter
        if not nodeFound and yellowCount > 4:
            nodeFound = True    #found node
            yellowNot = 0       #reset counter
    
    if (wL and wR) or forced>15:
        nodePassed = False
        
    #----white lines----#
    if nodePassed and wM and (not left and not right) :  #force lane found as (left or right)
        xm, winFitted = sliding_one(whiteMid[wmY1:wmY2, :], nwindows=4, margin=25, minpix = 1, offset =  int((mid/2)))
        steer = parse_steering(xm, (lastR, lastL, False))
        forced+=1
        
    elif not winFitted or (not left and not right):
        if wL and (hL >= hR+10 or not wR):
            xl, winFitted = sliding_one(whiteLeft[wlY1:wlY2, :], nwindows=4, margin=25, minpix = 1)
            if winFitted:
                steer = parse_steering(xl, (True, False, False))
            
        elif wR:
            xr, winFitted = sliding_one(whiteRight[wrY1:wrY2, :], nwindows=4, margin=25, minpix = 1, offset = mid+midOffset)
            if winFitted:
                steer = parse_steering(xr, (False, True, False))
                
        elif wM:
            xm, winFitted = sliding_one(whiteMid[wmY1:wmY2, :], nwindows=4, margin=25, minpix = 1, offset = int((mid/2)))
            if winFitted:
                steer = parse_steering(xm, (False, False, True))
        else:
            whiteNot += 1
            steer = lastSteer
        
        #reset white counter
        if winFitted:
            whiteNot = 0
        else:
            steer = lastSteer
    
    car.steering = steer
    car.throttle = -0.24