import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import cv2
import pickle as pkl
import os
import glob
import copy
import pickle as pkl

W,H = 1242,375


def bbox_normalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h]m size (times, 4)
    :Return:
        bbox
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H
    
    return new_bbox

def static_object(data):
    moving_thresh = 20
    size_thresh = 10
    seq = np.vstack([data['bbox'],data['target']])
    seq[:,0] *= W
    seq[:,1] *= H
    seq[:,2] *= W
    seq[:,3] *= H
        
    if np.std(seq,axis=0)[0] < moving_thresh and \
       np.std(seq,axis=0)[1] < moving_thresh and \
       np.std(seq,axis=0)[2] < size_thresh and \
       np.std(seq,axis=0)[3] < size_thresh:
        return True
    else:
        return False

def sequence_segmentation(seq, save_dir, input_len=10, target_len=10, shift=True):
    '''
    Given a sequence of features, randomly select 10 input frames and 10 target frames.
    :params: 
        seq: a dictionary with each key as one featrue (time_steps,)
    :return:
        x: (input_len, dim)
        y: (target_len, 2)
    '''
    stride = 1#int(input_len/2)
    total_len = len(seq['bbox'])
    x = {}
    
    # random select a start point to clip the sequence
    lb = 0
    ub = np.min([10,total_len-input_len-target_len+1])
    init_start = lb#np.random.randint(lb,ub)
    # clip the sequence with a gievn stride length
    i = 0
    
    for start in range(init_start, total_len-input_len-target_len+1, stride):
        for key,value in seq.items():
#             start = np.random.randint(0,total_len-input_len-target_len+1)
            end = start + input_len + target_len
            x[key] = value[start:start+input_len]
        
        # shift and transform ego motion
        if shift:
            '''shift the odometry data to the first frame of a sequence'''
            new_odo = shift_odo_data(np.array(seq['ego_motion'][start:end]))
            if new_odo.shape[1] == 4:
                new_ego_motion = vector_from_T(new_odo)
                x['ego_motion'] = new_ego_motion[:input_len]
            elif new_odo.shape[1] == 6:
                x['ego_motion'] = new_odo
            else:
                raise ValueError("Wrong odo data shape!")
        else:
            '''use global angle and shifted distance'''
            new_odo = seq['ego_motion'][start:end]
            
            
        
        x['target'] = seq['bbox'][start+input_len:end,:]
        
        if static_object(x) and np.random.rand() < 0: #0.75: # 75% prob to skip static sequence
            # skip the static sequence
            print("a static sequence is skipped!")
            continue
        else:
            x['future_ego_motion'] = new_ego_motion[input_len:input_len+target_len]
        i += 1
        save_path =  save_dir + '_' + str(format(i,'04')) + '.pkl'
        pkl.dump(x, open(save_path, 'wb'), protocol=2)
    return x


'''Load dense optical flow'''
def load_flow(flow_folder):
    '''
    Given video key, load the corresponding flow file
    '''
    
    flow_files = sorted(glob.glob(flow_folder + '*.flo'))
    flows = []
    for file in flow_files:
        flow = read_flo(file)
        flows.append(flow)
    return flows

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow

def load_vo(odo_file):
    Tcws = []
    num_matched_odo = 0
    
    with open(odo_file) as file:
        num_lines = sum(1 for line in open(odo_file))
        start = None
        end = None
        for idx, line in enumerate(file):
            matrix = np.zeros((4,4))
            if line is '\n':
                if start is None:
                    start = idx
                    end = None
            else:
                end = idx
                for i, element in enumerate(line.split()):
                    matrix[int(i/4), int(i%4)] = float(element)
            Tcws.append(matrix)

            if start is not None and (end is not None or idx+1 == num_lines):
                if end is None:
                    end = idx+1
#                 print("fixing lines from " + str(start+1) + " to " + str(end+1))
                Tcws = fix_odo_data(start, end, num_lines, Tcws)
                start = None
    return Tcws

def fix_odo_data(start, end, num_lines, Tcws):
    '''There are some missed values so that the vo data needs to be fixed.'''
    if start == 0:
        for i in range(end):
            Tcws[i] = np.eye(4)
    else:
#         if end == num_lines:
        derivative = np.linalg.inv(Tcws[start-2]).dot(Tcws[start-1])
#         else:
#             derivative = (Tcws[end] - Tcws[start-1]) / (end-start)
        for i in range(start, end):
                Tcws[i] = Tcws[i-1].dot(derivative)
    return Tcws

def shift_odo_data(Tcws):
    '''
    project a set of Tcws to the first frame
    :Params:
        Tcws: a (time, 4, 4) numpy array of transformation matrics, 
        or (time,6) with x
    :Return:
        new_Tcws: a (time, 4, 4) numpy array of shifted transformation matrics
    '''
    if Tcws.shape[1] == 4:
        compensation = np.linalg.inv(Tcws[0])
        for i in range(Tcws.shape[0]):
            Tcws[i] = compensation.dot(Tcws[i])
    elif Tcws.shape[1] == 6:
        Tcws = Tcws-Tcws[0,:]
    else:
        raise ValueError("Wrong odometry data shape!")
        
    return Tcws

def vector_from_T(Tcws):
    '''
    Compute ego motion vector from given tranformation matrix
    :Params:
        Tcws: a (time, 4, 4) numpy array of transformation matrics
    :Return:
        ego_motion: a (time,6) numpy array of ego motion vector [yaw, pitch, roll, x, y, z]
    '''
    ego_motion = np.zeros((Tcws.shape[0],6))
    for i in range(Tcws.shape[0]):
        sy = np.sqrt(Tcws[i][0,0] * Tcws[i][0,0] +  Tcws[i][1,0] * Tcws[i][1,0])
        
        singular = sy < 1e-6
 
        if  not singular :
    #         yaw = -np.arcsin(Tcws[i][2,0]) # ccw is positive, in [-pi/2, pi/2]
            yaw = np.arctan2(-Tcws[i][2,0], sy) # ccw is positive, in [-pi/2, pi/2]
            pitch = np.arctan2(Tcws[i][2,1], Tcws[i][2,2]) # small value, in [-pi/2, pi/2]
            roll = np.arctan2(Tcws[i][1,0], Tcws[i][0,0]) # small value, in [-pi/2, pi/2]
        else:
            yaw = np.atan2(-Tcws[i][2,0], sy)
            pitch = np.atan2(-Tcws[i][1,2], Tcws[i][1,1])
            roll = 0
        
        ego_motion[i,0] = yaw
        ego_motion[i,1] = pitch
        ego_motion[i,2] = roll
        
        ego_motion[i,3:] = -Tcws[i,:3,3]     
        
    return ego_motion
    

def shrink_flow_box(bbox):
    '''
        Shrink the bbox and compute the mean optical flow
        :Param: flow size is (h,w,2) containing two direction dense flow
        :Param: bbox format is:  [cx,cy,w,h]
        :return: [cx,cy,w,h]
    '''
    bbox_shrink = 1.2#0.8
#     cx = (bbox[1]+bbox[3])/2
#     cy = (bbox[0]+bbox[2])/2
#     w = bbox[3]-bbox[1]
#     h = bbox[2]-bbox[0]
    cx = bbox[0]
    cy = bbox[1]
    w = bbox[2]
    h = bbox[3]

    
    return np.array([[cx,cy,w*bbox_shrink,h*bbox_shrink]])

def roi_pooling_opencv(boxes,image,size=[5,5],W=1280,H=640):
    """Crop the image given boxes and resize with bilinear interplotation.
    :Params:
        image: Input image of shape (1, image_height, image_width, depth)
        boxes: Regions of interest of shape (num_boxes, 4), must be [0,1] in float
        each row [cx, cy, w, h]
        size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    :Returns:
        4D Tensor (number of regions, slice_height, slice_width, channels)
    """

    w = image.shape[2]
    h = image.shape[1]
    
    xmin = boxes[:,0]-boxes[:,2]/2
    xmax = boxes[:,0]+boxes[:,2]/2
    ymin = boxes[:,1]-boxes[:,3]/2
    ymax = boxes[:,1]+boxes[:,3]/2 
    

    ymin = np.max([0,int(h * ymin / H)])
    ymax = int(h * ymax / H)
    
    xmin = np.max([0, int(w * xmin / W)])
    xmax = int(w * xmax / W)
    size = (size[0],size[1])

    return np.expand_dims(cv2.resize(image[0,ymin:ymax, xmin:xmax,:], size),axis=0)
    
class ViewpointExtractor():
    def __init__(self, weight_file):
        self.model = Resnet_fc(input_shape=(224, 224, 3), num_classes=5)
        self.model.load_weights(weight_file)
        self.model.layers.pop()
        self.model.layers.pop()  # remove the top layers, final output is 1*2048

    def get_one_feature(self, bbox, image):
        bbox = bbox.astype('int')
        xmin = int(bbox[0] - bbox[2] / 2)
        xmax = int(bbox[0] + bbox[2] / 2)
        ymin = int(bbox[1] - bbox[3] / 2)
        ymax = int(bbox[1] + bbox[3] / 2)
        patch = image[ymin:ymax, xmin:xmax, :]
        patch = cv2.resize(patch, (224, 224))

        patch = np.expand_dims(patch, axis=0)
        self.model.predict(patch)
        return feature

    def run(self, images, all_bboxes):
        '''all_bboxes is (), [xc, yc, w, h]'''
        all_features = []
        for image, bbox in zip(images, all_bboxes):
            feature = self.get_one_feature(bbox, image)
            all_features.append(feature)

        print(np.array(all_features).shape)
        return np.array(all_features)