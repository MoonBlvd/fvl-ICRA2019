import numpy as np
# import cv2

def bbox2pixel_traj(bbox,predict_diff=True,prev_box=None,W=1280,H=640):
        '''
        for plotting purpose
        [cx,cy,w,h]
        '''
        if predict_diff:
            traj = np.vstack([prev_box[:,:4],bbox[:,:4]+prev_box[:,:4]])
        else:
            if prev_box is None:
                traj = bbox[:,:4]
            else:
                traj = np.vstack([prev_box[:,:4],bbox[:,:4]])
        traj[:,0] *= W
        traj[:,1] *= H
        traj[:,2] *= W
        traj[:,3] *= H

        return traj

# def prepare_img_for_show(img_path, W, H):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     in_img = False
#     for i in range(img.shape[0]):
#         if np.mean(img[i,:,:]) < 255 and not in_img:
#             ymin = i
#             in_img = True
#             break
#
#     in_img = False
#     for i in range(img.shape[0]-1,0,-1):
#         if np.mean(img[i,:,:]) < 255 and not in_img:
#             ymax = i
#             in_img = False
#             break
#
#     in_img = False
#     for j in range(img.shape[1]):
#         if np.mean(img[:,j,:]) < 255 and not in_img:
#             xmin = j
#             in_img = True
#             break
#
#     in_img = False
#     for j in range(img.shape[1]-1,0,-1):
#         if np.mean(img[:,j,:]) < 255 and not in_img:
#             xmax = j
#             in_img = True
#             break
#     print(ymin,ymax, xmin,xmax)
#     img = img[ymin:ymax, xmin:xmax,:]
#     img = cv2.resize(img, (W,H))
#
#     return img

def pred_to_box(pred, prev_box=None, W=1280, H=640):
    
    pred[:,[0,2]]  = pred[:,[0,2]] * W
    pred[:,[1,3]]  = pred[:,[1,3]] * H
    
    init_box = prev_box
    init_box[:,[0,2]]  = init_box[:,[0,2]] * W
    init_box[:,[1,3]]  = init_box[:,[1,3]] * H
    
    if prev_box is not None:
        traj = np.vstack([init_box[:,:2],pred[:,:2]+init_box[:,:2]])
    else:
        traj = bbox[:,:2]
    
    

# def add_attention_mask(img, attention_map):
#     '''
#     plot attention mask on the img
#     params: attention_map: shape (800,1)
#             img: shape(640,1280,3)
#     '''
#     num_cells = attention_map.shape[1]
#     if num_cells == 100:
#         heatmap = np.reshape(attention_map,(10,10))
#     elif num_cells == 50:
#         heatmap = np.reshape(attention_map,(5,10))
#     else:
#         raise ValueError("attention map size is unknown!!")
#     heatmap = cv2.resize(heatmap, (1280,640))
#     heatmap = heatmap / np.max(heatmap)
#
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET),cv2.COLOR_RGB2BGR)
#     superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#
#     return superimposed_img

def compute_IOU(bbox_true, bbox_pred, W, H):
    '''
    compute IOU
    [cx, cy, w, h]
    '''
    xmin = np.max([bbox_true[0] - bbox_true[2]/2, bbox_pred[0] - bbox_pred[2]/2]) * W
    xmax = np.min([bbox_true[0] + bbox_true[2]/2, bbox_pred[0] + bbox_pred[2]/2]) * W
    ymin = np.max([bbox_true[1] - bbox_true[3]/2, bbox_pred[1] - bbox_pred[3]/2]) * H
    ymax = np.min([bbox_true[1] + bbox_true[3]/2, bbox_pred[1] + bbox_pred[3]/2]) * H

    w_inter = np.max([0, xmax - xmin])
    h_inter = np.max([0, ymax - ymin])
    intersection = w_inter * h_inter
    union = (bbox_true[2]*bbox_true[3] + bbox_pred[2]*bbox_pred[3]) * W * H - intersection

    return intersection/union