import numpy as np
import pickle as pkl
from keras.callbacks import TensorBoard, Callback
from models.const_models import *
from utils.eval_utils import *

def load_cases(DATA_DIR):
    '''
    Load the keys for easy and challenge cases
    :param DATA_DIR:
    :return:
    '''
    easy_cases = []
    challenge_cases = []

    file = open(DATA_DIR + 'challenge_cases_auto.txt', 'r')
    for line in file:
        challenge_cases.append(DATA_DIR + 'val_10/' + line[:-1])
    print(len(challenge_cases))

    file = open(DATA_DIR + 'easy_cases_auto.txt', 'r')
    for line in file:
        easy_cases.append(DATA_DIR + 'val_10/' + line[:-1])
    print(len(easy_cases))

    return easy_cases, challenge_cases

class Generator():
    def __init__(self, train_keys, val_keys, 
                 inputs, zero_input_shape=256,
                 predict_diff = True,
                 FEATURE_DIR = '../feature_output/', 
                 batch_size=64, time_steps=10, pred_time_steps=10):
        '''
        x is a dictionary with each item corresponds to one sequence.
        '''
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.time_steps = time_steps
        self.batch_size=batch_size
        self.inputs = inputs
        self.pred_time_steps = pred_time_steps
        self.FEATURE_DIR = FEATURE_DIR
        self.zero_input_shape = zero_input_shape
        self.predict_diff=predict_diff

    def generate(self, train=True):
        while True:
            if train:
                batch_idxs = np.random.permutation(len(self.train_keys))[:self.batch_size]
            else:
                batch_idxs = np.random.permutation(len(self.val_keys))[:self.batch_size]
            
            bbox = []
            depth = []
            flow = []
            ego_motion = []
            y = []
            all_generated_inputs = {}
            for j in batch_idxs:
                if train:
                    key = self.train_keys[j]
                    video_key = key[63:81]
                else:
                    key = self.val_keys[j]
                    video_key = key[61:79]
                
                data = pkl.load(open(key,'rb'))
                
                if len(y) == 0:
                    bbox = np.expand_dims(data['bbox'][:self.time_steps],axis=0)

                    if 'depth' in self.inputs:
                        depth = np.expand_dims(np.expand_dims(data['depth_map'][:self.time_steps],axis=0),axis=-1)
                    if 'flow' in self.inputs:
                        flow = np.expand_dims(data['flow'][:self.time_steps],axis=0)
                    if 'ego_motion' in self.inputs:
                        ego_motion = np.expand_dims(data['ego_motion'][:self.time_steps],axis=0)
                    if 'future_ego_motion' in self.inputs:
                        future_ego_motion = np.vstack([future_ego_motion, 
                                                   np.expand_dims(data['future_ego_motion'][:self.time_steps],axis=0)])
                    if 'zero' in self.inputs:
                        zero_input = np.zeros([1,1,self.zero_input_shape])
                    if 'const_vel' in self.inputs:
                        const_vel = const_vel_predict(data['bbox'][:self.time_steps])
                    if 'feature_map' in self.inputs:
                        feature_map_input = np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][-1], '06') + '.npy')
                    if 'past_feature_map' in self.inputs:
                        past_feature_map_input = np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][0], '06') + '.npy')
                        for i in range(1, self.time_steps):
                            past_feature_map_input = np.vstack([past_feature_map_input, np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][i], '06') + '.npy')])
                        past_feature_map_input = np.expand_dims(past_feature_map_input,axis=0)
                            
                    new_target = np.vstack([data['bbox'][self.time_steps:],
                                           data['target'][:self.time_steps + self.pred_time_steps - 10]])
                    
#                     y = np.expand_dims(new_target, axis=0)
#                     y = np.vstack([data['bbox'][self.time_steps-1:], 
#                                    new_target - data['bbox'][self.time_steps-1]])
#                     y = np.expand_dims(y, axis=0)
                    if self.predict_diff:
                        y = np.expand_dims(new_target - data['bbox'][self.time_steps-1],axis=0)
                    else:
                        y = np.expand_dims(new_target,axis=0)
                        
                    # targets for regress encoder output
                    y_enc = np.expand_dims(np.vstack([data['bbox'][1:], data['target'][0]]) - data['bbox'][0], axis=0)

                else:
                    bbox = np.vstack([bbox, np.expand_dims(data['bbox'][:self.time_steps],axis=0)])
                    if 'depth' in self.inputs:
                        depth = np.vstack([depth, np.expand_dims(np.expand_dims(data['depth_map'][:self.time_steps],axis=0),axis=-1)])
                    if 'flow' in self.inputs:
                        flow = np.vstack([flow, np.expand_dims(data['flow'][:self.time_steps],axis=0)])
                    if 'ego_motion' in self.inputs:
                        ego_motion = np.vstack([ego_motion, np.expand_dims(data['ego_motion'][:self.time_steps],axis=0)])
                    if 'future_ego_motion' in self.inputs:
                        future_ego_motion = np.vstack([future_ego_motion, 
                                                   np.expand_dims(data['future_ego_motion'][:self.time_steps],axis=0)])

                    if 'zero' in self.inputs:
                        zero_input = np.vstack([zero_input, np.zeros([1,1,self.zero_input_shape])])
                    if 'const_vel' in self.inputs:
                        const_vel = np.vstack([const_vel, const_vel_predict(data['bbox'][:self.time_steps])])
                    if 'feature_map' in self.inputs:
                        feature_map = np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][-1], '06') + '.npy')
                        
                        feature_map_input = np.vstack([feature_map_input, feature_map])
                    if 'past_feature_map' in self.inputs:
                        temp_past_feature_map = np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][0], '06') + '.npy')
                        for i in range(1,self.time_steps):
                            temp_past_feature_map = np.vstack([temp_past_feature_map, np.load(self.FEATURE_DIR + video_key + '/' + format(data['frame_id'][i], '06') + '.npy')])
                        temp_past_feature_map = np.expand_dims(temp_past_feature_map,axis=0)
#                         print(temp_past_feature_map.shape)
#                         print(past_feature_map_input.shape)
                           
                        past_feature_map_input = np.vstack([past_feature_map_input,temp_past_feature_map])
                            
                        
                    new_target = np.vstack([data['bbox'][self.time_steps:],
                                           data['target'][:self.time_steps + self.pred_time_steps - 10]])
                    
#                     y_tmp = np.vstack([data['bbox'][self.time_steps-1:], 
#                                        new_target - data['bbox'][self.time_steps-1]])
#                     y_tmp = np.expand_dims(y_tmp, axis=0)
                    if self.predict_diff:
                        y_tmp = np.expand_dims(new_target - data['bbox'][self.time_steps - 1], axis=0)
                    else:
                        y_tmp = np.expand_dims(new_target, axis=0)
                    y = np.vstack([y, y_tmp])
            
                    # targets for regress encoder output
                    y_enc_tmp = np.expand_dims(np.vstack([data['bbox'][1:], data['target'][0]]) - data['bbox'][0], axis=0)
                    y_enc = np.vstack([y_enc, y_enc_tmp])

            
            all_generated_inputs['bbox'] = bbox
            if 'flow' in self.inputs:
                all_generated_inputs['flow'] = flow
            if 'depth' in self.inputs:
                all_generated_inputs['depth'] = depth
            if 'ego_motion' in self.inputs:
                all_generated_inputs['ego_motion'] = ego_motion
            if 'future_ego_motion' in self.inputs:
                        all_generated_inputs['future_ego_motion'] = future_ego_motion
            if 'feature_map' in self.inputs:
                all_generated_inputs['feature_map'] = feature_map_input
            if 'past_feature_map' in self.inputs:
                all_generated_inputs['past_feature_map'] = past_feature_map_input
            if 'zero' in self.inputs:
                all_generated_inputs['zero'] = zero_input
            if 'const_vel' in self.inputs:
                all_generated_inputs['const_vel'] = const_vel
            
            generate_inputs = [all_generated_inputs[name] for name in self.inputs]
            
            yield generate_inputs, y

class TestCallbacks(Callback):
    def __init__(self, inputs, val_keys,
                 zero_input_shape=256, 
                 FEATURE_DIR = '../feature_output/',
                 predictor='single_input', case='easy cases'):
        self.inputs = inputs
        self.val_keys = val_keys
        self.predictor = predictor
        self.case = case
        self.zero_input_shape = zero_input_shape
        self.FEATURE_DIR = FEATURE_DIR
        self.FDE = []
        self.MDE = []
        self.FIOU = []
        
    def on_epoch_end(self, epoch, logs={}):
        W = 1280
        H = 640
        all_traj_displacement = []
        all_end_displacement = []
        all_gt_traj = []
        all_pred_traj = []
        all_IOU = []
        time_steps = 10
        pred_time_steps = 10

        if epoch%5 == 0:

            for val_file in self.val_keys:
                val_trajectory = pkl.load(open(val_file,'rb'))
                video_key = val_file[61:79]

                '''predict'''
                if self.predictor == 'single_input':
                    predict_input = []
                    for name in self.inputs:
                        if name == 'depth':
                            predict_input.append(np.expand_dims(np.expand_dims(val_trajectory[name+'_map'][:time_steps],0),-1))
                        elif name == 'const_vel':
                            predict_input.append(const_vel_predict(val_trajectory['bbox'][:time_steps],
                                                                   pred_time_steps = pred_time_steps))
                        elif name == 'zero':
                            predict_input.append(np.zeros([1,1,self.zero_input_shape]))
                        elif name == 'feature_map':
                            predict_input.append(np.load(self.FEATURE_DIR + video_key + '/' + format(val_trajectory['frame_id'][-1], '06') + '.npy'))
                        elif name == 'past_feature_map':
                            past_feature_map_input = np.load(self.FEATURE_DIR + video_key + '/' + format(val_trajectory['frame_id'][0], '06') + '.npy')
                            for i in range(1, time_steps):
                                past_feature_map_input = np.vstack([past_feature_map_input, np.load(self.FEATURE_DIR + video_key + '/' + format(val_trajectory['frame_id'][i], '06') + '.npy')])
                            past_feature_map_input = np.expand_dims(past_feature_map_input,axis=0)
                            predict_input.append(past_feature_map_input)
                        else:
                            predict_input.append(np.expand_dims(val_trajectory[name][:time_steps],0))
                    if 'feature_map' in self.inputs:
                        pred = self.model.predict(predict_input)
                    else:
                        pred = self.model.predict(predict_input)

                if self.predictor == 'dual_input':
                    predict_input = []
                    for name in self.inputs:
                        if name == 'depth':
                            depth_0 = np.expand_dims(np.expand_dims(val_trajectory['depth_map'][:time_steps-1],axis=0),axis=-1)
                            depth_1 = np.expand_dims(np.expand_dims(val_trajectory['depth_map'][1:time_steps],axis=0),axis=-1)
                            new_depth = np.concatenate([depth_0,depth_1], axis=-1)
                            predict_input.append(new_depth)
                        elif name == 'const_vel':
                            predict_input.append(const_vel_predict(val_trajectory['bbox'][:time_steps],
                                                            pred_time_steps = pred_time_steps))
                        elif name == 'zero':
                            predict_input.append(np.zeros([1,1,self.zero_input_shape]))
                        elif name == 'feature_map':
                            predict_input.append(np.load(self.FEATURE_DIR + video_key + '/' + format(val_trajectory['frame_id'][-1], '06') + '.npy'))
                        else:
                            input_0= np.expand_dims(val_trajectory[name][:time_steps-1],axis=0)
                            input_1= np.expand_dims(val_trajectory[name][1:time_steps],axis=0)
                            new_input = np.concatenate([input_0,input_1], axis=-1)
                            predict_input.append(new_input)
                    pred,attention_map = self.model.predict(predict_input)#

                '''find trajectory'''
                new_target = np.vstack([val_trajectory['bbox'][time_steps:],
                                        val_trajectory['target'][:time_steps + pred_time_steps - 10]])

                gt_traj = bbox2pixel_traj(new_target-val_trajectory['bbox'][time_steps-1,:],
                                           predict_diff = True,
                                           prev_box = np.expand_dims(val_trajectory['bbox'][time_steps-1,:],0),
                                           W = W,
                                           H = H)

                pred_traj = bbox2pixel_traj(pred[0],
                                             predict_diff = True,
                                             prev_box = np.expand_dims(val_trajectory['bbox'][time_steps-1,:],0),
                                             W = W,
                                             H = H)
    #             gt_traj = bbox2pixel_traj(new_target,
    #                                            predict_diff = False,
    #                                            prev_box = np.expand_dims(val_trajectory['bbox'][time_steps-1,:],0),
    #                                            W = W,
    #                                            H = H)

    #             pred_traj = bbox2pixel_traj(pred[0], #- val_trajectory['bbox'][time_steps-1],
    #                                              predict_diff = False,
    #                                              prev_box = np.expand_dims(val_trajectory['bbox'][time_steps-1,:],0),
    #                                              W = W,
    #                                              H = H)

                end_dist = np.sqrt(np.sum((gt_traj[-1,:2] - pred_traj[-1,:2])**2))
                all_end_displacement.append(end_dist)

                mean_displacement = np.mean(np.sqrt(np.sum((gt_traj[1:,:2] - pred_traj[1:,:2])**2, axis=-1)))
                all_traj_displacement.append(mean_displacement)

                final_IOU = compute_IOU(gt_traj[-1,:], pred_traj[-1,:], W, H)
                all_IOU.append(final_IOU)

            FDE = np.mean(np.array(all_end_displacement))
            MDE = np.mean(np.array(all_traj_displacement))
            FIOU = np.mean(np.array(all_IOU))

            self.FDE.append(FDE)
            self.MDE.append(MDE)
            self.FIOU.append(FIOU)

            print("Case: ", self.case)
            print("FDE: ", np.mean(np.array(all_end_displacement)))
            print("MDE: ", np.mean(np.array(all_traj_displacement)))
            print("Final IOU: ", np.mean(np.array(all_IOU)))


# class TestCallbacks(Callback):
#     def __init__(self, inputs, val_keys,
#                  zero_input_shape=256,
#                  FEATURE_DIR='../feature_output/',
#                  predictor='single_input', case='easy cases'):
#         self.inputs = inputs
#         self.val_keys = val_keys
#         self.predictor = predictor
#         self.case = case
#         self.zero_input_shape = zero_input_shape
#         self.FEATURE_DIR = FEATURE_DIR
#         self.FDE = []
#         self.MDE = []
#         self.FIOU = []
#
#     def on_epoch_end(self, epoch, logs={}):

    
