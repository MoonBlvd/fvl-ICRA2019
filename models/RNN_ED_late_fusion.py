import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector, Lambda, TimeDistributed, RepeatVector
from keras.layers import Conv1D, Conv2DTranspose, Lambda, AveragePooling2D, Activation, Add
from keras.layers import Flatten, BatchNormalization, Reshape, Concatenate, Average
from keras.activations import softmax
import numpy as np

from keras.engine.topology import Layer

from utils.network_operations import *

def gru_encoder_decoder_avg_concat(inputs, pred_depth=False, time_steps=10, pred_time_steps=10):
    '''GRU encoder decoder without scene information'''
    net = {}

    activation = 'relu'
    hidden_size = 512

    net['bbox_input'] = Input(shape=(time_steps, 4))
    net['depth_input'] = Input(shape=(time_steps, 5, 5, 1))
    net['flow_input'] = Input(shape=(time_steps, 5, 5, 2))
    net['ego_motion_input'] = Input(shape=(time_steps, 6))
    net['const_vel_input'] = Input(shape=(time_steps, 4))
    net['zero_input'] = Input(shape=(1, hidden_size))

    net['depth_reshape'] = Reshape((time_steps, 25))(net['depth_input'])
    net['flow_reshape'] = Reshape((time_steps, 50))(net['flow_input'])

    net['bbox_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['bbox_input'])
    net['ego_motion_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['ego_motion_input'])
    net['flow_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['flow_reshape'])

    if len(inputs) == 2:
        net['merged'] = net[inputs[0] + '_fc']
    else:
        concat_input = []
        for name in inputs:
            if name == 'zero':
                continue
            else:
                concat_input.append(net[name + '_fc'])
        # net['merged'] = Concatenate(axis=-1)(concat_input)
        net['merged'] = Average()(concat_input)

    net['FC_1']  = net['merged']
    net['gru_enc_output'], net['gru_enc_h'] = GRU(hidden_size,
                                                    return_sequences=False,
                                                    return_state=True,
                                                    activation=activation)(net['FC_1'])

    #     net['gru_enc_output'],net['gru_enc_h'],net['lstm_enc_c']  = LSTM(256,
    #                                                                        return_sequences=False,
    #                                                                        return_state=True,
    #                                                                        activation=activation)(net['FC_1'])

    net['gru_enc_fc'] = Dense(units=hidden_size, activation=activation)(net['gru_enc_output'])
    net['gru_enc_h_fc'] = Dense(units=hidden_size, activation=activation)(net['gru_enc_h'])
    #     net['lstm_enc_c_fc'] = Dense(units=256,activation=activation)(net['lstm_enc_c'])

    net['gru_dec'] = GRU(units=hidden_size, activation=activation, return_state=True)  #
    net['FC_gru'] = Dense(units=hidden_size, activation=activation)

    if pred_depth:
        net['FC_outputs'] = Dense(units=5, activation='linear')
    else:
        net['FC_outputs'] = Dense(units=4, activation='linear')

    net['outputs'] = []
    #     state = [net['gru_enc_h_fc'],net['lstm_enc_c_fc']]
    state = net['gru_enc_h_fc']
    for i in range(pred_time_steps):
        if i == 0:
            input_i = net[
                'zero_input']  # expand_dim()(net['gru_enc_output'])#[:,i:i+1,:] # should use all zero but not allowed to directly generate in keras.
        #         output, h, c = net['lstm_dec'](input_i,state)
        output, h = net['gru_dec'](input_i, initial_state=state)
        input_i = expand_dim()(net['FC_gru'](h))
        state = h  # [h,c]
        net['outputs'].append(expand_dim()(net['FC_outputs'](output)))

    net['outputs'] = Concatenate(axis=1)(net['outputs'])
    return net

def gru_encoder_decoder_old_late_fuse(inputs, pred_depth=False, time_steps=10, pred_time_steps=10, enc_size=512, dec_size=512):
    '''GRU encoder decoder without scene information'''
    net = {}

    activation = 'relu'
    hidden_size = enc_size
    dec_hidden_size = dec_size

    net['bbox_input'] = Input(shape=(time_steps, 4))
    net['flow_input'] = Input(shape=(time_steps, 5, 5, 2))
    net['ego_motion_input'] = Input(shape=(time_steps, 6))
    net['zero_input'] = Input(shape=(1, hidden_size))

    net['flow_reshape'] = Reshape((time_steps, 50))(net['flow_input'])

    net['bbox_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['bbox_input'])
    net['ego_motion_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['ego_motion_input'])
    net['flow_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['flow_reshape'])

    # All GRUs
    for name in inputs:
        if name == 'zero':
            continue
        else:
            if name == 'bbox' or name == 'flow' or name == 'ego_motion':
                net[name + '_encoder'], net[name + '_enc_state'] = GRU(hidden_size,
                                                                       return_sequences=False,
                                                                       return_state=True,
                                                                       activation=activation)(net[name + '_fc'])# dropout=0.2

    # merge GRU ouputs
    concat_input = []
    for name in net.keys():
        if name[-9:] == 'enc_state':
            concat_input.append(net[name])
    if len(concat_input) > 1:
        net['merged_enc_state'] = Average()(concat_input)
    else:
        net['merged_enc_state'] = concat_input[0]

    net['gru_enc_h_fc'] = Dense(units=dec_hidden_size, activation=activation)(net['merged_enc_state'])

    net['gru_dec'] = GRU(units=dec_hidden_size, activation=activation, return_state=True)  #dropout=0.4
    net['FC_gru'] = Dense(units=dec_hidden_size, activation=activation)

    if pred_depth:
        net['FC_outputs'] = Dense(units=5, activation='linear')
    else:
        net['FC_outputs'] = Dense(units=4, activation='tanh')

    net['outputs'] = []
    #     state = [net['gru_enc_h_fc'],net['lstm_enc_c_fc']]
    state = net['gru_enc_h_fc']
    for i in range(pred_time_steps):
        if i == 0:
            input_i = net['zero_input']
        output, h = net['gru_dec'](input_i, initial_state=state)
        input_i = expand_dim()(net['FC_gru'](h))
        state = h  # [h,c]
        net['outputs'].append(expand_dim()(net['FC_outputs'](output)))

    net['outputs'] = Concatenate(axis=1)(net['outputs'])
    return net


def gru_encoder_decoder_late_fuse(inputs, pred_depth=False, time_steps=10, pred_time_steps=10):
    '''GRU encoder decoder without scene information'''
    net = {}

    activation = 'relu'
    hidden_size = 512
    dec_hidden_size = 512

    net['bbox_input'] = Input(shape=(time_steps, 4))
    net['flow_input'] = Input(shape=(time_steps, 5, 5, 2))
    net['ego_motion_input'] = Input(shape=(time_steps, 6))
    net['future_ego_motion_input'] = Input(shape=(pred_time_steps, 3))
    # net['future_ego_motion_input'] = Input(shape=(time_steps, 6))
    net['zero_input'] = Input(shape=(1, hidden_size))

    net['flow_reshape'] = Reshape((time_steps, 50))(net['flow_input'])

    net['bbox_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['bbox_input'])
    net['ego_motion_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['ego_motion_input'])
    net['future_ego_motion_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['future_ego_motion_input'])

    net['flow_fc'] = TimeDistributed(Dense(units=hidden_size, activation=activation))(net['flow_reshape'])

    # All GRUs
    for name in inputs:
        if name == 'zero':
            continue
        else:
            if name == 'bbox' or name == 'flow':# or name == 'ego_motion': #or name == 'future_ego_motion':
                net[name + '_encoder'], net[name + '_enc_state'] = GRU(hidden_size,
                                                                       return_sequences=False,
                                                                       return_state=True,
                                                                       activation=activation)(net[name + '_fc'])

    # merge GRU ouputs
    concat_input = []
    for name in net.keys():
        if name[-9:] == 'enc_state':
            concat_input.append(net[name])
    if len(concat_input) > 1:
        net['merged_enc_state'] = Average()(concat_input)
    else:
        net['merged_enc_state'] = concat_input[0]

    net['gru_enc_h_fc'] = Dense(units=dec_hidden_size, activation=activation)(net['merged_enc_state'])

    net['gru_dec'] = GRU(units=dec_hidden_size, activation=activation, return_state=True)  #
    net['FC_gru'] = Dense(units=hidden_size, activation=activation)

    net['FC_outputs'] = Dense(units=4, activation='tanh')

    net['outputs'] = []
    #     state = [net['gru_enc_h_fc'],net['lstm_enc_c_fc']]
    state = net['gru_enc_h_fc']
    for i in range(pred_time_steps):
        ego_motion = Lambda(query_tensor, arguments={'i': i, 'dim':3})(net['future_ego_motion_fc'])

        if i == 0:
            input_i = Average()([net['zero_input'], ego_motion])
            # input_i = Concatenate()([net['zero_input'], expand_dim()(ego_motion)])

        output, h = net['gru_dec'](input_i, initial_state=state)
        input_i = expand_dim()(net['FC_gru'](h))

        input_i = Average()([input_i, ego_motion])
        # input_i = Concatenate()([input_i, expand_dim()(ego_motion)])

        state = h  # [h,c]
        net['outputs'].append(expand_dim()(net['FC_outputs'](output)))

    net['outputs'] = Concatenate(axis=1)(net['outputs'])
    return net
