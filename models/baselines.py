from keras import backend as K
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector, Lambda, TimeDistributed, RepeatVector
from keras.layers import Conv1D, Conv2DTranspose, Lambda, AveragePooling2D, Activation, Add
from keras.layers import Flatten, BatchNormalization, Reshape, Concatenate

def Conv1DTranspose(input_tensor, filters, kernel_size, activation, strides=1, padding='valid'):
    x = Lambda(lambda x: K.expand_dims(x, axis=-2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        strides=(strides, 1),
                        padding=padding,
                        activation=activation,
                        data_format='channels_last')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-2))(x)
    return x


def conv1d(inputs='all', time_steps=10):
    net = {}

    net['bbox_input'] = Input(shape=(time_steps, 4))
    net['depth_input'] = Input(shape=(time_steps, 5, 5, 1))
    net['flow_input'] = Input(shape=(time_steps, 5, 5, 2))
    net['ego_motion_input'] = Input(shape=(time_steps, 6))
    net['const_vel_input'] = Input(shape=(time_steps, 4))

    net['depth_reshape'] = Reshape((time_steps, 25))(net['depth_input'])
    net['flow_reshape'] = Reshape((time_steps, 50))(net['flow_input'])

    net['bbox_conv1'] = Conv1D(32, 3, activation='relu')(net['bbox_input'])
    net['bbox_conv1'] = BatchNormalization()(net['bbox_conv1'])
    net['bbox_conv2'] = Conv1D(64, 3, activation='relu')(net['bbox_conv1'])
    net['bbox_conv2'] = BatchNormalization()(net['bbox_conv2'])
    net['bbox_conv3'] = Conv1D(128, 3, activation='relu')(net['bbox_conv2'])
    net['bbox_conv3'] = BatchNormalization()(net['bbox_conv3'])
    net['bbox_conv4'] = Conv1D(128, 3, activation='relu')(net['bbox_conv3'])
    net['bbox_conv4'] = BatchNormalization()(net['bbox_conv4'])

    net['depth_conv1'] = Conv1D(32, 3, activation='relu')(net['depth_reshape'])
    net['depth_conv1'] = BatchNormalization()(net['depth_conv1'])
    net['depth_conv2'] = Conv1D(64, 3, activation='relu')(net['depth_conv1'])
    net['depth_conv2'] = BatchNormalization()(net['depth_conv2'])
    net['depth_conv3'] = Conv1D(128, 3, activation='relu')(net['depth_conv2'])
    net['depth_conv3'] = BatchNormalization()(net['depth_conv3'])
    net['depth_conv4'] = Conv1D(128, 3, activation='relu')(net['depth_conv3'])
    net['depth_conv4'] = BatchNormalization()(net['depth_conv4'])

    net['flow_conv1'] = Conv1D(32, 3, activation='relu')(net['flow_reshape'])
    net['flow_conv1'] = BatchNormalization()(net['flow_conv1'])
    net['flow_conv2'] = Conv1D(64, 3, activation='relu')(net['flow_conv1'])
    net['flow_conv2'] = BatchNormalization()(net['flow_conv2'])
    net['flow_conv3'] = Conv1D(128, 3, activation='relu')(net['flow_conv2'])
    net['flow_conv3'] = BatchNormalization()(net['flow_conv3'])
    net['flow_conv4'] = Conv1D(128, 3, activation='relu')(net['flow_conv3'])
    net['flow_conv4'] = BatchNormalization()(net['flow_conv4'])

    net['ego_motion_conv1'] = Conv1D(32, 3, activation='relu')(net['ego_motion_input'])
    net['ego_motion_conv1'] = BatchNormalization()(net['ego_motion_conv1'])
    net['ego_motion_conv2'] = Conv1D(64, 3, activation='relu')(net['ego_motion_conv1'])
    net['ego_motion_conv2'] = BatchNormalization()(net['ego_motion_conv2'])
    net['ego_motion_conv3'] = Conv1D(128, 3, activation='relu')(net['ego_motion_conv2'])
    net['ego_motion_conv3'] = BatchNormalization()(net['ego_motion_conv3'])
    net['ego_motion_conv4'] = Conv1D(128, 3, activation='relu')(net['ego_motion_conv3'])
    net['ego_motion_conv4'] = BatchNormalization()(net['ego_motion_conv4'])

    net['const_vel_conv1'] = Conv1D(32, 3, activation='relu')(net['const_vel_input'])
    net['const_vel_conv1'] = BatchNormalization()(net['const_vel_conv1'])
    net['const_vel_conv2'] = Conv1D(64, 3, activation='relu')(net['const_vel_conv1'])
    net['const_vel_conv2'] = BatchNormalization()(net['const_vel_conv2'])
    net['const_vel_conv3'] = Conv1D(128, 3, activation='relu')(net['const_vel_conv2'])
    net['const_vel_conv3'] = BatchNormalization()(net['const_vel_conv3'])
    net['const_vel_conv4'] = Conv1D(128, 3, activation='relu')(net['const_vel_conv3'])
    net['const_vel_conv4'] = BatchNormalization()(net['const_vel_conv4'])

    if len(inputs) == 1:
        net['merged'] = net[inputs[0] + '_conv4']
    else:
        net['merged'] = Concatenate(axis=-1)([net[name + '_conv4'] for name in inputs])

    net['conv5'] = Conv1D(256, 1, activation='relu')(net['merged'])
    net['conv5'] = BatchNormalization()(net['conv5'])
    net['conv6'] = Conv1D(256, 1, activation='relu')(net['conv5'])
    net['conv6'] = BatchNormalization()(net['conv6'])

    #     net['conv6'] = K.expand_dims(net['conv6'],axis=-2)

    net['deconv1'] = Conv1DTranspose(net['conv6'], 256, 3, activation='relu')
    net['deconv1'] = BatchNormalization()(net['deconv1'])
    net['deconv2'] = Conv1DTranspose(net['deconv1'], 128, 3, activation='relu')
    net['deconv2'] = BatchNormalization()(net['deconv2'])
    net['deconv3'] = Conv1DTranspose(net['deconv2'], 64, 3, activation='relu')
    net['deconv3'] = BatchNormalization()(net['deconv3'])
    net['deconv4'] = Conv1DTranspose(net['deconv3'], 32, 3, activation='relu')
    net['deconv4'] = BatchNormalization()(net['deconv4'])
    #     net['deconv4'] = net['deconv4'][:,:,0,:]

    net['conv7'] = Conv1D(4, 1, activation='linear')(net['deconv4'])  # activation='linear'

    return net