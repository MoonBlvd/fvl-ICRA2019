import numpy as np
from keras import backend as K
from scipy.stats import multivariate_normal
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

def rmse(y_true, y_pred):
    print(y_true.shape)
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))
    #K.sqrt(K.mean(K.mean(K.square(y_pred - y_true), axis=-1)))


# def smooth_L1_v2(y, y_pred):
#     '''
#     Smooth l1 loss from fast RCNN
#     y_init is the anchor box
#     y_pred and y_true have shape (batch_size, time_steps, 4), [cx, cy, w, h]
#     '''
#     threshold = 1
#     MIN = 0.0001
#
#     y_true = y[:, 1:, :]  # target, difference between future bboxes and initial bbox
#     y_init = y[:, 0:1, :]  # initial bbox
#
#     print(y_init)
#     y_pred_bbox = y_init + y_pred  # (batch, 4) + (batch,10,4)
#     y_true_bbox = y_init + y_true
#
#     t_x = y_pred[:, :, 0] / y_init[:, :, 2]
#     t_y = y_pred[:, :, 1] / y_init[:, :, 3]
#     t_w = K.log((y_pred[:, :, 2] + y_init[:, :, 2]) / y_init[:, :, 2])
#     t_h = K.log((y_pred[:, :, 3] + y_init[:, :, 3]) / y_init[:, :, 3])
#
#     v_x = y_true[:, :, 0] / y_init[:, :, 2]
#     v_y = y_true[:, :, 1] / y_init[:, :, 3]
#     v_w = K.log((y_true[:, :, 2] + y_init[:, :, 2]) / y_init[:, :, 2])
#     v_h = K.log((y_true[:, :, 3] + y_init[:, :, 2]) / y_init[:, :, 3])
#
#     x = K.abs(t_x - v_x)
#     x = K.switch(x < threshold, 0.5 * x ** 2, threshold * (x - 0.5 * threshold))
#
#     y = K.abs(t_y - v_y)
#     y = K.switch(y < threshold, 0.5 * y ** 2, threshold * (y - 0.5 * threshold))
#
#     w = K.abs(t_w - v_w)
#     w = K.switch(w < threshold, 0.5 * w ** 2, threshold * (w - 0.5 * threshold))
#
#     h = K.abs(t_h - v_h)
#     h = K.switch(h < threshold, 0.5 * h ** 2, threshold * (h - 0.5 * threshold))
#
#     return K.sum(x + y + w + h, axis=1)


def smooth_L1(y, y_pred):
    '''
    Smooth l1 loss from fast RCNN
    y has shape (batch_size, time_steps+1, 4), [cx, cy, w, h]\
    y_true has shape [(batch_size, time_steps, 2), (batch_size, time_steps, 2)] for xy and wh
    '''
    threshold = 1
    MIN = 0.0001

    y_true = y[:, 1:, :]  # target, difference between future bboxes and initial bbox
    y_init = y[:, 0:1, :]  # initial bbox



    y_pred_xy = y_init[:, :2] + y_pred[0]  # (batch, 2) + (batch,10,2)
    y_pred_wh = y_init[:, 2:] + y_pred[1]

    y_true_bbox = y_init + y_true

    #     y_true = y
    #     y_pred_bbox = y_pred
    #     y_true_bbox = y

    t_x = (y_true[:, :, 0] - y_pred[:, :, 0]) / (y_pred_bbox[:, :, 2])
    t_y = (y_true[:, :, 1] - y_pred[:, :, 1]) / (y_pred_bbox[:, :, 3])
    t_w = K.log(y_true_bbox[:, :, 2] / (y_pred_bbox[:, :, 2]))
    t_h = K.log(y_true_bbox[:, :, 3] / (y_pred_bbox[:, :, 3]))

    x = K.abs(t_x)
    x = K.switch(x < threshold, 0.5 * x ** 2, threshold * (x - 0.5 * threshold))

    y = K.abs(t_y)
    y = K.switch(y < threshold, 0.5 * y ** 2, threshold * (y - 0.5 * threshold))

    w = K.abs(t_w)
    w = K.switch(w < threshold, 0.5 * w ** 2, threshold * (w - 0.5 * threshold))

    h = K.abs(t_h)
    h = K.switch(h < threshold, 0.5 * h ** 2, threshold * (h - 0.5 * threshold))

    return K.sum(x + y + w + h, axis=1)


def bbox_loss(y_true, y_pred):
    '''
    y_true and y_pred are differences between predicted bbox and the initial box
    y_true shape is (batch, timesteps, 4)
    '''
    L_loc = K.mean(K.sqrt(K.square(y_true[:, :, 0] - y_pred[:, :, 0]) + K.square(y_true[:, :, 1] - y_pred[:, :, 1])))
    L_wh = K.mean(K.abs(y_true[:, :, 2] - y_pred[:, :, 2]) + K.abs(y_true[:, :, 3] - y_pred[:, :, 3]))

    return L_loc + L_wh

def gaussian_output_loss(y_true, y_pred):
    '''
    Loss for models with separate location prediction and size prediction.
    :param y_true: (batch_size, time_steps, 4)
    :param y_pred: [batch_size, time_steps, 4 + 4 + 6]
    :return:
    '''
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)
    # log_likelihood = 0
    y_true = y_true[...,:4]
    print("y_true: ", y_true.shape)
    # for i in range(10):
    mean = y_pred[..., :4]
    print("mean_shape:", mean.shape)
    # sigma = tf.placeholder(tf.float32, [None, 4, 4]) # K.variable(K.zeros_like(mean))
    # sigma = tf.Variable()
    # print(sigma.shape)
    # sigma = K.repeat(sigma,4)
    # diag = y_pred[:, i, 4:8]
    # corr = y_pred[:, i, 8:]

    # first_row = tf.expand_dims(y_pred[..., 4:8], axis=-2)
    # second_row = tf.expand_dims(tf.concat([y_pred[..., 5:6], y_pred[..., 8:11]], axis=-1), axis=-2)
    # third_row = tf.expand_dims(tf.concat([y_pred[..., 6:7], y_pred[..., 9:10], y_pred[..., 11:13]], axis=-1), axis=-2)
    # fourth_row = tf.expand_dims(tf.concat([y_pred[..., 7:8], y_pred[..., 10:11], y_pred[..., 12:13], y_pred[..., 13:14]], axis=-1), axis=-2)

    first_row = tf.expand_dims(tf.concat([y_pred[..., 4:5], tf.zeros_like(y_pred[..., 7:10])], axis=-1), axis=-2)
    second_row = tf.expand_dims(tf.concat([y_pred[..., 5:7], tf.zeros_like(y_pred[..., 5:7])], axis=-1), axis=-2)
    third_row = tf.expand_dims(tf.concat([y_pred[..., 7:10], tf.zeros_like(y_pred[..., 4:5])], axis=-1), axis=-2)
    fourth_row = tf.expand_dims(y_pred[..., 10:14], axis=-2)

    # print(first_row.shape)
    # print(second_row.shape)
    # print(third_row.shape)
    # print(fourth_row.shape)

    L = tf.concat([first_row, second_row, third_row, fourth_row], axis=-2)
    print(L.shape)
    #L_trans = K.permute_dimensions(L, (0,1,3,2))
    sigma = K.batch_dot(L, L, axes=[3,3])

    print(sigma.shape)
        # for j in range(4):
        #     for k in range(4):
        #         if j == k:
        #             sigma[:-1, j, k].assign(diag[:, j])
        #         else:
        #             sigma[:, j, k].assign(corr[:, j+k])

        # log_likelihood -= multivariate_normal.logpdf(y_true[i,:], mean = mean, cov=sigma)
    mvn = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=sigma, validate_args=True)
    print(mvn)
    prob = mvn.prob(y_true)
    print("prob shape:", prob.shape)
    prob = tf.cast(prob, tf.float32)

    loss = K.mean(K.sum(-K.log(prob), axis=-1))
    print("loss shape: ", loss.shape)
    return loss
    # print(tf.matrix_inverse(sigma).shape)

def two_branch_loss(y_true, y_pred):
    '''
    Loss for models with separate location prediction and size prediction.
    :param y_true: (batch_size, time_steps, 4)
    :param y_pred: []
    :return:
    '''

    pred_xy = y_pred[0]
    pred_wh = y_pred[1]

    L_xy = K.mean(K.sqrt(K.sum(K.square(pred_xy - y_true[:, :, :2]), axis=-1)))

    L_wh = K.mean(K.sqrt(K.sum(K.square(pred_wh - y_true[:, :, 2:]), axis=-1)))

    return (L_xy + L_wh)

# def variety_loss(y_true, y_pred):





