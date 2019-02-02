from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(bbox):
    model = LinearRegression()
    X = np.expand_dims(np.arange(10), axis=-1)
    y = bbox
    model.fit(X,y)

    new_X = np.expand_dims(np.arange(10,20), axis=-1)
    pred = model.predict(new_X)

    return pred
    
def const_vel_predict(bbox, freq=30, pred_time_steps=30):
    '''
    bbox: (times, 4), [cx, cy, w, h]
    '''
    v_x = np.mean((bbox[1:, 0] - bbox[:-1, 0])) * freq

    v_y = np.mean((bbox[1:, 1] - bbox[:-1, 1])) * freq

    v_w = np.mean((bbox[1:, 2] - bbox[:-1, 2])) * freq

    v_h = np.mean((bbox[1:, 3] - bbox[:-1, 3])) * freq

    velocity = np.array([v_x, v_y, v_w, v_h])
    pred_bbox = np.zeros([pred_time_steps, bbox.shape[1]])

    prev = bbox[-1, :]
    for i in range(pred_time_steps):
        pred_bbox[i, :] = prev + velocity / freq

        prev = pred_bbox[i, :]

    return np.expand_dims(pred_bbox - bbox[-1], axis=0)


def const_accel_predict(bbox, freq=30, pred_time_steps=30):
    '''
    bbox: (times, 4), [cx, cy, w, h]
    '''
    v_x = (bbox[1:, 0] - bbox[:-1, 0]) * freq

    v_y = (bbox[1:, 1] - bbox[:-1, 1]) * freq

    v_w = (bbox[1:, 2] - bbox[:-1, 2]) * freq

    v_h = (bbox[1:, 3] - bbox[:-1, 3]) * freq

    a_x = np.mean((v_x[1:] - v_x[:-1]) * freq)
    a_y = np.mean((v_y[1:] - v_y[:-1]) * freq)
    a_w = np.mean((v_w[1:] - v_w[:-1]) * freq)
    a_h = np.mean((v_h[1:] - v_h[:-1]) * freq)

    accel = np.array([a_x, a_y, a_w, a_h])

    pred_bbox = np.zeros([pred_time_steps, bbox.shape[1]])

    prev = bbox[-1, :]
    v0 = np.array([v_x[-1], v_y[-1], v_w[-1], v_h[-1]])
    v0 = np.array([np.mean(v_x), np.mean(v_y), np.mean(v_w), np.mean(v_h)])
    for i in range(pred_time_steps):
        pred_bbox[i, :] = bbox[-1, :] + v0 * (i+1) / freq + 0.5 * accel * ((i+1) / freq) ** 2

        prev = pred_bbox[i, :]

    return np.expand_dims(pred_bbox - bbox[-1], axis=0)


def poly_predict(bbox, freq=30, pred_time_steps=30):
    x = bbox[:, 0]
    v_x = (x[1:] - x[:-1]) * freq
    a_x = np.mean(v_x[1:] - v_x[:-1]) * freq

    y = bbox[:, 1]
    p = np.polyfit(x, y, 3)

    new_x = []
    v0 = np.mean(v_x)

    for i in range(pred_time_steps):
        new_x.append(x[-1] + v0 * i / freq + 0.5 * a_x * (i / freq) ** 2)
    new_x = np.array(new_x)
    new_y = p[0] * new_x ** 3 + \
            p[1] * new_x ** 2 + \
            p[2] * new_x ** 1 + \
            p[3] * new_x ** 0

    new_w = np.zeros(pred_time_steps)
    new_h = np.zeros(pred_time_steps)

    new_boxes = np.vstack([new_x, new_y, new_w, new_h])

    return np.expand_dims(new_boxes.T - bbox[-1], 0)