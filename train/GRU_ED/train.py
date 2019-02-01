import numpy as np
import sys
sys.path.append('../../')
import argparse
import os
from keras.models import Model
from keras.callbacks import TensorBoard, Callback

from models.RNN_ED import *
from models.RNN_ED_late_fusion import *
from utils.eval_utils import *
from utils.loss import *
from utils.training_utils import *
import pickle as pkl
import glob

FEATURE_DIR = '/home/yyao/Documents/car_intersection/feature_output/'
DATA_DIR = '/home/yyao/Documents/Future_vehicle_localization/data/'
# DATA_DIR = '/home/brianyao/Documents/future_vehicle_localization/data/'

def model_setup(inputs, loss, model='late_fusion', weights=None):

    # net = gru_encoder_decoder(inputs=inputs,time_steps=10, pred_time_steps=10)
    if model == 'late_fusion':
        net = gru_encoder_decoder_late_fuse(inputs=inputs, time_steps=10, pred_time_steps=10)
    elif model == 'old_late_fusion':
        net = gru_encoder_decoder_old_late_fuse(inputs=inputs, time_steps=10, pred_time_steps=10)
    # net = gru_encoder_decoder_old_late_fuse(inputs=inputs, time_steps=10, pred_time_steps=10, enc_size=512, dec_size=512)

    model = Model(inputs = [net[name+'_input'] for name in inputs],
                  outputs = net['outputs'])
    model.summary()

    model.compile(optimizer='adam', loss=loss)#, loss_weights=[1.0, 1.0])
    # weights = 'checkpoints/bbox_flow/weights.20-0.02.hdf5'
    if weights is not None:
        model.load_weights(weights)
    return model

def schedule(epoch, decay=1.0):
    return base_lr * args.decay ** (epoch)

def get_arguments():
    """
    Pase all arguments from cmdline
    :return:
        a list of parsed arguments
    """
    parser = argparse.ArgumentParser(description="train RNN-ED")
    parser.add_argument("--model", type=str, default="late_fusion", help="name of model")
    parser.add_argument("--gpu", type=str, default="1", help="Indicate the gpu id to use")
    parser.add_argument("--lr", type=float, default=5e-4, help="base learning rate for training")
    parser.add_argument("--decay", type=float, default=1.0, help="base learning rate for training")
    parser.add_argument("--per_epoch", type=int, default=5, help="do evaluation every n epoch")
    parser.add_argument("--nb_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--inputs", nargs='+',
                        default=['bbox', 'flow', 'ego_motion','zero'],
                        help="Indicate the gpu id to use")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="directory to save the checkpoints")
    parser.add_argument("--output_name", type=str, default="metrics.pkl", help="name of the output")
    return parser.parse_args()

if __name__ == '__main__':

    args = get_arguments()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    train_keys = glob.glob(DATA_DIR + 'train_nw/*.pkl')
    val_keys = glob.glob(DATA_DIR + 'val_nw/*.pkl')
    print(len(train_keys))
    print(len(val_keys))

    inputs = args.inputs

    easy_cases, challenge_cases = load_cases(DATA_DIR)

    try:
        os.stat(args.save_dir)
    except:
        os.mkdir(args.save_dir)

    batch_size = 64

    zero_input_shape = 512
    gen = Generator(train_keys, val_keys, inputs=inputs,
                    zero_input_shape=zero_input_shape,
                    FEATURE_DIR=FEATURE_DIR,
                    batch_size=batch_size, time_steps=10, pred_time_steps=10)

    base_lr = args.lr

    # setup callbacks
    easy_case_callback = TestCallbacks(inputs, easy_cases,
                                       per_epoch=args.per_epoch,
                                       zero_input_shape=zero_input_shape,
                                       FEATURE_DIR=FEATURE_DIR, case='easy case')
    challenge_case_callback = TestCallbacks(inputs, challenge_cases,
                                            per_epoch=args.per_epoch,
                                            zero_input_shape=zero_input_shape,
                                            FEATURE_DIR=FEATURE_DIR,
                                            case='challenge case')
    all_case_callback = TestCallbacks(inputs, val_keys,
                                      per_epoch=args.per_epoch,
                                      zero_input_shape=zero_input_shape,
                                      FEATURE_DIR=FEATURE_DIR,
                                      case='all case')

    callbacks = [keras.callbacks.ModelCheckpoint(args.save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1, save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 easy_case_callback,
                 challenge_case_callback,
                 TensorBoard(log_dir='./tensorboard')]


    model = model_setup(inputs, rmse, model=args.model, weights=None)

    steps_per_epoch = int(len(train_keys) / batch_size)
    validation_steps = int(len(val_keys) / batch_size)

    history = model.fit_generator(gen.generate(train=True),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.nb_epoch,
                                  shuffle=True,
                                  validation_steps=validation_steps,
                                  validation_data=gen.generate(train=False),
                                  callbacks=callbacks,
                                  verbose=1)

    all_case_FDE = (np.array(easy_case_callback.FDE) * len(easy_cases) + np.array(challenge_case_callback.FDE) * len(challenge_cases)) / ( len(easy_cases) +  len(challenge_cases))
    all_case_MDE = (np.array(easy_case_callback.MDE) * len(easy_cases) + np.array(challenge_case_callback.MDE) * len(challenge_cases)) / ( len(easy_cases) +  len(challenge_cases))
    all_case_FIOU = (np.array(easy_case_callback.FIOU) * len(easy_cases) + np.array(challenge_case_callback.FIOU) * len(challenge_cases)) / ( len(easy_cases) +  len(challenge_cases))

    outputs = {}
    outputs['FDE'] = [np.array(easy_case_callback.FDE),
                      np.array(challenge_case_callback.FDE),
                      all_case_FDE]
    outputs['MDE'] = [np.array(easy_case_callback.MDE),
                      np.array(challenge_case_callback.MDE),
                      all_case_MDE]
    outputs['FIOU'] = [np.array(easy_case_callback.FIOU),
                       np.array(challenge_case_callback.FIOU),
                       all_case_FIOU]

    pkl.dump(outputs, open(args.output_name, 'wb'))
