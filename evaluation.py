import pickle
import os
import argparse
import logging
import torch
import time
from datetime import datetime
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.autograd import Variable

from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils.data_processing as dp
import utils.adsh_loss as al
import utils.rll_adsh_loss as rll
import utils.cnn_model as cnn_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default=[12,24,32,48], type=int, nargs='+',
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--name', default='cifar10', type=str,
                     help='log director name')
parser.add_argument('--arch', default='resnet50', choices=['resnet50', 'alexnet', 'vgg11'], type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=50, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--discrete-iter', default=1, type=int,
                    help='iterations of discrete optimization')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--adam', action='store_true', help='use adam optimizer')

parser.add_argument('--num-samples', default=2000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    # record['lr'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return

def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        # transforms.Scale(256),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingNUS_WIDE(
        'data/NUS-WIDE', 'database_img.txt', 'database_label.txt', transformations
    )
    dset_test =  dp.DatasetProcessingNUS_WIDE(
        'data/NUS-WIDE', 'test_img.txt', 'test_label.txt', transformations
    )
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        label_filepath = os.path.join(DATA_DIR, filename)
        label = np.loadtxt(label_filepath, dtype=np.int64)
        return torch.from_numpy(label)

    databaselabels = load_label('database_label.txt', 'data/NUS-WIDE')
    testlabels = load_label('test_label.txt', 'data/NUS-WIDE')

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(V, U, S, code_length, select_index, gamma):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        # data_input = Variable(data_input.cuda())
        data_input = data_input.cuda()
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.detach().cpu()).numpy()
    return B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10


def save_network(network, name, epoch):
    file_name = "net_{}.pth".format(epoch)
    save_path = os.path.join("./model", name, file_name)
    torch.save(network.cpu().state_dict(), save_path)
    network.cuda()


def adsh_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    discrete_iter = opt.discrete_iter
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5e-4
    update_step = [10, 20, 30, 40, 50]
    # update_step = [15, 30]
    num_samples = opt.num_samples
    gamma = opt.gamma

    record['param']['topk'] = 5000
    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)

    # adsh_loss = al.ADSHLoss(gamma, code_length, num_database)
    # ##======================
    # ## alpha: 5.0 margin: 4.0 temp: 2
    # rll_adsh_loss = rll.RLL_ADSHLoss(gamma, code_length, num_database, temp=2, alpha=4.0, margin=3.2, lam=1.0)
    #
    # if opt.adam:
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #
    # hash_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=update_step, gamma=0.1)
    #
    # V = np.zeros((num_database, code_length))
    # # V = np.ones((num_database, code_length))
    #
    # model.train()
    # writer = SummaryWriter(logdir=summary_path)
    # for iter in range(max_iter):
    #     iter_time = time.time()
    #
    #     print("Training Iteration: {}/{}".format(iter, max_iter))
    #
    #     '''
    #     sampling and construct similarity matrix
    #     '''
    #
    #     hash_lr_scheduler.step()
    #
    #     select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
    #     _sampler = subsetsampler.SubsetSampler(select_index)
    #     trainloader = DataLoader(dset_database, batch_size=batch_size,
    #                              sampler=_sampler,
    #                              shuffle=False,
    #                              num_workers=4)
    #     '''
    #     learning deep neural network: feature learning
    #     '''
    #     sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
    #     Sim = calc_sim(sample_label, database_labels)
    #     U = np.zeros((num_samples, code_length), dtype=np.float)
    #
    #     u_neg_list = defaultdict(list)
    #
    #     for epoch in range(epochs):
    #         print("Epoch: {}/{}".format(epoch, epochs))
    #
    #         for iteration, (train_input, train_label, batch_ind) in tqdm(enumerate(trainloader)):
    #             batch_size_ = train_label.size(0)
    #             u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
    #             train_input = train_input.cuda()
    #             output = model(train_input)
    #
    #             S = Sim.index_select(0, torch.from_numpy(u_ind))
    #             U[u_ind, :] = output.detach().cpu().numpy()
    #             S_omega = S.index_select(1, batch_ind)
    #
    #             model.zero_grad()
    #             # loss = adsh_loss(output, V, S, V[batch_ind.cpu().numpy(), :])
    #             loss = rll_adsh_loss(output, V, S, S_omega, V[batch_ind.cpu().numpy(), :], u_neg_list, u_ind)
    #             loss.backward()
    #             optimizer.step()
    #             writer.add_scalar("CNN_loss", loss.item())
    #             # record['CNN_loss'].append(loss.item())
    #             # logger.info('CNN_loss: {:.5f}'.format(loss.item()))
    #     # adjusting_learning_rate(optimizer, iter)
    #
    #     '''
    #     learning binary codes: discrete coding
    #     '''
    #     ##=====================================================================
    #     # barU = np.zeros((num_database, code_length))
    #     # barU[select_index, :] = U
    #     # Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
    #     # for d_iter in range(discrete_iter):
    #     #     for k in range(code_length):
    #     #         print("Discrete cyclic coordinate descent: {}/{}".format(k, code_length))
    #     #         sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
    #     #         V_ = V[:, sel_ind]
    #     #         Uk = U[:, k]
    #     #         U_ = U[:, sel_ind]
    #     #         V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
    #     #     iter_time = time.time() - iter_time
    #     #     loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma)
    #     #     logger.info('[Iteration: %3d/%3d : %d][Train Loss: %.4f]', iter, max_iter, d_iter, loss_)
    #     #     record['train loss'].append(loss_)
    #     #     record['iter time'].append(iter_time)
    #     # writer.add_scalar("Binary_loss", loss_)
    #     ##=====================================================================
    #     logger.info('[learning rate: {:.3e}]'.format(hash_lr_scheduler.get_lr()[0]))
    #
    #     ##=====================================================================
    #     ## RLL second update
    #
    #     ## numpy
    #     ## InNegativeSet
    #     ## d_ij = np.linalg.norm((U[i, :] - V[j, :]))
    #     ## d_ij < alpha
    #     ## WeightInNegativeSet
    #     ## for each i in U, there is a neg set in V
    #     ## if V_j_neg in u_neg_list
    #
    #     # for j in range(num_database):
    #     #     print("Second update of V: {}/{}".format(j, num_database))
    #     #     V_j_list = []
    #     #     for i in range(num_samples):
    #     #         if Sim[i,j] > 0:
    #     #             V_j_list.append(np.sign(U[i, :]))
    #     #         else:
    #     #             if j in u_neg_list[i]:
    #     #                 # V_j_neg = -np.sign(U[i, :]) * WeightInNegativeSet
    #     #                 V_j_neg = -np.sign(U[i, :])
    #     #                 V_j_list.append(V_j_neg)
    #     #     V_stack  = np.stack(V_j_list, axis=0)
    #     #     V[j,:] = np.sign(np.sum(V_stack, axis=0))
    #
    #     # V_expand = np.repeat(np.expand_dims(V, axis=0), num_samples, axis=0)
    #     # U_expand = np.repeat(np.expand_dims(U, axis=1), num_database, axis=1)
    #     # V_expand[Sim,:] = U_expand[Sim,:]
    #     for i in range(num_samples):
    #         V[Sim[i,:] > 0] = np.sign(U[i, :])
    #         V += V
    #     V = np.sign(V)
    #
    #     ##=====================================================================
    #
    #     if iter % 10 == 9:
    #         save_network(model, opt.name, iter)
    #
    # writer.close()

    '''
    training procedure finishes, evaluation
    '''
    print("Evaluation")

    load_model_path = os.path.join("model", opt.name, "net_{}.pth".format(49))
    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    model.cuda()

    testloader = DataLoader(dset_test, batch_size=1,
                             shuffle=False,
                             num_workers=4)
    qB = encode(model, testloader, num_test, code_length)
    rB = V
    map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
    topkmap = calc_hr.calc_topMap(qB, rB, test_labels.numpy(), database_labels.numpy(), record['param']['topk'])
    logger.info('[Evaluation: mAP: %.4f, top-%d mAP: %.4f]', map, record['param']['topk'], topkmap)
    record['rB'] = rB
    record['qB'] = qB
    record['map'] = map
    record['topkmap'] = topkmap
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)


if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    model_path = os.path.join("model", opt.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    summary_path = os.path.join("./tensorboard", opt.name)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    logdir = '-'.join(['log/RLL_ADSH', opt.name, datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    # bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in opt.bits:
        adsh_algo(bit)
