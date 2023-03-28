from torch.autograd import Variable
import pandas as pd
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import tqdm
import argparse
import sys
import os
import json
from datetime import date

today = date.today()
date = today.strftime("%m%d")
import importlib
import models
import prepare_data
# import make_optimizer
import utils

# import loss_fn
importlib.reload(models)
# importlib.reload(make_optimizer)
importlib.reload(prepare_data)
importlib.reload(utils)
# importlib.reload(loss_fn)
# import neptune.new as neptune
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=None, shuffle=False)

mse_loss = nn.MSELoss()


def calculate_l1(model):
    L1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.static.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    for name, param in model.static1.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    for name, param in model.static2.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    for name, param in model.static3.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    for name, param in model.s_composite.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    return L1_reg


def creat_checkpoint_folder(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

def mse_maskloss(output, target, mask):
    loss = [mse_loss(output[i][mask[i] == 0], target[i][mask[i] == 0]) for i in range(len(output))]
    return torch.mean(torch.stack(loss))


def simulate_data(inputs):
    s_inputs = [np.random.rand(i.shape[0], i.shape[1]) for i in inputs]
    return s_inputs


def crop_data_target(vital, target_dict, static_dict, mode):
    length = [i.shape[-1] for i in vital]
    train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m > 24]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m > 24]
    sofa_tail = [target_dict[mode][j][24:] / 15 for j in stayids]
    sname = 'static_' + mode
    static_data = [static_dict[sname][static_dict[sname].index.get_level_values('stay_id') == j].values for j in
                   stayids]
    # remove hospital mort flag and los
    # squeese from (1, 25) to (25, )
    static_data = [np.squeeze(np.concatenate((s[:, :2], s[:, 4:]), axis=1)) for s in static_data]
    return train_filter, static_data, sofa_tail, stayids


def filter_sepsis(vital, static, sofa, ids):
    id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/mimic_sepsis3.csv')
    sepsis3_id = id_df['stay_id'].values  # 1d array
    index_dict = dict((value, idx) for idx, value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    static_sepsis = [static[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    return vital_sepsis, static_sepsis, sofa_sepsis, [ids[i] for i in ind]


def slice_data(trainval_data, index):
    return [trainval_data[i] for i in index]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for Tranformer models")
    # data
    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--database", type=str, default='mimic', choices=['mimic', 'eicu'])
    parser.add_argument("patient_subset", type=str, )
    parser.add_argument("--bucket_size", type=int, default=300, help="path to the dataset")
    parser.add_argument("--project_name", type=str, default='TCN', help="Neptune data logging")
    parser.add_argument("--use_vital_only", action='store_true', default=False, help="Whethe only use vital features")
    parser.add_argument("--use_sepsis3", action='store_false', default=True, help="Whethe only use sepsis3 subset")

    # model_name fc means only use fc layer to work on previous transformer outputs
    parser.add_argument("--model_name", type=str, default='TCN', choices=['TCN', 'RNN', 'Transformer'])
    # how to fuse with transformer models and LSTM models is still pending
    parser.add_argument("--static_fusion", type=str, default='med',
                        choices=['no_static', 'early', 'med', 'late', 'all', 'inside'])

    # model parameters
    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    # parser.add_argument('--num_channels', nargs='+', help='<Required> Set flag')
    # parser.add_argument("--use_encode", action = 'store_true', help="Dimension of the feedforward model")

    # LSTM
    parser.add_argument("--rnn_type", type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")
    parser.add_argument("--idrop", type=float, default=0, help="RNN drop out in the very beginning")

    # transformer
    parser.add_argument('--warmup', action='store_true', help="whether use learning rate warm up")
    parser.add_argument('--lr_factor', type=int, default=0.1, help="warmup_learning rate factor")
    parser.add_argument('--lr_steps', type=int, default=2000, help="warmup_learning warm up steps")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='close', choices=['same', 'close', 'random'],
                        help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  # could be overwritten by warm up

    # keep training
    parser.add_argument('--resume_from', type=bool, default=False, required=False,
                        help="whether retrain from a certain old model")
    parser.add_argument('--old_dict', type=str)

    parser.add_argument("--checkpoint", type=str, default='med_fusion_ks3', help=" name of checkpoint model")
    # Parse and return arguments
    args = parser.parse_known_args()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # log run
    arg_dict = vars(args)
    # deal with num_channels and c_params here
    # num_channels also used in fc models
    # arg_dict['num_channels'] = [1024, 1024, 1204, 1024]
    # for simple fc
    # arg_dict['num_channels'] = [128]
    # arg_dict['c_params'] = [1024, 1024, 0.2]
    # arg_dict['encode_params'] = [182, 0.2]

    # for fusion all
    arg_dict['num_channels'] = [256, 256, 256, 256]
    arg_dict['s_param'] = [256, 256, 256, 0.2]
    arg_dict['c_param'] = [256, 256, 0.2]
    arg_dict['sc_param'] = [256, 256, 256, 0.2]
    arg_dict['use_encode'] = False
    arg_dict['fuse_inside'] = True
    arg_dict['encode_param'] = [128, 128, 128, 0.2]

    # log params
    run['parameters'] = arg_dict
    workname = date + '_' + args.database + '_' + args.model_name + args.checkpoint
    creat_checkpoint_folder('./checkpoints/' + workname, 'params.json', arg_dict)

    # load_data
    meep_mimic = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
        allow_pickle=True).item()
    train_vital = meep_mimic['train_head']
    dev_vital = meep_mimic['dev_head']
    test_vital = meep_mimic['test_head']
    mimic_static = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
        allow_pickle=True).item()
    mimic_target = np.load(
        '/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
        allow_pickle=True).item()

    train_head, train_static, train_sofa, train_id = crop_data_target(train_vital, mimic_target, mimic_static, 'train')
    dev_head, dev_static, dev_sofa, dev_id = crop_data_target(dev_vital, mimic_target, mimic_static, 'dev')
    test_head, test_static, test_sofa, test_id = crop_data_target(test_vital, mimic_target, mimic_static, 'test')

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = filter_sepsis(train_head, train_static, train_sofa, train_id)
        dev_head, dev_static, dev_sofa, dev_id = filter_sepsis(dev_head, dev_static, dev_sofa, dev_id)
        test_head, test_static, test_sofa, test_id = filter_sepsis(test_head, test_static, test_sofa, test_id)

    if args.use_vital_only == True:
        train_head = [tr[:184, :] for tr in train_head]
        dev_head = [de[:184, :] for de in dev_head]
        test_head = [te[:184, :] for te in test_head]
        input_dim = 184
    else:
        input_dim = 200

    if args.resume_from == True:
        if args.old_dict is not None:
            print("Loading weights from %s" % args.old_dict)
            model.load_state_dict(torch.load(args.old_dict + ".pt"))
            _, _, _, best_loss = utils.get_eval_results(model, test_dataloader)
            print('Best loss from the trained model is: {:.4f}'.format(best_loss))
        else:
            print('Keep training')

    else:
        # creat model default model is TCN to explore different kinds of fusion
        if args.static_fusion == 'no_static':

            # if args.model_name == 'TCN':
            model = models.TemporalConv(num_inputs=input_dim, num_channels=arg_dict['num_channels'], \
                                        kernel_size=args.kernel_size, dropout=args.dropout)
        elif args.static_fusion == 'med':
            model = models.TemporalConvStatic(num_inputs=input_dim, num_channels=arg_dict['num_channels'], \
                                              num_static=25, kernel_size=args.kernel_size, dropout=args.dropout)

        elif args.static_fusion == 'early':
            model = models.TemporalConvStaticE(num_inputs=225, num_channels=arg_dict['num_channels'], \
                                               num_static=25, kernel_size=args.kernel_size, dropout=args.dropout)

        elif args.static_fusion == 'late':
            model = models.TemporalConvStaticL(num_inputs=input_dim, num_channels=arg_dict['num_channels'], \
                                               num_static=25, kernel_size=args.kernel_size, dropout=args.dropout)
        elif args.static_fusion == 'all':
            model = models.TemporalConvStaticA(num_inputs=225, num_channels=arg_dict['num_channels'], \
                                               num_static=25, kernel_size=args.kernel_size, dropout=args.dropout)


        else:  # inside
            model = models.TemporalConvStaticI(num_inputs=225, num_channels=arg_dict['num_channels'], num_static=25,
                                               kernel_size=args.kernel_size, \
                                               dropout=args.dropout, s_param=arg_dict['s_param'],
                                               c_param=arg_dict['c_param'], sc_param=arg_dict['sc_param'], \
                                               use_encode=arg_dict['use_encode'], encode_param=arg_dict['encode_param'],
                                               fuse_inside=arg_dict['fuse_inside'])

            # elif args.model_name == 'RNN':
            #     model = models.RecurrentModel(cell=args.rnn_type, input_dim = input_dim, hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, \
            #                                 output_dim=1, dropout_prob=args.dropout, idrop=args.idrop)

            # elif args.model_name == 'Transformer':
            #     model = models.Trans_encoder(feature_dim=input_dim, d_model=args.d_model, \
            #           nhead=args.n_head, d_hid=args.dim_ff_mul * args.d_model, \
            #           nlayers=args.num_enc_layer, out_dim=1, dropout=args.dropout)

        print('Model trainable parameters are: %d' % utils.count_parameters(model))
        torch.save(model.state_dict(), '/content/start_weights.pt')

        model.to(device)
        best_loss = 1e4

        # loss fn and optimizer
        loss_fn = nn.MSELoss()
        model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # fuse inside opt term
        # model_opt = torch.optim.Adam([
        #     {'params': model.TB1.parameters()},
        #     {'params': model.TB2.parameters()},
        #     {'params': model.TB3.parameters()},
        #     {'params': model.TB4.parameters()},
        #     {'params': model.composite.parameters()},
        #     {'params': model.static.parameters(), 'weight_decay':  0.0001},
        #     {'params': model.static1.parameters(), 'weight_decay':  0.0001},
        #     {'params': model.static2.parameters(), 'weight_decay':  0.0001},
        #     {'params': model.static3.parameters(), 'weight_decay':  0.0001},
        #     {'params': model.s_composite.parameters(), 'weight_decay':  0.0001}
        # ], lr=args.lr)

        # prepare data  # dataversion 0413, 0414, 0511
        # data_label = np.load('/content/drive/My Drive/Colab Notebooks/MIMIC/eICU_compile_0520_2022.npy', \
        #                 allow_pickle=True).item()

        # train_head = data_label['train_head']
        # static_train_filter = data_label['static_train_filter']
        # train_sofa_tail = data_label['train_sofa_tail']
        # train_sofa_head = data_label['train_sofa_head']
        # dev_head = data_label['dev_head']
        # static_dev_filter = data_label['static_dev_filter']
        # dev_sofa_tail = data_label['dev_sofa_tail']
        # dev_sofa_head = data_label['dev_sofa_head']
        # test_head = data_label['test_head']
        # static_test_filter = data_label['static_test_filter']
        # test_sofa_tail = data_label['test_sofa_tail']
        # test_sofa_head = data_label ['test_sofa_head']
        # s_train = np.stack(static_train_filter, axis=0)
        # s_dev = np.stack(static_dev_filter, axis=0)
        # s_test = np.stack(static_test_filter, axis=0)
        # reduce to only dynamic ot intervention
        # def reduce_dynamic(full_dynamic):
        #     return [full_dynamic[i][184:, :] for i in range(len(full_dynamic))]

        # train_head = reduce_dynamic(train_head)
        # dev_head = reduce_dynamic(dev_head)
        # test_head = reduce_dynamic(test_head)

        # 10-fold cross validation
        trainval_head = train_head + dev_head
        trainval_static = train_static + dev_static
        trainval_stail = train_sofa + dev_sofa
        trainval_ids = train_id + dev_id

        for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
            best_loss = 1e4
            patience = 0
            if c_fold >= 1:
                model.load_state_dict(torch.load('/content/start_weights.pt'))
            print('Starting Fold %d' % c_fold)
            print("TRAIN:", len(train_index), "TEST:", len(test_index))
            train_head, val_head = slice_data(trainval_head, train_index), slice_data(trainval_head, test_index)
            train_static, val_static = slice_data(trainval_static, train_index), slice_data(trainval_static, test_index)
            train_stail, val_stail = slice_data(trainval_stail, train_index), slice_data(trainval_stail, test_index)
            train_id, val_id = slice_data(trainval_ids, train_index), slice_data(trainval_ids, test_index)

            train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader(args, train_head, val_head,
                                                                                             test_head, \
                                                                                             train_stail, val_stail,
                                                                                             test_sofa,
                                                                                             train_static=train_static,
                                                                                             dev_static=dev_static,
                                                                                             test_static=test_static,
                                                                                             train_id=train_id,
                                                                                             dev_id=val_id,
                                                                                             test_id=test_id)

            for j in range(args.epochs):
                model.train()
                sofa_list = []
                sofap_list = []
                loss_t = []
                loss_to = []

                for vitals, static, target, train_ids, key_mask in train_dataloader:
                    # print(label.shape)
                    if args.warmup == True:
                        model_opt.optimizer.zero_grad()
                    else:
                        model_opt.zero_grad()
                    # ti_data = Variable(ti.float().to(device))
                    # td_data = vitals.to(device) # (6, 182, 24)
                    # sofa = target.to(device)
                    # if args.model_name == 'TCN': # always TCN
                    sofa_p = model(vitals.to(device), static.to(device))

                    # elif args.model_name == 'RNN':
                    #     # x_lengths have to be a 1d tensor
                    #     td_transpose = vitals.to(device).transpose(1, 2)
                    #     x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                    #     sofa_p = model(td_transpose, x_lengths)
                    # elif args.model_name == 'Transformer':
                    #     tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                    #     sofa_p = model(vitals.to(device), tgt_mask, key_mask.to(device))

                    loss = mse_maskloss(sofa_p, target.to(device), key_mask.to(device))
                    # l1_penalty = calculate_l1(model)
                    # loss = loss + 0.001*l1_penalty
                    loss.backward()
                    model_opt.step()

                    sofa_list.append(target)
                    sofap_list.append(sofa_p)
                    loss_t.append(loss)

                loss_avg = np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())

                model.eval()
                y_list = []
                y_pred_list = []
                ti_list = []
                td_list = []
                id_list = []
                loss_val = []
                with torch.no_grad():  # validation does not require gradient

                    for vitals, static, target, val_ids, key_mask in dev_dataloader:
                        # ti_test = Variable(torch.FloatTensor(ti)).to(device)
                        # td_test = Variable(torch.FloatTensor(vitals)).to(device)
                        # sofa_t = Variable(torch.FloatTensor(target)).to(device)

                        # tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
                        # if args.model_name == 'TCN':
                        sofap_t = model(vitals.to(device), static.to(device))
                        # elif args.model_name == 'RNN':
                        #     # x_lengths have to be a 1d tensor
                        #     td_transpose = vitals.to(device).transpose(1, 2)
                        #     x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                        #     sofap_t = model(td_transpose, x_lengths)
                        # elif args.model_name == 'Transformer':
                        #     tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                        #     sofap_t = model(vitals.to(device), tgt_mask, key_mask.to(device))

                        loss_v = mse_maskloss(sofap_t, target.to(device), key_mask.to(device))
                        y_list.append(target.detach().numpy())
                        y_pred_list.append(sofap_t.cpu().detach().numpy())
                        loss_val.append(loss_v)
                        id_list.append(val_ids)

                loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
                if loss_te < best_loss:
                    patience = 0
                    best_loss = loss_te
                    run["train/loss"].log(loss_avg)
                    torch.save(model.state_dict(),
                               './checkpoints/' + workname + '/' + 'fold%d' % c_fold + '_best_loss.pt')
                else:
                    patience += 1
                    if patience >= 10:
                        print('Start next fold')
                        break

                run["train/loss_fold%d" % c_fold].log(loss_avg)
                run["val/loss_fold%d" % c_fold].log(loss_te)
                print('Epoch %d, : Train loss is %.4f, test loss is %.4f' % (j, loss_avg, loss_te))