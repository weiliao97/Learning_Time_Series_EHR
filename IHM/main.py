# parameter tuning
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import importlib
import IHM.models as models
import IHM.prepare_data as prepare_data
import IHM.make_optimizer as make_optimizer
import IHM.utils as utils
import IHM.loss_fn as loss_fn
importlib.reload(models)
importlib.reload(make_optimizer)
importlib.reload(prepare_data)
importlib.reload(utils)
importlib.reload(loss_fn)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.stats as st
from datetime import date
today = date.today()
date = today.strftime("%m%d")
kf = KFold(n_splits=10, random_state=42, shuffle=True)
f_sm = nn.Softmax(dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Tranformer models")

    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--model_name", type=str, default='TCN', choices=['Trans', 'TCN', 'RNN'])
    parser.add_argument("--rnn_type", type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])

    # important, which target to use as the prediction taregt 0: hospital mortality, 1: ARF, 2: shock
    parser.add_argument("--target_index", type=int, default=0, help="Which static column to target")
    parser.add_argument("--output_classes", type=int, default=2, help="Which static column to target")
    parser.add_argument("--cal_pos_acc", action='store_false', default=True,
                        help="Whethe calculate the acc of the positive class")
    parser.add_argument("--filter_los", action='store_false', default=True,
                        help="Whether filter the first xxx hours of stay")
    parser.add_argument("--thresh", type=int, default=48, help="how many hours of data to use")
    parser.add_argument("--gap", type=int, default=6, help="gap hours between record stop and data used in training")

    # model parameters
    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    parser.add_argument('--num_channels', nargs='+', help='<Required> Set flag')
    # LSTM
    parser.add_argument("--hidden_dim", type=int, default=512, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")
    parser.add_argument("--idrop", type=float, default=0, help="RNN drop out in the very beginning")

    # transformer
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='same', choices=['same', 'close', 'random'],
                        help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    # learning rate
    parser.add_argument('--warmup', action='store_true', default = False, help="whether use learning rate warm up")
    parser.add_argument('--lr_factor', type=int, default=0.1, help="warmup_learning rate factor")
    parser.add_argument('--lr_steps', type=int, default=2000, help="warmup_learning warm up steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")  # could be overwritten by warm up
    # loss compute, mean or last , output (16, 24, 2) for RNN and TCN
    parser.add_argument("--loss_rule", type=str, default='last', choices=['mean', 'last'])

    # Parse and return arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_map = {0: 'hosp_mort', 1: 'ARF', 2: 'shock'}
    # load data
    data_label = np.load(args.dataset_path, allow_pickle=True).item()
    train_head = data_label['train_head']
    static_train_filter = data_label['static_train_filter']
    dev_head = data_label['dev_head']
    static_dev_filter = data_label['static_dev_filter']
    test_head = data_label['test_head']
    static_test_filter = data_label['static_test_filter']
    s_train = np.stack(static_train_filter, axis=0)
    s_dev = np.stack(static_dev_filter, axis=0)
    s_test = np.stack(static_test_filter, axis=0)

    print('Running target %d, thresh %d, gap %d, model %s' % (args.target_index, args.thresh, args.gap, args.model_name))
    workname = date + '_%s' % task_map[args.target_index] + '_%dh' % args.thresh + '_%sh' % args.gap + '_%s' % (args.model_name.lower())
    print(workname)
    args.checkpoint_model = workname

    print('Before filtering, train size is %d' % (len(train_head)))
    train_label, train_data = utils.filter_los(s_train, train_head, args.thresh, args.gap)
    dev_label, dev_data = utils.filter_los(s_dev, dev_head, args.thresh, args.gap)
    test_label, test_data = utils.filter_los(s_test, test_head, args.thresh, args.gap)
    print('After filtering, train size is %d' % (len(train_data)))
    train_label = train_label[:, 0]
    dev_label = dev_label[:, 0]
    test_label = test_label[:, 0]

    trainval_data = train_data + dev_data

    # result_dict to log and save data
    result_dict = {}
    # create model
    if args.model_name == 'TCN':
        print('Creating TCN')
        model = models.TemporalConv(num_inputs=200, num_channels=[int(i) for i in args.num_channels], \
                                    kernel_size=args.kernel_size, dropout=args.dropout, \
                                    output_class=args.output_classes)
        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in TCN is %d" % utils.count_parameters(model))
    elif args.model_name == 'RNN':
        model = models.RecurrentModel(cell=args.rnn_type, hidden_dim=args.hidden_dim,
                                      layer_dim=args.layer_dim, \
                                      output_dim=args.output_classes, dropout_prob=args.dropout,
                                      idrop=args.idrop)

        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in RNN is %d" % utils.count_parameters(model))

    else:
        model = models.Trans_encoder(feature_dim=200, d_model=args.d_model, \
                                     nhead=args.n_head, d_hid=args.dim_ff_mul * args.d_model, \
                                     nlayers=args.num_enc_layer, out_dim=args.output_classes, dropout=args.dropout)
        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in RNN is %d" % utils.count_parameters(model))

    model.to(device)
    best_loss = 1e4
    best_acc = 0.5
    best_diff = 0.1
    best_roc = 0.5

    # loss fn and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    if args.warmup == True:
        print('Using warm up')
        model_opt = make_optimizer.NoamOpt(args.d_model, args.lr_factor, args.lr_steps,
                                           torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                                            eps=1e-9))
    else:
        print('No warm up')
        model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        torch.save(model_opt.state_dict(), '/content/start_weights_opt.pt')

    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_data)):
        best_loss = 1e4
        patience = 0
        if c_fold >= 1:
            model.load_state_dict(torch.load('/content/start_weights.pt'))
            model_opt.load_state_dict(torch.load('/content/start_weights_opt.pt'))
        print('Starting Fold %d' % c_fold)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        train_cv, dev_cv, train_labelcv, dev_labelcv = utils.get_cv_data(train_data, dev_data, train_label, dev_label, train_index, test_index)
        print('Compiled another CV data')
        train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader( \
            args, train_cv, dev_cv, test_data, train_labelcv, dev_labelcv, test_label)

        ctype, count = np.unique(dev_labelcv, return_counts=True)
        total_dev_samples = len(dev_labelcv)
        weights_per_class = torch.FloatTensor([total_dev_samples / k / len(ctype) for k in count]).to(
            device)
        ce_val_loss = nn.CrossEntropyLoss(weight=weights_per_class)

        best_model = utils.train_model(args, c_fold, model, model_opt, train_dataloader,
                                       dev_dataloader, ce_loss, ce_val_loss)

        # test auroc on test set
        y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(args, best_model, test_dataloader)
        y_l = torch.concat(y_list).cpu().numpy()
        y_pred_l = np.concatenate([f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
        test_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])

        if test_roc > best_roc:
            best_roc = test_roc
            print('Save a model with best roc %.3f' % best_roc)
            torch.save(best_model.state_dict(),
                       './checkpoints/' + args.checkpoint_model + '_fold%d' %c_fold + '_best_roc_%.3f.pt' % best_roc)
            # for best roc on test set models, perform bootstrapping on both test set and the whole another set
            # save the results in a dictionary and save that dictionary regularly
            roc = []
            prc = []
            for i in tqdm(range(1000)):
                test_index = np.random.choice(len(test_label), 1000)
                test_i = [test_data[i] for i in test_index]
                test_t = test_label[test_index]
                test_dataloader = prepare_data.get_test_loader(args, test_i, test_t)
                # test auroc on test set
                y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(args, best_model,
                                                                                           test_dataloader)
                y_l = torch.concat(y_list).cpu().numpy()
                y_pred_l = np.concatenate(
                    [f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
                # tpr, tnr = get_tp_tn(y_l.squeeze(-1), y_pred_l[:, 1])
                test_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])
                test_prc = average_precision_score(y_l.squeeze(-1), y_pred_l[:, 1])
                roc.append(test_roc)
                prc.append(test_prc)
            # create 95% confidence interval for population mean weight
            result_dict['fold%d'%c_fold] = ['%.3f' % np.mean(roc)]
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(roc), loc=np.mean(roc), scale=np.std(roc)))
            result_dict['fold%d'%c_fold].append('%.3f' % np.mean(prc))
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(prc), loc=np.mean(prc), scale=np.std(prc)))
            result_dict['fold%d'%c_fold].append(len(test_label))

    utils.write_json('./checkpoints', args.checkpoint_model + '.json', result_dict)