import argparse
import torch
import numpy as np
import pandas as pd
from datetime import date
import os
import json
import glob
import pickle
import utils
import make_optimizer
import prepare_data
import models
import torch.nn as nn
from sklearn.model_selection import KFold
from plot_metric.functions import BinaryClassification
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import softmax
matplotlib.rcParams["figure.dpi"] = 300

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
legend_properties = {'weight':'bold', 'size': 4}
plt.style.use('bmh')
today = date.today()
date = today.strftime("%m%d")
kf = KFold(n_splits=10, random_state=42, shuffle=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for read static info models")

    ## model representation to get from
    parser.add_argument("--model_name", type=str, default='Transformer', choices=['Transformer', 'TCN', 'LSTM'])
    # data
    parser.add_argument("--database", type=str, default='mimic', choices=['mimic', 'eicu'])
    parser.add_argument("--use_sepsis3", action = 'store_true', default= False, help="Whethe only use sepsis3 subset")
    parser.add_argument("--input_dim", type = int, default= 200, help="Dimension of variables used to train the extarction model")
    parser.add_argument("--bucket_size", type=int, default=300, help="path to the dataset")
    # TCN
    parser.add_argument('--num_channels', nargs='+', help='num of channels in TCN')
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    ## transformer
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")
    ## LSTM
    parser.add_argument("--hidden_dim", type=int, default=512, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")
    parser.add_argument("--idrop", type=float, default=0, help="RNN drop out in the very beginning")
    ## model to build
    parser.add_argument("--task_name", type=str, default='ihm', choices=['ihm', 'sofa'])
    parser.add_argument("--model_path", type=str, default='/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/checkpoints/0125_mimic_TCNsepsis_3_256_ks3/fold0_best_loss.pt')
    # hosp mort general Trans MIMIC/MEEP/checkpoints/0222_mimic_hosp_mort_48h_6h_tran_group0_fold2_best_roc_0.889.pt
    # SOFA sepsis_3 TRANS MIMIC/TCN/checkpoints/0125_mimic_Transformertransformer/fold5_best_loss.pt
    # hosp mort general TCN MIMIC/MEEP/checkpoints/0222_mimic_hosp_mort_48h_6h_tcn_group0_fold1_best_roc_0.879.pt
    # SOFA sepsis_3 TCN  MIMIC/TCN/checkpoints/0125_mimic_TCNsepsis_3_256_ks3/fold0_best_loss.pt
    # hosp mort general LSTM MIMIC/MEEP/checkpoints/0222_mimic_hosp_mort_48h_6h_rnn_group0_fold6_best_roc_0.881.pt
    # to do weights
    # SOFA sepsis_3 LSTM MIMIC/TCN/checkpoints/0126_mimic_RNN_lstm/fold0_best_loss.pt
    # SOFA general TCN MIMIC/TCN/checkpoints/0226_mimic_TCN_lstm/fold0_best_loss.pt
    # SOFA general TRAN MIMIC/TCN/checkpoints/0227_mimic_Transformer_general/fold3_best_loss.pt
    # SOFA general LSTM MIMIC/TCN/checkpoints/0227_mimic_RNN_general/fold0_best_loss.pt

    ## FC read model parameters
    parser.add_argument("--read_drop", type=float, default=0.2, help="Model dropout in FC read model")
    parser.add_argument("--read_reluslope", type=float, default=0.1, help="Relu slope in the FC read model")
    parser.add_argument("--read_channels", nargs='+', help='num of channels in FC read model')
    parser.add_argument("--output_classes", type=int, default=2, help="Which static column to target")
    parser.add_argument("--infer_ind", type=int, default=1, help="Which static column to target")
    parser.add_argument("--cal_pos_acc", action = 'store_false', default=True, help="Whethe calculate the acc of the positive class")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='close', choices=['same', 'close', 'random'], help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Epochs to wait before stop the training when the loss is not decreasing")

    ## data logging
    parser.add_argument("--checkpoint", type=str, default='test_read_models', help="name of checkpoint model")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # arg_dict['c_params'] = [1024, 1024, 0.2]
    # arg_dict['encode_params'] = [182, 0.2]
    # read data
    meep_mimic = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
                    allow_pickle=True).item()
    train_vital = meep_mimic ['train_head']
    dev_vital = meep_mimic ['dev_head']
    test_vital = meep_mimic ['test_head']
    mimic_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
                            allow_pickle=True).item()
    mimic_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
                            allow_pickle=True).item()

    #read eicu data
    meep_eicu = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_compile_0911_2022_2.npy', \
                    allow_pickle=True).item()
    train_vital_e = meep_eicu['train_head']
    dev_vital_e = meep_eicu ['dev_head']
    test_vital_e = meep_eicu ['test_head']
    eicu_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_static_0922_2022.npy', \
                            allow_pickle=True).item()
    eicu_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/eICU_target_0922_2022.npy', \
                            allow_pickle=True).item()
    # gender, age, race #
    target_index = [0, 1, 21] + [i for i in range(4, 21)]
    target_name = ['Gender', 'Age', 'Ethnicity', 'MI', 'CHF',
        'PVD', 'CBVD', 'Dementia', 'CPD', 'RD',
        'PUD', 'MLD', 'Diabetes_wo_cc',
        'Diabetes_cc', 'Paraplegia', 'Renal', 'Mal_cancer',
        'SLD', 'MST'] 
    bucket_sizes =  [300, 300, 300, 300, 300, 300, 300, 1200, 300, 1200, 1200, 600, 300, 600, 1200, 300, 300, 1200, 1200]
   
    true_ind = target_index[args.infer_ind]
    args.bucket_size = bucket_sizes[args.infer_ind]
    workname = date + '_' + args.database + '_' + args.model_name + '_' + target_name[true_ind]
    utils.creat_checkpoint_folder('./checkpoints/' + workname, 'params.json', vars(args))
    train_head, train_sofa, train_id, train_target =  utils.crop_data_target(train_vital, mimic_target, mimic_static, 'train', true_ind)
    dev_head, dev_sofa, dev_id, dev_target =  utils.crop_data_target(dev_vital , mimic_target, mimic_static, 'dev', true_ind)
    test_head, test_sofa, test_id, test_target =  utils.crop_data_target(test_vital, mimic_target, mimic_static, 'test', true_ind)
    train_head_e, train_sofa_e, train_id_e, train_target_e =  utils.crop_data_target_e(train_vital_e, eicu_target, eicu_static, 'train', true_ind)
    dev_head_e, dev_sofa_e, dev_id_e, dev_target_e =  utils.crop_data_target_e(dev_vital_e, eicu_target, eicu_static, 'dev', true_ind)
    test_head_e, test_sofa_e, test_id_e, test_target_e =  utils.crop_data_target_e(test_vital_e, eicu_target, eicu_static, 'test', true_ind)

    if args.use_sepsis3 == True:
        train_head, train_sofa, train_id, train_target = utils.filter_sepsis(train_head, train_sofa, train_id, train_target)
        dev_head, dev_sofa, dev_id, dev_target = utils.filter_sepsis(dev_head, dev_sofa, dev_id, dev_target)
        test_head, test_sofa, test_id, test_target = utils.filter_sepsis(test_head, test_sofa, test_id, test_target)
        train_head_e, train_sofa_e, train_id_e, train_target_e = utils.filter_sepsis_e(train_head_e, train_sofa_e, train_id_e, train_target_e)
        dev_head_e, dev_sofa_e, dev_id_e, dev_target_e = utils.filter_sepsis_e(dev_head_e, dev_sofa_e, dev_id_e, dev_target_e)
        test_head_e, test_sofa_e, test_id_e, test_target_e = utils.filter_sepsis_e(test_head_e, test_sofa_e, test_id_e, test_target_e)

    # get representations
    oc = 1 if args.task_name == 'sofa' else 2
    if args.model_name == 'TCN':
        train_encode, dev_encode, test_encode = utils.get_tcn_encode(args, train_head, dev_head, test_head, args.model_path, output_class=oc)
        train_encode_e, dev_encode_e, test_encode_e = utils.get_tcn_encode(args, train_head_e, dev_head_e, test_head_e, args.model_path, output_class=oc)
    elif args.model_name == "Transformer":
        train_encode, dev_encode, test_encode = utils.get_trans_encode(args, train_head, dev_head, test_head, args.model_path, output_class=oc)
        train_encode_e, dev_encode_e, test_encode_e = utils.get_trans_encode(args, train_head_e, dev_head_e, test_head_e, args.model_path, output_class=oc)
    elif args.model_name == "LSTM":
        train_encode, dev_encode, test_encode = utils.get_lstm_encode(args, train_head, dev_head, test_head, args.model_path, output_class=oc)
        train_encode_e, dev_encode_e, test_encode_e = utils.get_lstm_encode(args, train_head_e, dev_head_e, test_head_e, args.model_path, output_class=oc)
    else:
        raise ValueError('Model name not supported')


    trainval_data = train_encode + dev_encode
    # read dimension is dependent on encode dimension
    encode_dim = train_encode[0].shape[0]
    # crossval dataloader
    crossval_dataloader = prepare_data.get_huge_dataloader(args, train_encode_e, dev_encode_e, test_encode_e, train_target_e, dev_target_e, test_target_e)
    
    model = models.FCNet(num_inputs=encode_dim, num_channels=args.read_channels, 
                         dropout=args.read_drop, reluslope=args.read_reluslope, 
                         output_class=args.output_classes)
    torch.save(model.state_dict(), '/content/start_weights.pt')
    model.to(device)
    best_loss = 1e4
    best_acc = 0.5
    best_diff = 0.1

    # loss fn and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_data)):
        best_loss = 1e4
        patience = 0
        if c_fold >=1:
            model.load_state_dict(torch.load('/content/start_weights.pt'))
        print('Starting Fold %d'%c_fold)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        train_cv, dev_cv, train_labelcv, dev_labelcv = utils.get_cv_data(train_encode, dev_encode, np.asarray(train_target), np.asarray(dev_target), train_index, test_index)
        print('Compiled another CV data')
        train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader(args, train_cv, dev_cv, test_encode, train_labelcv, \
                                                                                            dev_labelcv, test_target)

        ctype, count= np.unique(dev_labelcv, return_counts=True)
        total_dev_samples = len(dev_labelcv)
        weights_per_class = torch.FloatTensor([ total_dev_samples / k / len(ctype) for k in count]).to(device)
        ce_val_loss = nn.CrossEntropyLoss(weight = weights_per_class)

        for j in  range(args.epochs):
            model.train()
            sofa_list = []
            sofap_list = []
            loss_t = []
            loss_to = []

            for vitals, target, key_mask in train_dataloader:
                # print(label.shape)
                if args.warmup == True:
                    model_opt.optimizer.zero_grad()
                else:
                    model_opt.zero_grad()
                # ti_data = Variable(ti.float().to(device))
                td_data = vitals.to(device) # (6, 182, 24)
                sofa = target.to(device) #(6, )
                key_mask = key_mask.to(device)
                # tgt_mask = model.get_tgt_mask(td_data.shape[-1]).to(device)
                sofa_p = model(td_data)
                pred  = torch.stack([sofa_p[i][key_mask[i]==0].mean(dim=-2) for i in range(len(sofa_p))])
                loss = ce_loss(pred, sofa.squeeze(-1))
                loss.backward()
                model_opt.step()

                sofa_list.append(sofa)
                sofap_list.append(pred)
                loss_t.append(loss)

            train_acc = utils.cal_acc(sofap_list, sofa_list)
            print('Train acc is %.2f%%'%(train_acc*100))

            loss_avg = ce_loss(torch.concat(sofap_list), torch.concat(sofa_list).squeeze(-1)).cpu().detach().item()
            #  np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())


            model.eval()
            y_list = []
            y_pred_list = []
            ti_list = []
            td_list = []
            # loss_val = []
            with torch.no_grad():  # validation does not require gradient

                for vitals, target, key_mask in dev_dataloader:
                    # ti_test = Variable(torch.FloatTensor(ti)).to(device)
                    td_test = vitals.float().to(device)
                    sofa_t = target.long().to(device)
                    key_mask = key_mask.to(device)

                    sofap_t = model(td_test)

                    pred  = torch.stack([sofap_t[i][key_mask[i]==0].mean(dim=-2) for i in range(len(sofa_t))])
                    # loss_v = ce_val_loss(pred, sofa_t.squeeze(-1))

                    y_list.append(sofa_t)
                    y_pred_list.append(pred)
                    # loss_val.append(loss_v)

            loss_te = ce_val_loss(torch.concat(y_pred_list), torch.concat(y_list).squeeze(-1)).cpu().numpy().item()
            # np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
            val_acc = utils.cal_acc(y_pred_list, y_list)

            if args.cal_pos_acc == True:
                val_pos_acc = utils.cal_pos_acc(y_pred_list, y_list, pos_ind = 1)
                print('Validation pos acc is %.2f%%'%(val_pos_acc*100))
                diff = abs(val_pos_acc - val_acc)
                if diff < best_diff:
                    print('best diff is %.2f%%'%(diff*100))
                    torch.save(model.state_dict(), './checkpoints/' +  workname + '/' + 'fold%d'%c_fold + '_best_diff.pt')
                    best_diff = diff


            print('Validation acc is %.2f%%'%(val_acc*100))

            if loss_te < best_loss:
                best_loss = loss_te
                patience = 0
                best_loss = loss_te
                #run["train/loss"].log(loss_avg)
                torch.save(model.state_dict(), './checkpoints/' +  workname + '/' + 'fold%d'%c_fold + '_best_loss.pt')
            else:
                patience +=1
                if patience >=args.patience:
                    print('Start next fold')
                    break
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), './checkpoints/' +  workname + '/' + 'fold%d'%c_fold + '_best_acc.pt')
            print('Epoch %d, : Train loss is %.4f, validation loss is %.4f' %(j, loss_avg, loss_te))


    #
    sm = nn.Softmax(dim=1)
    weights_file = glob.glob('./checkpoints/' + workname + '/*.pt')
    auc = []
    crossval_auc = []
    for w in weights_file:
        model.load_state_dict(torch.load(w))
        wname = w.split('/')[-1].split('.')[0]
        model.to(device)
        y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(model, test_dataloader)
        fig = utils.plot_confusion_matrix(y_list, y_pred_list,
                                        # label_x = ['Pred-White', 'Pred-Black\nAfrican American'], label_y = ['White', 'Black/African\nAmerican'], \
                                        #   label_x = ['Pred-Female', 'Pred-Male'], label_y = ['Female', 'Male'], \
                                    title='%s Prediction'%target_name[true_ind])
        fig.savefig('./checkpoints/' + workname + '/cm_maps/' + '%s.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        pred = sm(torch.concat(y_pred_list))[:, 1].cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(torch.concat(y_list).detach().cpu().numpy(),\
                                                pred, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))

        # plot auc and auprc
        # y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(model, test_dataloader)
        y_true = np.concatenate([i.cpu().detach().numpy() for i in y_list])
        y_score =  np.concatenate([i.cpu().detach().numpy() for i in y_pred_list])
        y_sig = np.stack([softmax(data) for data in y_score])
        # # for race
        # y_true_r = np.asarray([0 if i==1 else 1 for i in y_true])

        bc = BinaryClassification(y_true.squeeze(-1), y_sig[:,1], labels=["Class 1", "Class 2"])
        # Figures
        plt.figure(figsize=(5,5))
        bc.plot_roc_curve(plot_threshold=False, x_text_margin = 0.01)
        plt.title('ROC')
        plt.savefig('./checkpoints/' + workname + '/auc_plots/' + '%s_prc.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)

        plt.figure(figsize=(5,5))
        bc.plot_precision_recall_curve(plot_threshold=False, x_text_margin = 0.01)
        plt.title('PRC')
        plt.savefig('./checkpoints/' + workname + '/auc_plots/' + '%s_roc.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.show()


        # the crossvalidation
        y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(model, crossval_dataloader)
        fig = utils.plot_confusion_matrix(y_list, y_pred_list,
                                        # label_x = ['Pred-White', 'Pred-Black\nAfrican American'], label_y = ['White', 'Black/African\nAmerican'], \
                                        #   label_x = ['Pred-', 'Pred-Male'], label_y = ['Female', 'Male'], \
                                    title='%s Prediction'%target_name[true_ind])
        fig.savefig('./checkpoints/' + workname + '/cm_maps_cv/' + '%s.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        pred = sm(torch.concat(y_pred_list))[:, 1].cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(torch.concat(y_list).detach().cpu().numpy(),\
                                                pred, pos_label=1)
        crossval_auc.append(metrics.auc(fpr, tpr))

        # plot auc and auprc
        # y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(model, test_dataloader)
        y_true = np.concatenate([i.cpu().detach().numpy() for i in y_list])
        y_score =  np.concatenate([i.cpu().detach().numpy() for i in y_pred_list])
        y_sig = np.stack([softmax(data) for data in y_score])
        # # for race
        # y_true_r = np.asarray([0 if i==1 else 1 for i in y_true])

        bc = BinaryClassification(y_true.squeeze(-1), y_sig[:,1], labels=["Class 1", "Class 2"])
        # Figures
        plt.figure(figsize=(5,5))
        bc.plot_roc_curve(plot_threshold=False, x_text_margin = 0.01)
        plt.title('ROC')
        plt.savefig('./checkpoints/' + workname + '/auc_plots_cv/' + '%s_prc.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)

        plt.figure(figsize=(5,5))
        bc.plot_precision_recall_curve(plot_threshold=False, x_text_margin = 0.01)
        plt.title('PRC')
        plt.savefig('./checkpoints/' + workname + '/auc_plots_cv/' + '%s_roc.eps'%wname, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.show()

    # save auc stats
    with open(os.path.join('./checkpoints/' + workname, 'auc_stats.pkl'), 'wb') as f:
        pickle.dump(auc, f)
    with open(os.path.join('./checkpoints/' + workname, 'auc_cv_stats.pkl'), 'wb') as f:
        pickle.dump(crossval_auc, f)
