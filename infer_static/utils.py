import os 
import json 
import torch 
from torch.autograd import Variable
import numpy as np 
import pandas as pd 
import torch.nn as nn
import loss_fn
import models
import sklearn.metrics as metrics
# plot cm matrix 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
import seaborn as sns
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 6}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ce_loss = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=1) 

def creat_checkpoint_folder(target_path, target_file, data):
    """
    Create a folder to save the checkpoint
    input: target_path,
           target_file,
           data
    output: None
    """
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
            os.makedirs(target_path + '/auc_plots')
            os.makedirs(target_path + '/auc_plots_cv')
            os.makedirs(target_path + '/cm_maps')
            os.makedirs(target_path + '/cm_maps_cv')
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

def plot_confusion_matrix(y_list, y_pred_list, title='Confusion matrix', label_x=None, label_y=None):
    """
    Plot confusion matrix
    input: y_list: list of true labels
           y_pred_list: list of predicted labels
           title: title of the plot
           label_x: x label of the plot
           label_y: y label of the plot
    output: fig: figure of the plot
    """
    num_class = y_pred_list[0].shape[-1]
    y_label = torch.concat(y_list).detach().cpu().numpy()
    pred_t = torch.concat(y_pred_list)
    y_pred = torch.argmax(pred_t, dim=-1).unsqueeze(-1).detach().cpu().numpy()
    
    cm = metrics.confusion_matrix(y_label, y_pred)
    cf_matrix = cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    # percentage based on true label 
    gr = (cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)).flatten()
    group_percentages = ['{0:.2%}'.format(value) for value in gr]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]

    labels = np.asarray(labels).reshape(num_class, num_class)

    if label_x is not None:
        xlabel = label_x
        ylabel = label_y
    else:
        xlabel = ['Pred-%d'%i for i in range(num_class)]
        ylabel = ['%d'%i for i in range(num_class)]
    
    sns.set(font_scale = 1.5)

    hm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = 'OrRd', \
    annot_kws={"fontsize": 16}, xticklabels=xlabel, yticklabels=ylabel, cbar=False)
    hm.set(title=title)
    fig = plt.gcf()
    plt.show()
    return fig 

def plot_confusion_matrix_cpu(y_list, y_pred_list, title='Confusion matrix', label_x=None, label_y=None):
    num_class = y_pred_list[0].shape[-1]
    y_pred = np.argmax(y_pred_list, axis=-1)
    
    cm = metrics.confusion_matrix(y_list, y_pred)
    cf_matrix = cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    # percentage based on true label 
    gr = (cm/np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)).flatten()
    group_percentages = ['{0:.2%}'.format(value) for value in gr]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]

    labels = np.asarray(labels).reshape(num_class, num_class)

    if label_x is not None:
        xlabel = label_x
        ylabel = label_y
    else:
        xlabel = ['Pred-%d'%i for i in range(num_class)]
        ylabel = ['%d'%i for i in range(num_class)]
    
    sns.set(font_scale = 1.5)

    hm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = 'OrRd', \
    annot_kws={"fontsize": 16}, xticklabels=xlabel, yticklabels=ylabel, cbar=False)
    hm.set(title=title)
    fig = plt.gcf()
    plt.show()
    return fig 

# plot_curve
# from get_evalacc_results
def plot_auprc(y_list, y_pred_list):
    binary_label = torch.concat(y_list).detach().cpu().numpy()
    binary_outputs = softmax(torch.concat(y_pred_list)).detach().cpu().numpy()
    metrics.PrecisionRecallDisplay.from_predictions(binary_label,  binary_outputs[:, 1])

    no_skill = len(binary_label[binary_label==1]) / len(binary_label)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (AP = %.2f)'%no_skill)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig

def plot_roc(y_list, y_pred_list):
    binary_label = torch.concat(y_list).detach().cpu().numpy()
    binary_outputs = softmax(torch.concat(y_pred_list)).detach().cpu().numpy()
    metrics.RocCurveDisplay.from_predictions(binary_label,  binary_outputs[:, 1])

    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # no_skill = len(binary_label[binary_label==1]) / len(binary_label)
    # plot the no skill precision-recall curve
    # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (AP = %.2f)'%no_skill)
    plt.legend()
    plt.xlabel(xlabel = 'FPR', fontsize=8)
    plt.ylabel(ylabel = 'TPR', fontsize=8)
    fig = plt.gcf()
    plt.show()
    return fig


# calculate acc
def cal_acc(pred, label):
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    acc = (prediction == label_t).sum()/len(pred_t)
    return acc

# for classification
def get_evalacc_results(model, test_loader):
    
    model.eval()
    y_list = []
    y_pred_list = []
    td_list = []
    loss_val = []
    with torch.no_grad():  # validation does not require gradient

        for vitals, target, key_mask in test_loader:
            # ti_test = Variable(torch.FloatTensor(ti)).to(device)
            td_test =  Variable(vitals.float().to(device))
            sofa_t =  Variable(target.long().to(device))

            # tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
            sofap_t = model(td_test)
            
            pred  = torch.stack([sofap_t[i][key_mask[i]==0].mean(dim=-2) for i in range(len(sofa_t))])
            loss_v = ce_loss(pred, sofa_t.squeeze(-1))
            
            y_list.append(sofa_t)
            y_pred_list.append(pred)
            loss_val.append(loss_v)
            td_list.append(td_test)

        loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
        val_acc = cal_acc(y_pred_list, y_list)
  
    return y_list, y_pred_list, td_list, loss_te, val_acc

def get_eval_results(model, test_loader):
    
    model.eval()
    y_list = []
    y_pred_list = []
    td_list = []
    loss_val = []
    with torch.no_grad():  # validation does not require gradient

        for vitals, target, key_mask in test_loader:
            # ti_test = Variable(torch.FloatTensor(ti)).to(device)
            td_test = Variable(torch.FloatTensor(vitals)).to(device)
            sofa_t = Variable(torch.FloatTensor(target)).to(device)

            tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
            sofap_t = model(td_test, tgt_mask_test, key_mask.to(device))
            
            loss_v = loss_fn.mse_maskloss(sofap_t, sofa_t, key_mask.to(device))
            y_list.append(sofa_t.cpu().detach().numpy())
            y_pred_list.append(sofap_t.cpu().detach().numpy())
            loss_val.append(loss_v)
    loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())

    return y_list, y_pred_list, td_list, loss_te

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def creat_encoded_input(model, vitals):
    output_t = []
    for d in vitals:
        d_tensor = torch.from_numpy(d).unsqueeze(0).float().to(device)
        tgt_mask = model.get_tgt_mask(d_tensor.shape[-1]).to(device)
        key_mask = torch.zeros((1, d_tensor.shape[-1])).to(device)
        output = model(d_tensor, tgt_mask, key_mask)
        output_t.append(output.transpose(-2, -1).squeeze(0).detach().cpu().numpy())
    return output_t

def get_encoded_outputs(train_head, dev_head, test_head, weights):
    model = models.Trans_encoder(feature_dim=202, d_model=256, \
                      nhead=8, d_hid=1024, \
                      nlayers=5, out_dim=1, dropout=0.2)
    model.to(device)
    # to work with new mimic data 

    model.load_state_dict(torch.load(weights))
    
    model.decoder = Identity()
    model.eval()
    train_encode = creat_encoded_input(model, train_head)
    dev_encode = creat_encoded_input(model, dev_head)
    test_encode = creat_encoded_input(model, test_head)

    return train_encode, dev_encode, test_encode

def creat_tcn_encode(model, vitals):
    output_t = []
    for d in vitals:
        d_tensor = torch.from_numpy(d).unsqueeze(0).float().to(device)
        output = model.network(d_tensor)
        output_t.append(output.squeeze(0).detach().cpu().numpy())
    return output_t

def get_tcn_encode(args, train_head, dev_head, test_head, weights, output_class=1):

    model = models.TemporalConv(num_inputs = args.input_dim, num_channels=args.num_channels, \
                                            kernel_size=args.kernel_size, dropout = args.dropout, output_class=output_class)                                        
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    train_encode = creat_tcn_encode(model, train_head)
    dev_encode = creat_tcn_encode(model, dev_head)
    test_encode = creat_tcn_encode(model, test_head)

    return train_encode, dev_encode, test_encode

def get_tcn_encode_t(args, test_head, weights, output_class=1):

    model = models.TemporalConv(num_inputs = args.input_dim, num_channels=args.num_channels, \
                                            kernel_size=args.kernel_size, dropout = args.dropout, output_class=output_class)                                        
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    test_encode = creat_tcn_encode(model, test_head)

    return test_encode

def creat_lstm_encode(model, vitals):
    output_t = []
    for d in vitals: # (200, 24), (200, 32)
        d_tensor = torch.from_numpy(d).unsqueeze(0).float().to(device)
        out, (_, _) = model.rnn(d_tensor.transpose(1, 2)) # (1, 24, 512)
        output_t.append(out.transpose(1, 2).squeeze(0).detach().cpu().numpy())
    return output_t

def get_lstm_encode(args, train_head, dev_head, test_head, weights, output_class=1):
    model = models.RecurrentModel(cell='lstm', hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, \
                                            output_dim=output_class, dropout_prob=args.dropout, idrop=args.idrop)
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    train_encode = creat_lstm_encode(model, train_head)
    dev_encode = creat_lstm_encode(model, dev_head)
    test_encode = creat_lstm_encode(model, test_head)

    return train_encode, dev_encode, test_encode

def get_lstm_encode_t(args, test_head, weights, output_class=1):
    model = models.RecurrentModel(cell='lstm', hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, \
                                            output_dim=output_class, dropout_prob=args.dropout, idrop=args.idrop)
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    # train_encode = creat_lstm_encode(model, train_head)
    # dev_encode = creat_lstm_encode(model, dev_head)
    test_encode = creat_lstm_encode(model, test_head)

    return test_encode

def creat_trans_encode(model, vitals):
    output_t = []
    for d in vitals: # (200, 24), (200, 32)
        src = torch.from_numpy(d).unsqueeze(0).float().to(device)
        key_mask = torch.zeros((1, d.shape[-1]), dtype=torch.bool).to(device)
# torch.from_numpy(np.asarray(key_mask))
        tgt_mask = model.get_tgt_mask(d.shape[-1]).to(device)
        src = model.encoder(src.transpose(1, 2)) # (1, 24, 256)
        src = model.pos_encoder(src) # (1, 24, 256)
        output = model.transformer_encoder(src, tgt_mask, key_mask)
        # (bs, len, dimension)
        output_t.append(output.transpose(1, 2).squeeze(0).detach().cpu().numpy())
    return output_t

def get_trans_encode(args, train_head, dev_head, test_head, weights, output_class=1):
    model = models.Trans_encoder(feature_dim=args.input_dim, d_model=args.d_model, nhead=args.n_head, d_hid=args.dim_ff_mul*args.d_model, \
                      nlayers=args.num_enc_layer, out_dim=output_class, dropout=args.dropout)  
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    train_encode = creat_trans_encode(model, train_head)
    dev_encode = creat_trans_encode(model, dev_head)
    test_encode = creat_trans_encode(model, test_head)

    return train_encode, dev_encode, test_encode

def get_trans_encode_t(args, test_head, weights, output_class=1):
    model = models.Trans_encoder(feature_dim=args.input_dim, d_model=args.d_model, nhead=args.n_head, d_hid=args.dim_ff_mul*args.d_model, \
                      nlayers=args.num_enc_layer, out_dim=output_class, dropout=args.dropout)  
    model.to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    # train_encode = creat_trans_encode(model, train_head)
    # dev_encode = creat_trans_encode(model, dev_head)
    test_encode = creat_trans_encode(model, test_head)

    return test_encode

def crop_data_target(vital, target_dict, static_dict, mode, target_index):
    '''
    vital: a list of nd array [[200, 81], [200, 93], ...] 
    target_dict: dict of SOFA score: {'train': {30015933: [81, 1], 30016009: [79, 1], ...}, 'dev': }
    static_dict: dict of static variables: {'static_train': a DataFrame (27136, 27) , 'static_dev':}
    variables in static dict: 'gender', 'age', 'hospital_expire_flag', 'max_hours',
       'myocardial_infarct', 'congestive_heart_failure',
       'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
       'chronic_pulmonary_disease', 'rheumatic_disease',
       'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
       'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
       'severe_liver_disease', 'metastatic_solid_tumor', 'aids',
       'ethnicity_AMERICAN INDIAN', 'ethnicity_ASIAN', 'ethnicity_BLACK',
       'ethnicity_HISPANIC/LATINO', 'ethnicity_OTHER', 'ethnicity_WHITE'
    return:
    train_filter: [ndarray with shape (200, 8),...
    sofa_tail: [ndarray with shape (8, 1),
    stayids: [39412629, 37943756, 32581623, 37929132,
    train_target: [0, 0, 1, 0, 0, 1]

    '''
    idx = pd.IndexSlice
    length = [i.shape[-1] for i in vital]
    train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m >24]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m >24]
    sofa_tail = [target_dict[mode][j][24:]/15 for j in stayids]
    static_key = 'static_' + mode
    # for eicu eicu_static['static_train'].loc[141168][1] becomes a value 
    if target_index == 21: # race: 2 is balck, 5 is white 
        # shape [1,6] then use nonzero, after e.g.array([5])
        train_target = [np.nonzero(static_dict[static_key].loc[idx[:, :, j]].iloc[:, 21:].values)[1] for j in stayids]
        sub_ind = [i for i, m in enumerate(train_target) if m == 2 or m == 5]
        race_dict = {2: 1, 5:0}
         # a list of target class
        train_targets = [race_dict[train_target[i][0]]for i in sub_ind]
        train_filters = [train_filter[i] for i in sub_ind]
        sofa_tails = [sofa_tail[i] for i in sub_ind]
        stayidss = [stayids[i] for i in sub_ind]
        return train_filters, sofa_tails, stayidss, train_targets
    elif target_index == 1: # age, binarize it
        # age median is 0.1097
        train_target = [static_dict[static_key].loc[idx[:, :, j]].iloc[:, 1].values[0] for j in stayids]
        train_target = [1 if i >= 0.1097 else 0 for i in train_target]
    else:
        # a list of target class
        train_target = [static_dict[static_key].loc[idx[:, :, j]].iloc[:, target_index].values[0] for j in stayids]
        
    return train_filter, sofa_tail, stayids, train_target

def crop_data_target_e(vital, target_dict, static_dict, mode, target_index):
    '''
    vital: a list of nd array [[200, 81], [200, 93], ...] 
    target_dict: dict of SOFA score: {'train': {30015933: [81, 1], 30016009: [79, 1], ...}, 'dev': }
    static_dict: dict of static variables: {'static_train': a DataFrame (27136, 27) , 'static_dev':}
    variables in static dict: 'gender', 'age', 'hospital_expire_flag', 'max_hours',
       'myocardial_infarct', 'congestive_heart_failure',
       'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
       'chronic_pulmonary_disease', 'rheumatic_disease',
       'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
       'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
       'severe_liver_disease', 'metastatic_solid_tumor', 'aids',
       'ethnicity_AMERICAN INDIAN', 'ethnicity_ASIAN', 'ethnicity_BLACK',
       'ethnicity_HISPANIC/LATINO', 'ethnicity_OTHER', 'ethnicity_WHITE'
    return:
    train_filter: [ndarray with shape (200, 8),...
    sofa_tail: [ndarray with shape (8, 1),
    stayids: [39412629, 37943756, 32581623, 37929132,
    train_target: [0, 0, 1, 0, 0, 1]

    '''
    # idx = pd.IndexSlice
    length = [i.shape[-1] for i in vital]
    train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m >24]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m >24]
    sofa_tail = [target_dict[mode][j][24:]/15 for j in stayids]
    static_key = 'static_' + mode
    # for eicu eicu_static['static_train'].loc[141168][1] becomes a value 
    if target_index == 21: # race: 2 is balck, 5 is white 
        # shape [1,6] then use nonzero, after e.g.array([5])
        train_target = [np.nonzero(static_dict[static_key].loc[j][21:].values)[0] for j in stayids]
        sub_ind = [i for i, m in enumerate(train_target) if m == 2 or m == 5]
        race_dict = {2: 1, 5:0}
         # a list of target class
        train_targets = [race_dict[train_target[i][0]]for i in sub_ind]
        train_filters = [train_filter[i] for i in sub_ind]
        sofa_tails = [sofa_tail[i] for i in sub_ind]
        stayidss = [stayids[i] for i in sub_ind]
        return train_filters, sofa_tails, stayidss, train_targets

    elif target_index == 1: # age, binarize it
        # age median is 0.1097
        train_target = [static_dict[static_key].loc[j][1] for j in stayids]
        train_target = [1 if i >= 0.1097 else 0 for i in train_target]
    else:
        # a list of target class
        train_target = [static_dict[static_key].loc[j][target_index] for j in stayids]
    
    if target_index == 0: # eicu gender has unknown:
        known_ids = [k for k, i in enumerate(stayids) if train_target[k] != 2.0]
        train_filter = [train_filter[i] for i in known_ids]
        sofa_tail = [sofa_tail[i] for i in known_ids]
        stayids= [stayids[i] for i in known_ids]
        train_target= [train_target[i] for i in known_ids]
        
        return train_filter, sofa_tail, stayids, train_target

    return train_filter, sofa_tail, stayids, train_target

def filter_sepsis(vital, sofa, ids, target):
    id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/mimic_sepsis3.csv')
    sepsis3_id = id_df['stay_id'].values # 1d array 
    index_dict = dict((value, idx) for idx,value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    target_sepsis = [target[i] for i in ind]
    return vital_sepsis, sofa_sepsis, [ids[i] for i in ind], target_sepsis

def filter_sepsis_e(vital, sofa, ids, target):
    id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/eicu_sepsis3.csv')
    sepsis3_id = id_df['patientunitstayid'].values # 1d array 
    index_dict = dict((value, idx) for idx,value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    target_sepsis = [target[i] for i in ind]
    return vital_sepsis, sofa_sepsis, [ids[i] for i in ind], target_sepsis

def get_cv_data(train_data, dev_data, train_target, dev_target, train_index, dev_index):
    trainval_head = train_data + dev_data
    trainval_static = np.concatenate((train_target, dev_target), axis=0)
    train_cv = [trainval_head[i] for i in train_index]
    train_cvl = [trainval_static[i] for i in train_index]
    dev_cv = [trainval_head[i] for i in dev_index]
    dev_cvl = [trainval_static[i] for i in dev_index]
    return train_cv, dev_cv, np.asarray(train_cvl), np.asarray(dev_cvl)

def cal_acc(pred, label):
    pred_t = torch.concat(pred)
    prediction =  torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    acc = (prediction == label_t).sum()/len(pred_t)
    return acc

def cal_pos_acc(pred, label, pos_ind):
    pred_t = torch.concat(pred)
    prediction =  torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    # positive index
    ind = [i for i in range(len(pred_t)) if label_t[i] == pos_ind]
    acc = (prediction[ind] == label_t[ind]).sum()/len(ind)
    return acc






