import pandas as pd
import numpy as np 
import torch 
import importlib
import loss_fn
importlib.reload(loss_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1D ALE 
def get_centres(x):
    """Return bin centres from bin edges.
    Parameters
    ----------
    x : array-like
        The first axis of `x` will be averaged.
    Returns
    -------
    centres : array-like
        The centres of `x`, the shape of which is (N - 1, ...) for
        `x` with shape (N, ...).
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> _get_centres(x)
    array([0.5, 1.5, 2.5])
    """
    return (x[1:] + x[:-1]) / 2

def data_piece(train_set_rep, index, record_flag):
    train_set_piece = []
    all_features = []
    # when >=164# something here needs to be fixed
    # INDEX 162, ph urine mean, last numerical 
    offset = 1 if index<=162 else 0 
    for record in train_set_rep:
        true_record = np.where(record[index+offset, :] == record_flag)[0]
        if len(true_record) >= 1:
            for rec_ind in true_record:
                train_set_piece.append(record[:, :rec_ind+1])
                all_features.append(record[index, :rec_ind+1])
    return train_set_piece, np.concatenate(all_features)

def get_1d_ale(model, test_head, index, bins, monte_carlo_ratio, monte_carlo_rep, record_flag=1):
    mc_replicates = np.asarray(
                    [
                        [
                            np.random.choice(range(len(test_head)))
                            for _ in range(int(monte_carlo_ratio * len(test_head)))
                        ]
                        for _ in range(monte_carlo_rep)
                    ])
    ale_nc = []
    quantile_nc = []
    ale_t = []
    quantile_t = []   
    for k, rep in enumerate(mc_replicates):
        train_set_rep = [test_head[i] for i in rep]
        train_set_piece, lin_feature = data_piece(train_set_rep, index, record_flag=record_flag)

        quantiles = np.unique(
            np.quantile(
                lin_feature, np.linspace(0, 1, bins + 1), interpolation="lower"
            )
        )
        bins = len(quantiles) - 1
        piece_shape = [train_set_piece[i].shape[1] for i in range(len(train_set_piece))]
        piece_hist, _ = np.histogram(piece_shape, bins = range(0, 219))

        len_dict = {}
        # key: lenth, value: a list of pieces 
        for k, j in enumerate(piece_shape):
            if j in len_dict.keys():
                len_dict[j].append(train_set_piece[k])
            else: 
                len_dict[j] = [train_set_piece[k]]

        piece_3d = []
        for i in len_dict.keys():
            piece_3d.append(np.stack(len_dict[i]))

        model.eval()
        effects_t = []
        indices_t = []

        for piece in piece_3d: 
        # [(200, 5), (200, 6), (200, 7), (200, 5)]
        # train_set_piece could be 20000 pieces, given there are 100 repetitions 
        # gdigitalize using [(6, 200, 1), (10, 200, 5) ...] piece_3d, last dim is lenth 
            indices = np.clip(
                    np.digitize(piece[:, index, -1], quantiles, right=True) - 1, 0, None
                )
            predictions = []
            for offset in range(2):
                mod_train_set = piece.copy()
                mod_train_set[:, index, -1] = quantiles[indices + offset]
                predictions.append(model(torch.FloatTensor(mod_train_set).to(device)))  # (6, 1, 1) or (6, 5, 1) depending on the length 
            # The individual effects.
            # (139, 60, 1) diffrent indices  (139)
            effects = np.subtract(predictions[1][:, -1, :].cpu().detach().numpy(), predictions[0][:, -1, :].cpu().detach().numpy())
            effects_t.append(effects)
            indices_t.append(indices)

        index_groupby = pd.DataFrame({"index": np.concatenate(indices_t, axis=0), \
                                "effects": np.concatenate(effects_t, axis=0).squeeze(-1)}).groupby("index")
        mean_effects = index_groupby.mean().to_numpy().flatten()
        ale = np.array([0, *np.cumsum(mean_effects)])
        ale_nc.append(ale)
        quantile_nc.append(quantiles)
        # The uncentred mean main effects at the bin centres.
        ale = get_centres(ale)

        # Centre the effects by subtracting the mean (the mean of the individual
        # `effects`, which is equivalently calculated using `mean_effects` and the number
        # of samples in each bin).
        ale -= np.sum(ale * index_groupby.size() / len(train_set_piece))
        ale_t.append(ale)
        quantile_t.append(get_centres(quantiles))

    return quantile_t, ale_t, quantile_nc, ale_nc

def get_1d_ale_cat(model, test_head, index, bins, monte_carlo_ratio, monte_carlo_rep, record_flag=1):
    mc_replicates = np.asarray(
                    [
                        [
                            np.random.choice(range(len(test_head)))
                            for _ in range(int(monte_carlo_ratio * len(test_head)))
                        ]
                        for _ in range(monte_carlo_rep)
                    ])
    ale_nc = []
    quantile_nc = []
    ale_t = []
    quantile_t = [] 
    mean_effects = []
    for k, rep in enumerate(mc_replicates):
        train_set_rep = [test_head[i] for i in rep]
        train_set_piece, lin_feature = data_piece(train_set_rep, index, record_flag=record_flag)

        # quantiles = np.unique(
        #     np.quantile(
        #         lin_feature, np.linspace(0, 1, bins + 1), interpolation="lower"
        #     )
        # )
        # bins = len(quantiles) - 1
        piece_shape = [train_set_piece[i].shape[1] for i in range(len(train_set_piece))]
        piece_hist, _ = np.histogram(piece_shape, bins = range(0, 219))

        len_dict = {}
        # key: lenth, value: a list of pieces 
        for k, j in enumerate(piece_shape):
            if j in len_dict.keys():
                len_dict[j].append(train_set_piece[k])
            else: 
                len_dict[j] = [train_set_piece[k]]

        piece_3d = []
        for i in len_dict.keys():
            piece_3d.append(np.stack(len_dict[i]))

        model.eval()
        effects_t = []
        indices_t = []

        for piece in piece_3d: 
        # [(200, 5), (200, 6), (200, 7), (200, 5)]
        # train_set_piece could be 20000 pieces, given there are 100 repetitions 
        # gdigitalize using [(6, 200, 1), (10, 200, 5) ...] piece_3d, last dim is lenth 
            # indices = np.clip(
            #         np.digitize(piece[:, index, -1], quantiles, right=True) - 1, 0, None
            #     )
            predictions = []
            for offset in range(2):
                mod_train_set = piece.copy()
                mod_train_set[:, index, -1] = offset
                predictions.append(model(torch.FloatTensor(mod_train_set).to(device)))  # (6, 1, 1) or (6, 5, 1) depending on the length 
            # The individual effects.
            # (139, 60, 1) diffrent indices  (139)
            effects = np.subtract(predictions[1][:, -1, :].cpu().detach().numpy(), predictions[0][:, -1, :].cpu().detach().numpy())
            effects_t.append(effects)
            # indices_t.append(indices)

        # index_groupby = pd.DataFrame({"index": np.concatenate(indices_t, axis=0), \
        #                         "effects": np.concatenate(effects_t, axis=0).squeeze(-1)}).groupby("index")
        mean_effects.append(np.concatenate(effects_t, axis=0))
        # ale = np.array([0, *np.cumsum(mean_effects)])
        # ale_nc.append(ale)
        # quantile_nc.append(quantiles)
        # The uncentred mean main effects at the bin centres.
        # ale = get_centres(ale)

        # Centre the effects by subtracting the mean (the mean of the individual
        # `effects`, which is equivalently calculated using `mean_effects` and the number
        # of samples in each bin).
        # ale -= np.sum(ale * index_groupby.size() / len(train_set_piece))
        # ale_t.append(ale)
        # effects_tt.append(mean_effects)
        # quantile_t.append(get_centres(quantiles))
    return mean_effects

def data_piece_latent(train_set_rep, index, latency, record_flag):
    train_set_piece = []
    all_features = []
    # when >=164# something here needs to be fixed
    # INDEX 162, ph urine mean, last numerical 
    offset = 1 if index<=162 else 0 
    for record in train_set_rep:
        len_r = record.shape[1]
        true_record = np.where(record[index+offset, :] == record_flag)[0]
        if len(true_record) >= 1:
            for rec_ind in true_record:
                # chech if data length is enough 
                if rec_ind + latency < len_r:
                    train_set_piece.append(record[:, :(rec_ind + latency+1)])
                    all_features.append(record[index, :(rec_ind + latency+1)])
    return train_set_piece, np.concatenate(all_features)

def get_1d_ale_cat_latent(model, test_head, index, bins, monte_carlo_ratio, monte_carlo_rep, latency, record_flag=1):
    mc_replicates = np.asarray(
                    [
                        [
                            np.random.choice(range(len(test_head)))
                            for _ in range(int(monte_carlo_ratio * len(test_head)))
                        ]
                        for _ in range(monte_carlo_rep)
                    ])
    ale_nc = []
    quantile_nc = []
    ale_t = []
    quantile_t = [] 
    mean_effects = []
    for k, rep in enumerate(mc_replicates):
        train_set_rep = [test_head[i] for i in rep]
        train_set_piece, lin_feature = data_piece_latent(train_set_rep, index, latency, record_flag=record_flag)

        # quantiles = np.unique(
        #     np.quantile(
        #         lin_feature, np.linspace(0, 1, bins + 1), interpolation="lower"
        #     )
        # )
        # bins = len(quantiles) - 1
        piece_shape = [train_set_piece[i].shape[1] for i in range(len(train_set_piece))]
        piece_hist, _ = np.histogram(piece_shape, bins = range(0, 219))

        len_dict = {}
        # key: lenth, value: a list of pieces 
        for k, j in enumerate(piece_shape):
            if j in len_dict.keys():
                len_dict[j].append(train_set_piece[k])
            else: 
                len_dict[j] = [train_set_piece[k]]

        piece_3d = []
        for i in len_dict.keys():
            piece_3d.append(np.stack(len_dict[i]))

        model.eval()
        effects_t = []
        indices_t = []

        for piece in piece_3d: 
        # [(200, 5), (200, 6), (200, 7), (200, 5)]
        # train_set_piece could be 20000 pieces, given there are 100 repetitions 
        # gdigitalize using [(6, 200, 1), (10, 200, 5) ...] piece_3d, last dim is lenth 
            # indices = np.clip(
            #         np.digitize(piece[:, index, -1], quantiles, right=True) - 1, 0, None
            #     )
            predictions = []
            for offset in range(2):
                mod_train_set = piece.copy()
                mod_train_set[:, index, -(1+latency)] = offset
                predictions.append(model(torch.FloatTensor(mod_train_set).to(device)))  # (6, 1, 1) or (6, 5, 1) depending on the length 
            # The individual effects.
            # (139, 60, 1) diffrent indices  (139)
            effects = np.subtract(predictions[1][:, -1, :].cpu().detach().numpy(), predictions[0][:, -1, :].cpu().detach().numpy())
            effects_t.append(effects)
            # indices_t.append(indices)

        # index_groupby = pd.DataFrame({"index": np.concatenate(indices_t, axis=0), \
        #                         "effects": np.concatenate(effects_t, axis=0).squeeze(-1)}).groupby("index")
        mean_effects.append(np.concatenate(effects_t, axis=0))
        # ale = np.array([0, *np.cumsum(mean_effects)])
        # ale_nc.append(ale)
        # quantile_nc.append(quantiles)
        # The uncentred mean main effects at the bin centres.
        # ale = get_centres(ale)

        # Centre the effects by subtracting the mean (the mean of the individual
        # `effects`, which is equivalently calculated using `mean_effects` and the number
        # of samples in each bin).
        # ale -= np.sum(ale * index_groupby.size() / len(train_set_piece))
        # ale_t.append(ale)
        # effects_tt.append(mean_effects)
        # quantile_t.append(get_centres(quantiles))
    return mean_effects
# get evaluation results

def get_eval_results(args, model, test_loader):
    
    model.eval()
    y_list = []
    y_pred_list = []
    td_list = []
    loss_val = []
    with torch.no_grad():  # validation does not require gradient

        for test_dl in test_loader:
            if len(test_dl) ==3: 
                vitals, target, key_mask = test_dl
                if args.model_name == 'TCN':
                    sofap_t = model(vitals.to(device))
                elif args.model_name == 'RNN':
                    # x_lengths have to be a 1d tensor 
                    td_transpose = vitals.to(device).transpose(1, 2)
                    x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                    sofap_t = model(td_transpose, x_lengths)
                elif args.model_name == 'Transformer':
                    tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                    sofap_t = model(vitals.to(device), tgt_mask, key_mask.to(device))

            elif len(test_dl) ==4: 
                vitals, target, test_ids, key_mask = test_dl
                sofap_t = model(vitals.to(device))
                # for read static model 
                # vitals, static, target, key_mask = test_dl 
                # sofap_t = model(vitals.to(device), static.to(device))
                        
            
            # ti_test = Variable(torch.FloatTensor(ti)).to(device)
            # td_test = Variable(torch.FloatTensor(vitals)).to(device)
            # sofa_t = Variable(torch.FloatTensor(target)).to(device)

            # tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
            # sofap_t = model(td_test, tgt_mask_test, key_mask.to(device))
            else:
                vitals, static, target, test_ids, key_mask = test_dl 
                sofap_t = model(vitals.to(device), static.to(device))

            
            loss_v = loss_fn.mse_maskloss(sofap_t, target.to(device), key_mask.to(device))
            y_list.append([target[i][key_mask[i]==0].detach().numpy() for i in range(len(target))])
            y_pred_list.append([sofap_t[i][key_mask[i]==0].cpu().detach().numpy() for i in range(len(target))])
            loss_val.append(loss_v)
            td_list.append(vitals.detach().numpy())
    
    loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())

    return y_list, y_pred_list, td_list, loss_te

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
