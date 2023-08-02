import torch.utils.data as data
from torch.utils.data import Sampler
import torch 
import random 
import numpy as np 
# when target is race, the sampler does specially 
# treatment: 3: under sample white 3, black is 1, latino is 2 asian is 0
# for eval: no need to under sample white 
class RaceTrainSampler(Sampler):

    def __init__(self, args, data_source, label, 
                bucket_boundaries, batch_size=64):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[1]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.label = label 
        self.args = args
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            # each data bucket is a list of [0, 5, 6, 8, ...], [1, 3, 4....]
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            # if self.args.race_sample == 1: 
            # # use class label 1 as the lower basis for white 
            # #num of 1s which is black 
            #     ind_pos =  np.asarray([i for i in data_buckets[k] if self.label[i] == 1 ])
            #     num_pos= len(ind_pos)
            #     # other is 0: asian and 2: hispanic
            #     ind_other =  np.asarray([i for i in data_buckets[k] if self.label[i] == 0])
            #     # 3 is white, which needs to be downsampled 
            #     ind_neg =  [i for i in data_buckets[k] if self.label[i] == 2 ]
            #     num_neg = min(max(1, num_pos), len(ind_neg))

            #     neg_choice = np.random.choice(ind_neg, num_neg, replace=False)

            #     data_buckets[k] = np.concatenate((ind_pos, neg_choice, ind_other))

            # elif self.args.race_sample == 0:  # use class label 0 asian as the lower bound 
                # after adjustment, 0 is black, 1 is white 
                # ind_pos =  np.asarray([i for i in data_buckets[k] if self.label[i] == 1 ])
                
                # other is 0: asian and 2: hispanic
                # print(data_buckets)
                # print(len(self.label))
            ind_black =  np.asarray([i for i in data_buckets[k] if self.label[i] == 1])
            num_black= len(ind_black)

            # ind_hispanic =  np.asarray([i for i in data_buckets[k] if self.label[i] == 2])

            # 2 is white, which needs to be downsampled 
            # 1 is white, which needs to be downsampled 
            ind_neg =  [i for i in data_buckets[k] if self.label[i] == 0 ]

            num_neg = min(max(1, num_black), len(ind_neg)) # white
            # num_pos = min(max(1, num_asian), len(ind_pos)) # black
            # downsample bothe neg and pos 
            neg_choice = np.random.choice(ind_neg, num_neg, replace=False)
            # pos_choice = np.random.choice(ind_pos, num_pos, replace=False)

            data_buckets[k] = np.concatenate((neg_choice, ind_black))

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

class TrainSampler(Sampler):

    def __init__(self, data_source, label, 
                bucket_boundaries, batch_size=64):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[1]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.label = label 
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            # each data bucket is a list of [0, 5, 6, 8, ...], [1, 3, 4....]
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            #num of 1s
            ind_pos =  np.asarray([i for i in data_buckets[k] if self.label[i] == 1 ])
            num_pos= len(ind_pos)
            ind_neg =  [i for i in data_buckets[k] if self.label[i] == 0 ]
            num_neg = min(max(1, num_pos), len(ind_neg))
            neg_choice = np.random.choice(ind_neg, num_neg, replace=False)
            data_buckets[k] = np.concatenate((ind_pos, neg_choice))
            # make sure there is equal number of 0s and 1s
            
        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

# eval sampler doesn't need to resample negative class
class EvalSampler(Sampler):

    def __init__(self, data_source, label, 
                bucket_boundaries, batch_size=64):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[1]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.label = label 
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            # each data bucket is a list of [0, 5, 6, 8, ...], [1, 3, 4....]
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])
            
        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

    
class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, data, target, static=None):

        
        # self.ti_data = ti_data
        self.data = data
        self.static = static 
        self.target = target

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        data, target = self.data[index], self.target[index]
        data = np.float32(data)
        target = np.uint8(target)

        if not self.static:
            return data, target
        else:
            static = self.static[index]
            static = np.float32(static)
            return data, static, target

    def __len__(self):
        return len(self.target)

class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,  
                bucket_boundaries, batch_size=64,):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[1]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

def col_fn(batchdata):
# dat = [train_dataset[i] for i in range(32)]
    len_data = len(batchdata)  
    # in batchdata, shape [(182, 48)]
    seq_len = [batchdata[i][0].shape[-1] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    len_tem = [np.zeros((batchdata[i][0].shape[-1])) for i in range(len_data)]
    max_len = max(seq_len)

    # [(182, 48) ---> (182, 100)]
    padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                   mode='constant', constant_values=-3) for i in range(len_data)]
    # [0, 1, 0, 0, 0, ...]
    padded_label = [batchdata[i][1] for i in range(len_data)]

    # [(48, ) ---> (100, )]
    mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
               mode='constant', constant_values=1) for i in range(len_data)]
    

    return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.asarray(padded_label)).unsqueeze(-1),\
            torch.from_numpy(np.stack(mask))

def generate_buckets(args, bs, train_hist):
    buckets = []
    sum = 0
    for i in range(217): 
        # train hist 216 is [216, 217), 217 is [217, 218) is 0
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i+1)
            sum = 0 
    # residue is 58 < 128, remove index 205, attach 217, largest is 216
    if sum < bs:
        buckets.pop(-1)
        buckets.append(218)
    return buckets

def get_data_loader(args, train_head, dev_head, test_head, \
                train_sofa_tail, dev_sofa_tail, test_sofa_tail):
    
    train_dataset = Dataset(train_head, train_sofa_tail)
    val_dataset = Dataset(dev_head, dev_sofa_tail)
    test_dataset = Dataset(test_head, test_sofa_tail)

    train_len = [train_head[i].shape[1] for i in range(len(train_head))]
    # start_bin 
    bin_start = 0 
    len_range = [i for i in range(218)]
    train_hist, _ = np.histogram(train_len, bins=len_range)

    if args.data_batching == 'random':


        train_dataloader = data.DataLoader(train_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)  

        dev_dataloader = data.DataLoader(val_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False) 

        test_dataloader = data.DataLoader(test_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False) 
        
    elif args.data_batching == 'same':
        # same is not that useful for this since 6 is just too small without resampling 
        batch_sizes=6
        val_batch_sizes = 1
        test_batch_sizes = 1

        bucket_boundaries = [i for i in range(1, 218)]
        val_bucket_boundaries = [k for k in bucket_boundaries if k not in [129, 168, 188, 201,206]]

        sampler = TrainSampler(train_head, bucket_boundaries, batch_sizes)
        dev_sampler = EvalSampler(dev_head, val_bucket_boundaries, val_batch_sizes)
        test_sampler = EvalSampler(test_head, bucket_boundaries, test_batch_sizes)

        train_dataloader = data.DataLoader(train_dataset, batch_size=1, 
                                batch_sampler=sampler, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)

        dev_dataloader = data.DataLoader(val_dataset, batch_size=1, 
                                batch_sampler=dev_sampler, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, 
                                batch_sampler=test_sampler, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)

    elif args.data_batching == 'close':

        batch_sizes= args.bs
        val_batch_sizes = args.bs
        test_batch_sizes = args.bs

        # bucket_boundaries = [i for i in range(1, 34)]
        # calcuated 
        # bucket_boundaries  = bucket_boundaries + [36, 39, 42, 45, 47, 49, 52, 55, 58, 62, 66, \
        #                                                70, 73, 76, 80, 86, 91, 95, 99, 104, 113, 119,\
        #                                                125, 135, 144, 152, 164, 176, 190, 203, 217]
        
        bucket_boundaries = generate_buckets(args, args.bucket_size, train_hist)
        # train hist 0 is [0, 1) which is zero, train hist[-1] is [217, 218) which is also 0, not interated in generate_buckets func
        # val_bucket_boundaries = [k for k in range(1, 218) if k not in [129, 168, 188, 201, 206]]
        if args.infer_ind == 1:
            # ne need to resample if using age as a target
            sampler = EvalSampler(train_head, train_sofa_tail, bucket_boundaries, batch_sizes)
        elif args.infer_ind == 2:
            # if race as the target 
            print(len(train_head))
            sampler = RaceTrainSampler(args, train_head, train_sofa_tail, bucket_boundaries, batch_sizes)
        else:
            sampler = TrainSampler(train_head, train_sofa_tail, bucket_boundaries, batch_sizes)
        
        dev_sampler = EvalSampler(dev_head, dev_sofa_tail, bucket_boundaries, val_batch_sizes)
        test_sampler = EvalSampler(test_head, test_sofa_tail, bucket_boundaries, test_batch_sizes)

        train_dataloader = data.DataLoader(train_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=sampler, 
                                drop_last=False, pin_memory=False)

        dev_dataloader = data.DataLoader(val_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=dev_sampler, 
                                drop_last=False, pin_memory=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=test_sampler, 
                                drop_last=False, pin_memory=False)
        
    return train_dataloader, dev_dataloader, test_dataloader

def get_huge_dataloader(args, train_head, dev_head, test_head, \
                train_sofa_tail, dev_sofa_tail, test_sofa_tail):
    
    total_head = train_head + dev_head + test_head
    total_target = np.concatenate((train_sofa_tail, dev_sofa_tail, test_sofa_tail), axis=0)
    train_len = [total_head[i].shape[1] for i in range(len(train_head))]
    # start_bin 
    # bin_start = 0 if args.data_version == 12 else 24
    len_range  = [i for i in range(218)]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    
    bucket_boundaries = generate_buckets(args, args.bucket_size, train_hist)

    train_dataset = Dataset(total_head, total_target)
    sampler = BySequenceLengthSampler(total_head, bucket_boundaries, args.bs)
    dataloader = data.DataLoader(train_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=sampler, 
                                drop_last=False, pin_memory=False)
    return dataloader

def get_test_dataloader_only(args, test_head, test_sofa_tail):

    test_dataset = Dataset(test_head, test_sofa_tail)
    test_batch_sizes = args.bs
    
    train_len = [test_head[i].shape[1] for i in range(len(test_head))]
    # start_bin 
    bin_start = 0 
    len_range = [i for i in range(218)]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    bucket_boundaries = generate_buckets(args, 64, train_hist)

    test_sampler = EvalSampler(test_head, test_sofa_tail, bucket_boundaries, test_batch_sizes)

    test_dataloader = data.DataLoader(test_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=test_sampler, 
                                drop_last=False, pin_memory=False)
        
    return test_dataloader
    

def get_test_dataloader_only_e(args, test_head, test_sofa_tail):

    test_dataset = Dataset(test_head, test_sofa_tail)
    test_batch_sizes = args.bs
    
    train_len = [test_head[i].shape[1] for i in range(len(test_head))]
    # start_bin 
    bin_start = 0 
    len_range = [i for i in range(218)]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    bucket_boundaries = generate_buckets(args, 640, train_hist)

    test_sampler = EvalSampler(test_head, test_sofa_tail, bucket_boundaries, test_batch_sizes)

    test_dataloader = data.DataLoader(test_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=test_sampler, 
                                drop_last=False, pin_memory=False)
        
    return test_dataloader

            