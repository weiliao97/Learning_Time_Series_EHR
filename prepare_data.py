import torch.utils.data as data
from torch.utils.data import Sampler, ConcatDataset, Subset
import torch 
import random 
import numpy as np 

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, data, target, static = None, stayid = None):

        
        # self.ti_data = ti_data
        self.data = data
        self.target = target
        self.static = static 
        self.stayid = stayid

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        if self.static is not None and self.stayid is None:

            data, static, target = self.data[index], self.static[index], self.target[index]
        elif self.static is None and self.stayid is not None:
            data, target, stayid = self.data[index], self.target[index], self.stayid[index]
        elif self.static is not None and self.stayid is not None:
            data, static, target, stayid = self.data[index], self.static[index], self.target[index], self.stayid[index]
        else:
            data, target = self.data[index], self.target[index]
      
        # img = img.type(torch.FloatTensor)

        data = np.float32(data)
        # td_data = np.float32(td_data)
        target = np.float32(target)

        if self.static is not None and self.stayid is None:
            static = np.float32(static)
            return data, static, target        
        elif self.static is None and self.stayid is not None:
            return data, target, stayid
        elif self.static is not None and self.stayid is not None:
            static = np.float32(static)
            return data, static, target, stayid
        else:
            return data, target            
        # # class_target = np.long(class_target)
        # class_target = np.float32(class_target) # for noise data model 

    def __len__(self):
        return len(self.target)

class Dataset_Head(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, data, static, target, head):

        
        # self.ti_data = ti_data
        self.data = data
        self.static = static 
        self.target = target
        self.head = head 

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
 
        data, static, target, head= self.data[index], self.static[index], self.target[index], self.head[index]

        
        # img = img.type(torch.FloatTensor)

        data = np.float32(data)
        # td_data = np.float32(td_data)
        static = np.float32(static)
        target = np.float32(target)
        head = np.float32(head)
        # # class_target = np.long(class_target)
        # class_target = np.float32(class_target) # for noise data model 

        return data, static, target, head

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
    variety_data = len(batchdata[0])
    # in batchdata, shape [(182, 48)]
    seq_len = [batchdata[i][0].shape[-1] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    len_tem = [np.zeros((batchdata[i][0].shape[-1])) for i in range(len_data)]
    max_len = max(seq_len)
    # whether static is in it or not 
    if variety_data == 3: # vital. static, target or vital. target, stayid
        # [(182, 48) ---> (182, 100)]
        if not isinstance(batchdata[0][-1], int): # target is the last: 
            padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                        mode='constant', constant_values=-3) for i in range(len_data)]
            # [(48, 1) ---> (100, 1)]
            padded_label = [np.pad(batchdata[i][2], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
                        mode='constant', constant_values=0) for i in range(len_data)]
            # 
            static = [batchdata[i][1] for i in range(len_data)]
            
            # [(48, ) ---> (100, )]
            mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
                    mode='constant', constant_values=1) for i in range(len_data)]
            
            return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.stack(static)), \
                torch.from_numpy(np.asarray(padded_label)), torch.from_numpy(np.stack(mask))
        else:
            padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
            mode='constant', constant_values=-3) for i in range(len_data)]
            # [(48, 1) ---> (100, 1)]
            padded_label = [np.pad(batchdata[i][1], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
                        mode='constant', constant_values=0) for i in range(len_data)]
            # 
            stayids = [batchdata[i][2] for i in range(len_data)]
            
            # [(48, ) ---> (100, )]
            mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
                    mode='constant', constant_values=1) for i in range(len_data)]
            
            return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.asarray(padded_label)), \
            torch.from_numpy(np.asarray(stayids)), torch.from_numpy(np.stack(mask))

        
    elif variety_data == 2:

        # [(182, 48) ---> (182, 100)]
        padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                    mode='constant', constant_values=-3) for i in range(len_data)]
        # [(48, 1) ---> (100, 1)]
        padded_label = [np.pad(batchdata[i][1], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
                    mode='constant', constant_values=0) for i in range(len_data)]
        # [(48, ) ---> (100, )]
        mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
                mode='constant', constant_values=1) for i in range(len_data)]
        

        return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.stack(padded_label)),\
                torch.from_numpy(np.stack(mask))
    
    elif variety_data == 4:
                # [(182, 48) ---> (182, 100)]
        padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                    mode='constant', constant_values=-3) for i in range(len_data)]
        # [(48, 1) ---> (100, 1)]
        padded_label = [np.pad(batchdata[i][2], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
                    mode='constant', constant_values=0) for i in range(len_data)]
        # 
        static = [batchdata[i][1] for i in range(len_data)]
        stayids = [batchdata[i][3] for i in range(len_data)]
        
        # [(48, ) ---> (100, )]
        mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
                mode='constant', constant_values=1) for i in range(len_data)]
        
        return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.stack(static)), \
            torch.from_numpy(np.asarray(padded_label)), torch.from_numpy(np.asarray(stayids)), torch.from_numpy(np.stack(mask))



def generate_buckets(bs, train_hist):
    buckets = []
    sum = 0
    s = 0
    for i in range(0, 218): 
        # train_hist len 218, 
        # train_hist[0] is len [0, 1), train_hist[217] is  [217, 218]
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i)
            sum = 0 
    # residue is 58 < 128, remove index 205, attach 217, largest is 216
    if sum < bs:
        buckets.pop(-1)    
    buckets.append(219)
    return buckets

def get_data_loader(args, train_head, dev_head, test_head, \
                train_sofa_tail, dev_sofa_tail, test_sofa_tail, 
                train_static = None, dev_static = None, test_static = None, 
                train_id = None, dev_id = None, test_id = None):
    
    train_dataset = Dataset(train_head, train_sofa_tail, static = train_static, stayid = train_id)
    val_dataset = Dataset(dev_head, dev_sofa_tail, static = dev_static, stayid = dev_id)
    test_dataset = Dataset(test_head, test_sofa_tail, static = test_static, stayid = test_id)

    train_len = [train_head[i].shape[1] for i in range(len(train_head))]
    val_len = [dev_head[i].shape[1] for i in range(len(dev_head))]
    len_range = [i for i in range(0, 219)]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    val_hist, _ = np.histogram(val_len, bins=len_range)

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

        bucket_boundaries = [i for i in range(1, 219)]
        val_bucket_boundaries = [i for i in range(len(val_hist)) if val_hist[i]>0 ] + [219]

        sampler = BySequenceLengthSampler(train_head, bucket_boundaries, batch_sizes)
        dev_sampler = BySequenceLengthSampler(dev_head, val_bucket_boundaries, val_batch_sizes)
        test_sampler = BySequenceLengthSampler(test_head, bucket_boundaries, test_batch_sizes)

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
        val_batch_sizes = 2
        test_batch_sizes = 2

        # bucket_boundaries = [i for i in range(1, 34)]
        # calcuated 
        # bucket_boundaries  = bucket_boundaries + [36, 39, 42, 45, 47, 49, 52, 55, 58, 62, 66, \
        #                                                70, 73, 76, 80, 86, 91, 95, 99, 104, 113, 119,\
        #                                                125, 135, 144, 152, 164, 176, 190, 203, 217]
        
        bucket_boundaries = generate_buckets(args.bucket_size, train_hist)
        # val_bucket_boundaries = [i for i in range(len(val_hist)) if val_hist[i]>0 ] + [219]
        
        sampler = BySequenceLengthSampler(train_head, bucket_boundaries, batch_sizes)
        
        dev_sampler = BySequenceLengthSampler(dev_head, bucket_boundaries, val_batch_sizes)
        test_sampler = BySequenceLengthSampler(test_head, bucket_boundaries, test_batch_sizes)

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
    bin_start = 0 
    len_range = [i for i in range(bin_start, 219+bin_start)]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    
    bucket_boundaries = generate_buckets(args.bucket_size, train_hist)

    train_dataset = Dataset(total_head, total_target)
    sampler = BySequenceLengthSampler(total_head, bucket_boundaries, args.bs)
    dataloader = data.DataLoader(train_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=sampler, 
                                drop_last=False, pin_memory=False)
    return dataloader






