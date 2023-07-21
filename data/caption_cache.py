from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, OrderedDict
import pickle
from multiprocessing import Manager

################################################################################################################
class CaptionCache(Dataset):
    def __init__(self, shared_dict, length):
        self.shared_dict = shared_dict
        self.length = length

    def __getitem__(self, index, imagepath, client_backup, client_backup2, K = None):
        if index not in self.shared_dict:
            if index == 0: print('Adding {} to shared_dict'.format(index))
            res = client_backup.query(image=imagepath)
            if len(res) < self.length:
                res = client_backup2.query(image=imagepath)
            if isinstance(res, list) and len(res) >= self.length:
                self.shared_dict[index] = [each['caption'] for each in res[:self.length]]
            else:
                self.shared_dict[index] = None
        else:
            if index == 0: print("use cache")
        if self.shared_dict[index]: return self.shared_dict[index][:K]
        else: return None

    def __len__(self):
        return self.length

def create_cache(path=None):
    if path:
        with open(path, 'rb') as f:
            cap_cache_dict = pickle.load(f)
        manager = Manager()
        shared_dict = manager.dict(cap_cache_dict)
    else:
        manager = Manager()
        shared_dict = manager.dict()
    cap_cache = CaptionCache(shared_dict, length=128)
    return cap_cache
################################################################################################################