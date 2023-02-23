import os
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset
from data.laserscan import SemLaserScan
# from laserscan import SemLaserScan

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class KittiDataset(Dataset):
    def __init__(self, data_root='dataset', sequence=['00'],config='None', num_point=4096):
        super().__init__()
        # load parameters
        self.num_point = num_point
        # open config file
        try:
            print("Opening config file %s" % config)
            CFG = yaml.safe_load(open(config, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()

        # load path
        self.scan_names, self.label_names = [], []
        for seq in sequence:
            scan_paths = os.path.join(data_root, "sequences", seq, "velodyne")
            self.scan_names += [os.path.join(root, f) for root, dirs, files in os.walk(os.path.expanduser(scan_paths)) for f in files]
            label_paths = os.path.join(data_root, "sequences", seq, "labels")
            self.label_names += [os.path.join(root, f) for root, dirs, files in os.walk(os.path.expanduser(label_paths)) for f in files]

        # populate the pointclouds
        self.scan_names.sort()
        self.label_names.sort()

        # check that there are same amount of labels and scans
        assert(len(self.label_names) == len(self.scan_names))
        # remapping label
        remapping_map = CFG["dynamic_learning_map"]
        # make lookup table for mapping
        nr_classes = max(remapping_map.keys()) + 1
        # +100 hack making lut bigger just in case there are unknown labels
        self.remap_lut = np.zeros((nr_classes + 100), dtype=np.int32)
        self.remap_lut[list(remapping_map.keys())] = list(remapping_map.values())

        # create a scan
        color_dict = CFG["color_map"]
        self.scan = SemLaserScan(nr_classes, color_dict, project=True)

    
    def __getitem__(self, idx):
        self.scan.open_scan(self.scan_names[idx])   # N*3
        self.scan.open_label(self.label_names[idx])   # N
        self.points = self.scan.points
        self.labels = self.remap_lut[self.scan.sem_label]
        # n_car, n_bicycle, n_person = 0, 0, 0
        # for i in self.labels:
        #     if i == 1:
        #         n_car+=1
        #     if i == 2:
        #         n_bicycle+=1
        #     if i == 3:
        #         n_person+=1
        # print('all:', len(self.labels),'n_car:', n_car,'n_bicycle:',n_bicycle, 'n_person:', n_person)
        N_points = self.points.shape[0]

        self.points[:, 0:3] = pc_normalize(self.points[:, 0:3])

        choice = np.random.choice(N_points, self.num_point, replace=True)
        current_points = self.points[choice, :]
        current_labels = self.labels[choice]
        return current_points.astype(float), current_labels.astype(float)

    def __len__(self):
        return len(self.scan_names)


if __name__ == '__main__':
    data_root = '/home/robot/Repository/data_odometry_velodyne/dataset'
    sequence=['00','01']
    config='config/semantic-kitti.yaml'
    num_point=4096

    point_data = KittiDataset(data_root=data_root, num_point=num_point,sequence=sequence,config=config)
    #print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    for i in range(200,300):
        print('point label 0 shape:', point_data.__getitem__(i)[1].shape)
    #print('point data size:', point_data.__len__())

    # import torch, time, random
    # manual_seed = 123
    # random.seed(manual_seed)
    # np.random.seed(manual_seed)
    # torch.manual_seed(manual_seed)
    # torch.cuda.manual_seed_all(manual_seed)
    # def worker_init_fn(worker_id):
    #     random.seed(manual_seed + worker_id)
    # train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    # for idx in range(4):
    #     end = time.time()
    #     for i, (input, target) in enumerate(train_loader):
    #         print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
    #         end = time.time()
    