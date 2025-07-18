import os
import os.path
import torch
import numpy as np
import pandas
import csv
from glob import glob
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame

class Coesot(BaseVideoDataset):
    """ VisEvent dataset.
    """

    def __init__(self, root=None, dtype='rgbrgb', split='train', image_loader=jpeg4py_loader_w_failsafe): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        root = env_settings().visevent_dir if root is None else root
        assert split in ['train'], 'Only support train split in VisEvent, got {}'.format(split)
        super().__init__('Coesot', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._build_sequence_list()


    def _build_sequence_list(self):

        #voxel_path
        # file_path = "data/Disk_C/pengcheng/dataset/COESOT_dataset/train_voxel_list/trainlist.txt"
        file_path = os.path.join(self.root, '{}list.txt'.format(self.split))
        sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        return sequence_list

    def get_name(self):
        return 'coesot'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        if not os.path.exists(bb_anno_file):
            print("no exist")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absent.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in list(csv.reader(f))])

        target_visible = occlusion

        return target_visible

    def _get_img_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_aps")

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 5.0) & (bbox[:, 3] > 5.0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return rgb event image path
        '''
        path = seq_path
        filename1 = path.split("/")[-1] + '_aps'
        filename2 = path.split("/")[-1] + '_dvs'
        path1 = os.path.join(path,filename1)
        path2 = os.path.join(path,filename2)
        # if os.path.exists(os.path.join(path, 'frame{:04}.png'.format(frame_id))):
        #     vis_img_files = sorted(glob(os.path.join(path, '*.png')))
        # else:
        #     vis_img_files = sorted(glob(os.path.join(path, '*.bmp')))
        #
        # try:
        #     vis_path = vis_img_files[frame_id]
        # except:
        #     print(f"seq_path: {seq_path}")
        #     print(f"vis_img_files: {vis_img_files}")
        #     print(f"frame_id: {frame_id}")
        #
        #
        # event_path = vis_path.replace('_aps', '_dvs')
        # if not os.path.exists(event_path):
        #     event_path = event_path.replace('png', 'bmp')

        vis_path = sorted(os.listdir(os.path.join(seq_path, path1)))
        event_path = sorted(os.listdir(os.path.join(seq_path, path2)))
        # return (os.path.join(seq_path, 'visible', vis_frame_names[frame_id]),
        #         os.path.join(seq_path, 'infrared', inf_frame_names[frame_id]))
        vis_path = os.path.join(path1, vis_path[frame_id])
        event_path = os.path.join(path2,event_path[frame_id])
        return vis_path, event_path # frames start irregularly


    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - rgb+event_colormap
        '''

        color_path, event_path = self._get_frame_path(seq_path, frame_id)
        # img = get_x_frame(color_path, event_path, dtype=self.dtype, depth_clip=False)
        # if os.makedirs(color_path):
        #     print("True1")
        # if os.makedirs(event_path):
        #     print("True2")
        # if os.path.exists(color_path):
        #     print("color文件夹存在。\n")
        # else:
        #     print("color文件夹不存在。\n")
        # if os.path.exists(event_path):
        #     print("event文件夹存在。\n")
        # else:
        #     print("e文件夹不存在。\n")
        img = get_x_frame(color_path, event_path, dtype=self.dtype, depth_clip=False)
        return img  # (h,w,6)
        # return img, flag  # (h,w,6)

    # def _get_event_frame(self,seq_path, frame_id):
    #     color_path, event_path = self._get_frame_path(seq_path, frame_id)
    #     return event_path

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        # seq_path = self._get_img_path(seq_id)

        # if anno is None:
        #     anno = self.get_sequence_info(seq_id)
        #
        # anno_frames = {}
        # for key, value in anno.items():
        #     anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]
        #
        # # frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        # # frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]
        #
        # frame_list = []  # 初始化空列表
        # flag_list = []
        # for ii, f_id in enumerate(frame_ids):  # 对 frame_ids 进行迭代，同时提供元素的索引
        #     frame, flag = self._get_frame(seq_path, f_id)  # 为每个 f_id 调用 _get_frame 方法获取“帧”
        #     frame_list.append(frame)  # 将“帧”加入到 frame_list 列表中
        #     flag_list.append(flag)
        #
        # object_meta = OrderedDict({'object_class_name': None,
        #                            'motion_class': None,
        #                            'major_class': None,
        #                            'root_class': None,
        #                            'motion_adverb': None})
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta


#
# class Coesot(BaseVideoDataset):
#     def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
#
#         root = env_settings().got10k_dir if root is None else root
#         super().__init__('Coesot', root, image_loader)
#
#         self.sequence_list = self._get_sequence_list()
#
#         # seq_id is the index of the folder inside the got10k root path
#
#     def get_name(self):
#         return 'coesot'
#
#     def _get_sequence_list(self):
#         with open(os.path.join(self.root,'trainlist.txt')) as f:
#             dir_list = list(csv.reader(f))#827
#         dir_list = [dir_name[0] for dir_name in dir_list]
#         return dir_list
#
#     def _read_bb_anno(self, seq_path):
#         bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
#         gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
#         return torch.tensor(gt)
#
#     def _get_sequence_path(self, seq_id):
#         return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_aps")
#
#     def _get_event_img_sequence_path(self, seq_id):
#         return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_dvs")
#
#     def _get_grountgruth_path(self, seq_id):
#         return os.path.join(self.root, self.sequence_list[seq_id])
#
#     def get_sequence_info(self, seq_id):
#         bbox_path = self._get_grountgruth_path(seq_id)
#         bbox = self._read_bb_anno(bbox_path)
#
#         valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
#         visible = valid.clone().byte()
#         # return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
#         return {'bbox': bbox, 'valid': valid, 'visible': visible, }
#
#     def _get_frame_path(self, seq_path, frame_id):
#         if os.path.exists(os.path.join(seq_path, 'frame{:04}.png'.format(frame_id))):
#             return os.path.join(seq_path, 'frame{:04}.png'.format(frame_id))    # frames start from 0
#         else:
#             return os.path.join(seq_path, 'frame{:04}.bmp'.format(frame_id))    # some image is bmp
#
#     def _get_frame(self, seq_path, frame_id):
#         return self.image_loader(self._get_frame_path(seq_path, frame_id))
#
#     def _get_event_sequence_path(self, seq_id):        ## get evemts' frames
#         return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_voxel")
#
#     def _get_event_frame(self, seq_path, frame_id):
#         frame_event_list = []
#         for f_id in frame_id:
#             event_frame_file = os.path.join(seq_path, 'frame{:04}.mat'.format(f_id))
#             if os.path.getsize(event_frame_file) == 0:
#                 event_features = np.zeros(4096, 19)
#                 # need_data = [np.zeros([4096, 3]), np.zeros([4096, 16])]
#             else:
#                 mat_data = scio.loadmat(event_frame_file)
#                 # need_data = [mat_data['coor'], mat_data['features']]
#                 event_features = np.concatenate((mat_data['coor'], mat_data['features']), axis=1)        # concat coorelate and features (x,y,z, feauture32/16)
#                 if np.isnan(event_features).any():
#                     event_features = np.zeros(4096, 19)
#                     print(event_frame_file, 'exist nan value in voxel.')
#             frame_event_list.append(event_features)
#         return frame_event_list
#
#     def get_frames(self, seq_id, frame_ids, anno=None):
#         seq_path = self._get_sequence_path(seq_id)
#         # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
#
#         frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
#         seq_event_path = self._get_event_img_sequence_path(seq_id)
#         frame_event_img_list = [self._get_frame(seq_event_path, f_id) for f_id in frame_ids]
#         if anno is None:
#             anno = self.get_sequence_info(seq_id)
#
#         anno_frames = {}
#         for key, value in anno.items():
#             anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
#
#         object_meta = OrderedDict({'object_class_name': None,
#                                    'motion_class': None,
#                                    'major_class': None,
#                                    'root_class': None,
#                                    'motion_adverb': None})
#
#         seq_event_path = self._get_event_sequence_path(seq_id)
#         frame_event_list = self._get_event_frame(seq_event_path, frame_ids)
#
#         return frame_list, anno_frames, object_meta, frame_event_list, frame_event_img_list

