from data3d.datasets.nuscenes.videobev_advanced import NuscBevDetData
import random
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import cv2
import torch
import copy
from datetime import datetime
from .data_utils.nuscmap_extractor import NuscMapExtractor
from .data_utils.vectorize import VectorizeMap
from .data_utils.render import Renderer
import logging
logging.captureWarnings(True)
logging.getLogger('shapely.geos').setLevel(logging.ERROR)


def get_rot(h):
    return np.array(
        [
            [np.cos(h), -np.sin(h), 0],
            [np.sin(h), np.cos(h), 0],
            [0, 0, 1],
        ]
    )

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (3840, 2160)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def draw_rect(img, selected_corners, color, linewidth):
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(img,
                    (int(prev[0]), int(prev[1])),
                    (int(corner[0]), int(corner[1])),
                    color, linewidth)
        prev = corner

class ControlNuscenes(NuscBevDetData):
    prompt_list = [
        "A driving scene image at {}, constructed from six surrounding viewpoint images, when the time is {}. {}",
    ]
    view_colors = {
            'CAM_FRONT':[0, 130, 180],
            'CAM_FRONT_RIGHT':[220, 20, 60],
            'CAM_BACK_RIGHT':[255, 0, 0],
            'CAM_BACK':[0, 0, 142],
            'CAM_BACK_LEFT':[0, 60, 100],
            'CAM_FRONT_LEFT': [119, 11, 32]
    }
        
    colors = np.array([
                [255, 255, 255],
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [0, 165, 255],
                [250, 170, 30],
                [144, 238, 144],
                [42, 42, 165],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32]])
    coords_dim = 2 # polylines coordinates dimension, 2 or 3
    # bev configs
    roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
    canvas_size = (200, 100) # bev feature size
    cat2id_map = {
        'ped_crossing': 0,
        'divider': 1,
        'boundary': 2,
    }
    num_class_map = max(list(cat2id_map.values())) + 1
    map_path = "/data/proj/gengyu/datasets/"
    sample_num = 20

    def __init__(self, resemble_size=(2,3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resemble_size = resemble_size
        r, c = self.resemble_size
        h, w = self.ida_aug_conf["final_dim"]
        self.image_size = (r*h, c*w)
        self.num_frames = self.num_sweeps
        self.map_extractor = NuscMapExtractor(self.map_path, self.roi_size)
        self.vectorizer = VectorizeMap(
                coords_dim=self.coords_dim,
                roi_size=self.roi_size,
                sample_num=self.sample_num,
                normalize=True,)
        # self.polygonizer = PolygonizeLocalMapBbox(
        #         canvas_size=self.canvas_size,
        #         coord_dim=2,
        #         num_class=self.num_class_map,
        #         threshold=4/200,
        # )
        self.renderer = Renderer(self.cat2id_map, self.roi_size, 'nusc')


    def generate_prompts(self, metas):
        time_str = datetime.strptime(metas["timestr"], "%Y-%m-%d-%H-%M-%S%z")
        formatted_time = time_str.strftime('%B %d, %Y, at %I:%M:%S %p')
        prompt = random.sample(self.prompt_list, 1)[0]
        prompt = prompt.format(metas["location"], formatted_time, metas["description"])
        return prompt

    def draw_bboxes(self, target, bboxes, labels, depths, colors, thickness=12):
        
        img = np.ones((target.shape[0], target.shape[1], len(self.class_names))) * 255 
        img = img.copy().astype(np.uint8)
        if labels is None or len(labels) == 0:
            return img

        for i, name in enumerate(self.class_names):
            mask = (labels == i)
            lab = labels[mask]
            dep = depths[mask]
            if bboxes is not None: bbox = bboxes[mask] 
            if bboxes is None or len(bbox) == 0:
                continue
            dep = dep * 3
            for j in range(len(bbox)):
                xmin,ymin,xmax,ymax = bbox[j]
                img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] = np.where(img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] > dep[j], dep[j], img[int(ymin) : int(ymax), int(xmin) : int(xmax), i])

        return img 
    
    def draw_corners(self, target, corners, labels, depths2d, colors, linewidth=4):
        
        img = np.ones((target.shape[0], target.shape[1], 3)) * 255 
        img = img.copy().astype(np.uint8)

        if corners is None or len(corners) == 0:
            return img
        
        # print(corners.shape, labels.shape, depths2d.shape)
        sort_indexes = np.argsort(depths2d)[::-1]
        corners = corners[sort_indexes]
        labels = labels[sort_indexes]
        depths2d = depths2d[sort_indexes]
        for j in range(len(corners)):
            color = colors[labels[j] + 1]
            color = (int(color[0]), int(color[1]), int(color[2]))

            # points = corners[j, [0, 1, 2, 3]]
            # points =  np.array([[int(corners[j, 0, 0]), int(corners[j, 0, 1])], [int(corners[j, 1, 0]), int(corners[j, 1, 1])], [int(corners[j, 2, 0]), int(corners[j, 2, 1])], [int(corners[j, 3, 0]), int(corners[j, 3, 1])]])
            points =  np.array([[int(corners[j, 1, 0]), int(corners[j, 1, 1])], [int(corners[j, 2, 0]), int(corners[j, 2, 1])], [int(corners[j, 6, 0]), int(corners[j, 6, 1])], [int(corners[j, 5, 0]), int(corners[j, 5, 1])]])
            points = points.reshape(-1, 1, 2)
            # points[..., 0] = np.clip(points[..., 0], 0, target.shape[1]) 
            # points[..., 1] = np.clip(points[..., 1], 0, target.shape[0]) 
            points = points.astype(np.int32)
            ori_color = (int(color[0]*0.5 + 255*0.5), int(color[1]*0.5 + 255*0.5), int(color[2]*0.5 + 255*0.5))

            cv2.fillPoly(img, [points], ori_color)

            for i in range(4):
                cv2.line(img,
                    (int(corners[j][i][0]), int(corners[j][i][1])),
                    (int(corners[j][i + 4][0]), int(corners[j][i + 4][1])),
                    color[::-1], linewidth)

            draw_rect(img, corners[j][:4], color[::-1], linewidth)
            draw_rect(img, corners[j][4:], color[::-1], linewidth)
        
        return img 

    def render_views(self, shapes, camera_views):
        img_list = []
        for i, view in enumerate(camera_views):
            img = np.zeros((shapes[0], shapes[1], 3))
            img = img.copy().astype(np.uint8)
            color = np.array(self.view_colors[view])
            img = img + color[None, None, :]
            img_list.append(img)
        return img_list
    
    def render_map(self, imgs, ego2imgs, img_metas):
        '''Visualize ground-truth.

        Args:
            idx (int): index of sample.
            out_dir (str): output directory.
        '''
        roi_size = self.roi_size
        location = img_metas["location"]
        map_geoms = self.map_extractor.get_map_geom(location, img_metas['ego2global_translation'], 
                img_metas['ego2global_rotation'])
        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id_map.keys():
                map_label2geom[self.cat2id_map[k]] = v
        img_metas["map_geoms"] = map_label2geom
        self.vectorizer(img_metas)   # add vectors key

        vectors = img_metas['vectors']
        # print("vectors", np.array(vectors[1]).shape)
        roi_size = np.array(roi_size)
        origin = -np.array([roi_size[0]/2, roi_size[1]/2])

        # for k, vector_list in vectors.items():
        #     vectors[k] = vector_list[0]

        for k, vector_list in vectors.items():
            for i, v in enumerate(vector_list):
                v[:, :2] = v[:, :2] * (roi_size + 2) + origin
                vector_list[i] = v
        img_list = self.renderer.render_camera_views_from_vectors(vectors, imgs, 
            ego2imgs, 4, None)
        return img_list, vectors


    def render_directions(self, shapes, img2egos):

        eps = 1e-5
        N = len(img2egos)
        H, W, _ = shapes
        coords_h = np.arange(H)
        coords_w = np.arange(W)
        # coords_d = np.array([1.0])
        coords_d = np.array([1.0, 2.0])

        D = coords_d.shape[0]
        coords = np.stack(np.meshgrid(coords_w, coords_h, coords_d)).transpose((1, 2, 3, 0)) # W, H, D, 3
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * np.maximum(coords[..., 2:3], np.ones_like(coords[..., 2:3])*eps)
        coords = coords.reshape(1, W, H, D, 4, 1)
        img2egos = img2egos.reshape(N, 1, 1, 1, 4, 4)
        # coords3d = np.matmul(img2lidar, coords).squeeze(-1).squeeze(-2)[..., :3]
        # coords3d = coords3d.transpose((0, 2, 1, 3))
        coords3d = np.matmul(img2egos, coords).squeeze(-1)[..., :3]
        coords3d = coords3d.transpose((0, 2, 1, 3, 4))

        directions = coords3d[:, :, :, 1, :] - coords3d[:, :, :, 0, :]
        coords3d = (directions - directions.min()) / (directions.max() - directions.min()) * 255
        coords3d = coords3d.copy().astype(np.uint8)
        coords3d = [cord3d for cord3d in coords3d]

        # directions = coords3d[:, :, :, 1, :] - coords3d[:, :, :, 0, :]
        # print(directions.min(), directions.max())
        # coords3d = sigmoid(directions) * 255
        # print(coords3d.min(), coords3d.max())
        # coords3d = coords3d.copy().astype(np.uint8)
        # coords3d = [cord3d for cord3d in coords3d]

        # coords3d = [direction for direction in directions]

        return coords3d
    
    def get_corners(self, gt_box) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        x, y, z, l, w, h, yaw = gt_box[:7]

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        rot = get_rot(-yaw)
        corners = np.dot(rot, corners)

        # Translate
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners.T
    
    
    def _get_2d_annos(self, shape, source_bbox_3d, source_label_3d, lidar2img):
        gt_bboxes_3d = source_bbox_3d
        gt_label_3d = source_label_3d
        corners_3d = np.stack([self.get_corners(gt_box) for gt_box in gt_bboxes_3d], 0)   # (n, 8, 3)
        # corners_3d = bbox3d_to_8corners(gt_bboxes_3d)
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)

        gt_bbox2d = []
        gt_depth2d = []
        gt_label2d = []
        gt_corners3d = []
        for i in range(len(self.img_key_list)):
            lidar2img_rt = np.array(lidar2img[i])
            pts_2d = pts_4d @ lidar2img_rt.T
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=None)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            
            H, W = shape[0], shape[1]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
            imgfov_pts_depth = pts_2d[..., 2].reshape(num_bbox, 8)
            mask = imgfov_pts_depth.mean(1) > 0.1

            if mask.sum() == 0:
                gt_bbox2d.append([])
                gt_depth2d.append([])
                gt_label2d.append([]) 
                gt_corners3d.append([])
                continue

            imgfov_pts_2d = imgfov_pts_2d[mask]
            imgfov_pts_depth = imgfov_pts_depth[mask]
            imgfov_pts_label= gt_label_3d[mask]

            bbox = []
            label = []
            depth = []
            corners3d = []
            for j, corner_coord in enumerate(imgfov_pts_2d):
                final_coords = post_process_coords(corner_coord, imsize = (W,H))
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                    if ((max_x - min_x) >W-100) and ((max_y - min_y)>H-100):
                        continue
                    bbox.append([min_x, min_y, max_x, max_y])
                    label.append(imgfov_pts_label[j])
                    depth.append(imgfov_pts_depth[j].mean())
                    corners3d.append(copy.deepcopy(corner_coord))
            gt_bbox2d.append(np.array(bbox))
            gt_depth2d.append(np.array(depth))
            gt_label2d.append(np.array(label)) 
            gt_corners3d.append(np.array(corners3d)) 
        bbox2d_info = {
            'gt_bbox2d' : gt_bbox2d,
            'gt_depth2d' : gt_depth2d,
            'gt_label2d' : gt_label2d,
            'gt_corners3d': gt_corners3d
        }

        return bbox2d_info
    
    def _prepare_one_frame(self, item):
        source_img = item['img'].numpy() #torch.Size([6, 3, 256, 704]) 
        source_label_3d = item['gt_labels_3d'].numpy() # torch.Size([32])  
        source_bbox_3d = item['gt_bboxes_3d'].numpy() # torch.Size([32, 9])
        # source_corner = item["corners"].numpy()
        cam2lidar = item["cam2lidar"].numpy() #torch.Size([6, 4, 4]) 
        bda_mat = item["bda_mat"].numpy() #torch.Size([4, 4])
        ida_mats = item["ida_mats"].numpy() #torch.Size([6, 4, 4]) 
        intrin_mats = item["intrin_mats"].numpy() #torch.Size([6, 4, 4]) 

        lidar2imgs = ida_mats @ intrin_mats @ np.linalg.inv(cam2lidar) @ np.linalg.inv(bda_mat)[None]
        img2lidars = np.linalg.inv(lidar2imgs)

        w, x, y, z = item["img_metas"]["lidar2ego_rotation"]
        lidar2ego_rot = Quaternion(w, x, y, z).rotation_matrix
        lidar2ego_tran = item["img_metas"]["lidar2ego_translation"]
        lidar2ego = np.zeros((4, 4), dtype=np.float32)
        lidar2ego[3, 3] = 1
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, -1] = lidar2ego_tran
        ego2imgs = lidar2imgs @ np.linalg.inv(lidar2ego)

        # import ipdb;ipdb.set_trace()
        # intrin = ida_mats @ intrin_mats
        # extrin = np.linalg.inv(cam2ego) @ lidar2ego[None] @ np.linalg.inv(bda_mat)[None]

        # img = draw_boxes_on_img(source_img[0].transpose(1,2,0), item["gt_bboxes_3d"], extrin[0], intrin[0])
        if len(source_bbox_3d) == 0:
            bbox2d_info = {
                'gt_bbox2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_depth2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_label2d' : [[] for _ in range(len(self.img_key_list))],
                'gt_corners3d': [[] for _ in range(len(self.img_key_list))],
            }
        else:
            bbox2d_info = self._get_2d_annos((source_img.shape[2], source_img.shape[3]), source_bbox_3d, source_label_3d, lidar2imgs)
        
        source_label_2d = bbox2d_info['gt_label2d']
        source_bbox_2d = bbox2d_info['gt_bbox2d']
        source_depth_2d = bbox2d_info['gt_depth2d']
        source_corner_2d = bbox2d_info['gt_corners3d']

        # if self.shift_view and self.split == "train":
        #     if self.random_shift:
        #         random.shuffle(camera_views)
        #     else:
        #         camera_views = list_move_right(camera_views, random.choice(range(len(camera_views)))) 
        img_list = []
        source_list = []
        for view_id, view in enumerate(self.img_key_list):
            img = source_img[view_id].transpose((1,2,0))
            
            bboxes2d = source_bbox_2d[view_id]
            labels2d = source_label_2d[view_id]
            depths2d = source_depth_2d[view_id]
            corners2d = source_corner_2d[view_id]
            source = self.draw_bboxes(img, bboxes2d, labels2d, depths2d, self.colors)
            source_corner = self.draw_corners(img, corners2d, labels2d, depths2d, self.colors, linewidth=2) ###for 512
            # source_corner = self.draw_corners(img, corners2d, labels2d, self.colors, linewidth=4) ###for 800
            source = np.concatenate([source_corner, source], -1)
            img_list.append(img)
            source_list.append(source)

        map_list, vectors = self.render_map(img_list, ego2imgs, item["img_metas"])
        render_list = self.render_directions(img.shape, img2lidars)

        target = np.stack(img_list, 0)
        source = np.stack(source_list, 0)
        render_map = np.stack(map_list, 0)
        render_pe = np.stack(render_list, 0)

        # source = np.concatenate([source, render_pe], -1)
        source = np.concatenate([source, render_map, render_pe], -1) 


        # source = np.concatenate([source, np.zeros([source.shape[0], source.shape[1], 13], dtype=np.float32)], -1)
        # filenames = item['img_metas']._data[0]['filename']
        return target, source

    def resemble_data(self, sweep_data: torch.Tensor, img_format):
        img_r, img_c, img_h, img_w = img_format
        data = sweep_data.reshape(self.num_frames, img_r, img_c, img_h, img_w, -1)
        data = data.permute(0, 1, 3, 2, 4, 5)
        data = data.reshape(self.num_frames, img_r*img_h, img_c*img_w, -1)
        return data


    def __getitem__(self, idx):
        (sweep_imgs,
        sweep_intrins,
        sweep_ida_mats,
        sweep_sensor2lidar_mats,
        bda_mat,
        img_metas,
        gt_boxes,
        gt_labels,
        ) = super().__getitem__(idx)
        key_frame_idx = 0
        sweep_targets = list()
        sweep_sources = list()
        for frame_idx in range(self.num_sweeps):
            item = {
                "img": sweep_imgs[frame_idx],
                "gt_labels_3d": gt_labels[frame_idx],
                "gt_bboxes_3d": gt_boxes[frame_idx],
                "cam2lidar": sweep_sensor2lidar_mats[frame_idx],
                "bda_mat": bda_mat,
                "ida_mats": sweep_ida_mats[frame_idx],
                "intrin_mats": sweep_intrins[frame_idx],
                "img_metas": img_metas
                }
            target, source = self._prepare_one_frame(item)
            # cv2.imwrite(f"tmp/t_{frame_idx}.jpg", target[:,:,::-1])
            # cv2.imwrite(f"tmp/s_{frame_idx}.jpg", source[:, :, :3][:,:,::-1])
            sweep_targets.append(torch.from_numpy(target))
            sweep_sources.append(torch.from_numpy(source))
        sweep_targets = torch.stack(sweep_targets, 0)      # T, S, H, W, 3
        sweep_sources = torch.stack(sweep_sources, 0)

        img_r, img_c = self.resemble_size
        img_h, img_w = self.ida_aug_conf["final_dim"]
        img_format = (img_r, img_c, img_h, img_w)
        sweep_targets = self.resemble_data(sweep_targets, img_format)
        sweep_sources = self.resemble_data(sweep_sources, img_format)
        # Normalize source images to [-1, 1].
        sweep_targets = (sweep_targets.to(torch.float32) / 127.5) - 1.0
        # 时间顺序
        sweep_targets = sweep_targets.permute(0, 3, 1, 2).flip(0) # n_frames, 3, H, W

        sweep_sources = (sweep_sources.to(torch.float32) / 127.5) - 1.0
        sweep_sources = sweep_sources.permute(3, 0, 1, 2).flip(1) # 19, n_frames, H, W
        prompt = self.generate_prompts(img_metas)

        ret = dict(
            mp4=sweep_targets,
            txt=prompt, 
            hint=sweep_sources,
            fps=12,
            height=sweep_targets.shape[2],
            width=sweep_targets.shape[3],
            num_frames=self.num_frames
                   )
        return ret
    
    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(**kwargs)