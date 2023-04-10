import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg
import numpy as np

class Plane_propagated_depth_guider(nn.Module):
    def __init__(self, input_size,
                 delta_cos=0.867,
                 delta_z=0.0001, sample_ratio=0.3):
        super(Plane_propagated_depth_guider, self).__init__()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda() # x, y focal center
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
    
    def init_image_coor(self): # take care of point cloud
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0
    
    def transfer_xyz(self, depth, k_maritix): # take care of point cloud
        fx = k_maritix[0,0]
        fy = k_maritix[1,1]
        x = self.u_u0 * torch.abs(depth) / fx
        y = self.v_v0 * torch.abs(depth) / fy
        z = depth
        pw = torch.cat([x, y, z], 0).permute(1, 2, 0)
        return pw

    def select_index(self, num):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        if not num <= valid_width * valid_height:
            raise AssertionError()
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)
        p123 = {'p1_x': p1, 'p2_x': p2, 'p3_x': p3}
        return p123
    
        
    def estimate_plane_equation(self, points):
        """_summary_
        Args:
            points (_type_): num_sample x 3
        Returns:
            _type_: _description_
        """
        # print(points.shape)
        points = points.detach().cpu().numpy()
        num_sample = points.shape[0]
        A = np.matrix(np.column_stack((points[:, 0], points[:, 1], np.ones((num_sample, 1)))))
        b = np.matrix(points[:, 2].reshape(num_sample, 1))
        fit = (A.T * A).I * A.T * b
        a = torch.as_tensor(fit[0])[0][0]
        b = torch.as_tensor(fit[1])[0][0]
        d = torch.as_tensor(fit[2])[0][0]
        # z = fit_0 * x + fit_1 * y + fit_2
        return a, b, d
    
    def random_select_points(self, gt_mask):
        # select points for plane estimation
        random_mask = torch.rand(480, 640) > 0.95
        return gt_mask * random_mask
    
    def select_points(self, gt_mask, random_rate=0.1, num_neighbors=10):
        gt_mask = gt_mask.detach().cpu().numpy()
        x, y = np.where(gt_mask == True)
        num_positives = x.shape[0]
        query_indexs = np.random.choice(num_positives, int(num_positives * random_rate), replace=True)
        # propagated_indexs = np.random.choice(num_positives, int(num_positives * random_rate), replace=True)
        propagated_indexs = np.random.choice(num_positives, num_neighbors, replace=False)
        x_queries = x[query_indexs]
        y_queries = y[query_indexs]
        x_propagated = x[propagated_indexs]
        y_propagated = y[propagated_indexs]
        return x_queries, y_queries, x_propagated, y_propagated
    
    def forward(self, pred_depth, gt_masks, gt_depth, k_maritix):
        depth_gt_valid_mask = gt_depth[0] > cfg.dataset.min_depth
        
        C, H, W = pred_depth.shape
        pred_pointcloud = self.transfer_xyz(pred_depth, k_maritix) #480*640*3
        
        pred_depth = pred_depth[0]
        N = gt_masks.shape[0]
        output = pred_depth.clone()

        for i in range(0, N):
            # estimate the equation
            candidate = self.random_select_points(gt_masks[i].clone()) * depth_gt_valid_mask
            # self.save_mask(gt_masks[i].clone(), i)
            candidate_points = pred_pointcloud[candidate]
            # points_pred = gt_3d_points_pred[candidate]
            try:
                a, b, d = self.estimate_plane_equation(candidate_points)
                # self.visualise(plane_equation, points, points_pred, i)
            except:
                continue
            
            # original index 
            x_queries, y_queries, x_propagated, y_propagated = self.select_points(gt_masks[i])
            
            # indexes after calibration 
            query_x = pred_pointcloud[x_queries, y_queries, 0].detach()
            query_y = pred_pointcloud[x_queries, y_queries, 1].detach()
            propagated_x = pred_pointcloud[x_propagated, y_propagated, 0].detach()
            propagated_y = pred_pointcloud[x_propagated, y_propagated, 1].detach()
            output[x_queries, y_queries] = torch.mean(pred_depth[x_propagated, y_propagated].reshape(1, -1) * ((a * query_x + b * query_y + d).reshape(-1, 1)/ (a * propagated_x + b * propagated_y + d)), 1)
            # pred_depth[x_propagated, y_propagated].reshape(1, -1) * 
            # output[0, x_queries, y_queries] = torch.mean(((a * query_x + b * query_y + d).reshape(-1, 1)/ (a * propagated_x + b * propagated_y + d)), 1)
            
            #
        # check_2 = output[x_queries, y_queries]
        return torch.unsqueeze(output, 0)