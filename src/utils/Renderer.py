import torch
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        # 下面几个参数是nerf经常用的，可以从一些nerf的实现代码里面找到
        # 比如我注释的nerf-pytorch: https://github.com/Immortalqx/nerf-pytorch-noted
        self.lindisp = cfg['rendering']['lindisp']  # 根据视差还是深度进行线性采样，True为采用视差
        self.perturb = cfg['rendering']['perturb']  # 如果非零，则在分层随机时间点对每条射线进行采样
        self.N_samples = cfg['rendering']['N_samples']  # 沿每条射线采样的不同次数
        self.N_surface = cfg['rendering']['N_surface']  # 这个和N_importance类似，不过是直接在物体表面附近采样的附加次数
        self.N_importance = cfg['rendering']['N_importance']  # 沿每条射线采样的附加次数

        self.scale = cfg['scale']  # 未使用
        self.occupancy = cfg['occupancy']  # occupancy or volume density
        self.nice = slam.nice  # NICE-SLAM or iMAP
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)  # 把一个tensor拆分成多个tensor
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z  # x,y,z任意一个不在范围内就记False

            pi = pi.unsqueeze(0)  # 扩充了第0维

            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)  # 经过decoder得到occupancy或color
            else:
                ret = decoders(pi, c_grid=None)

            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100  # 对于超出边界的部分，这里将occupancy赋值成100 | TODO 有什么用呢？用100表示需要被忽略掉？
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        # TODO 为什么coarse不需要gt_depth，并且将N_surface设置为0？
        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)  # 展开成向量
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples * 0.01
            # TODO 这里为什么这么做？这里相当于每个点的near都不一样？

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o) / det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
            # TODO 这里得到的是什么？？？每个点用统一的far，并且还要加上一个0.01，是为了防止far=0的情况吗？

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0, torch.max(gt_depth * 1.2))  # 限制far的范围在[0,torch.max(gt_depth * 1.2)]内
        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = (0.95 * gt_depth_surface * (1. - t_vals_surface) +
                                  1.05 * gt_depth_surface * (t_vals_surface))
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.

                # 先对有深度的区域进行处理，得到这部分的采样间隔
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = (0.95 * gt_depth_surface * (1. - t_vals_surface) +
                                                  1.05 * gt_depth_surface * (t_vals_surface))
                # TODO 上面这个公式需要再看一下，对照一下NeRF！
                z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero
                # 在对没有深度的区域进行处理，得到这部分的采样间隔
                # 由于没有深度，就指定near和far了
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = (near_surface * (1. - t_vals_surface) +
                                             far_surface * (t_vals_surface))
                # TODO 上面这个公式需要再看一下，对照一下NeRF！
                z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        if not self.lindisp:  # 根据深度采样
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:  # 根据视差采样
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        if self.perturb > 0.:  # 是否在每个间隔内随机采样
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples + N_surface, -1)

        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)

        # NICE-SLAM默认N_importance=0，毕竟论文里说自己不会像NeRF或者iMAP那样查询两次网络，而是一步到位。
        # 如果查询两次的话，上面的权重会被使用到，其他的东西似乎就抛弃了。
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples + N_importance + N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color

        return depth, uncertainty, color

    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i + ray_batch_size]
                    ret = self.render_batch_ray(
                        c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth * 0.85
        z_vals = near * (1. - t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1]
        return sigma
