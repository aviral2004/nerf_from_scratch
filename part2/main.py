import torch
from torch.utils.data import Dataset
import numpy as np

def transform(c2w, x_c):
    """
        Inputs:
            c2w: Nx4x4 matrix containing the inverse of the extrinsic matrix
            x_c: Nx3 tensor containing the coordinates to be transformed from camera coordinates to world coordinates

        Outputs:
            x_w: transformed coordinates in world space, Nx4x1
    """
    x_c = torch.cat([x_c, torch.ones((x_c.shape[0], 1))], dim = 1).unsqueeze(-1) # N x 4 x 1
    x_w = torch.bmm(c2w, x_c) # Nx4x4 * Nx4x1 = Nx4x1
    return x_w

def pixel_to_camera(K, uv, s):
    """
        Inputs:
            K: 3 x 3 tensor containing the intrinsic matrix
            uv: N x 2 tensor containing the pixel coordinates
            s: scalar representing the depth of the point in camera coordinates

        Outputs:
            x_c: N x 3 tensor containing the coordinates in camera space
    """
    o_x = K[0, 2] #1
    o_y = K[1, 2] #1

    f_x, f_y = K[0, 0], K[1, 1] # N

    u = uv[:, 0] # N
    v = uv[:, 1] # N
    x_c = s * (u - o_x) / f_x # N
    y_c = s * (v - o_y) / f_y # N

    return torch.stack([x_c, y_c, s * torch.ones_like(x_c)], dim=1) # N x 3

def pixel_to_ray(K, c2w, uv):
    """
        Inputs:
            K: 3 x 3 tensor containing the intrinsic matrix
            c2w: N x 4 x 4 tensor containing the inverse of the extrinsic matrix
            uv: N x 2 tensor containing the pixel coordinates

        Outputs:
            r_o: N x 2 tensor containing the origin of the ray in world space
            r_d: N x 3 tensor containing the direction of the ray in world space
    """
    w2c = torch.inverse(c2w) # N x 4 x 4
    R = w2c[:, :3, :3] # N x 3 x 3
    t = w2c[:, :3, 3] # N x 3

    r_o = torch.bmm(-torch.inverse(R), t.unsqueeze(-1)) # Nx3x3 * Nx3x1 = Nx3x1
    r_o = r_o.squeeze(-1) # Nx3
    # make homogenous
    # r_o = torch.cat([r_o, torch.ones((r_o.shape[0], 1))], dim = 1) # N x 4

    X_w = pixel_to_camera(K, uv, 1) # Nx3
    # X_w = torch.cat([X_w, torch.ones((X_w.shape[0], 1))], dim = 1).unsqueeze(-1) # N x 4
    X_w = transform(c2w, X_w).squeeze(-1)[:, :3] # N x 3
    
    r_d = (X_w - r_o) # N x 3
    r_d = r_d / torch.norm(r_d, dim=1, keepdim=True) # N x 3

    return r_o, r_d


class RayDataset(Dataset):
    def __init__(self, images, c2w, focal):
        self.ims = images
        # self.im_height = np.array([im.shape[0] for im in self.ims])
        # self.im_width = np.array([im.shape[1] for im in self.ims])
        # self.sizes = self.im_height * self.im_width

        self.im_height, self.im_width = self.ims.shape[1:3]
        self.size = self.im_height * self.im_width

        # self.sizes_cumsum = np.cumsum(self.sizes)
        # self.sizes_cumsum = np.insert(self.sizes_cumsum, 0, 0)

        self.K = self.get_intrinsic(focal)
        self.c2w = torch.from_numpy(c2w)

    def get_intrinsic(self, focal):
        """
            Inputs:
                focal: scalar representing the focal length

            Outputs:
                K: N x 3 x 3 tensor containing the intrinsic matrix
        """
        # get a tensor of K
        K = np.array([[focal, 0, self.im_width / 2], [0, focal, self.im_height / 2], [0, 0, 1]], dtype=np.float64)
        K = torch.tensor(K, dtype=torch.float64)
        return K

    def __len__(self):
        return self.ims.shape[0] * self.size
    
    def sample_rays(self, n_samples, bounds=None, indices=None):
        """
            Inputs:
                n_samples: scalar representing the number of rays to sample

            Outputs:
                rays_o: n_samples x 3 tensor containing the origin of the rays in world space
                rays_d: n_samples x 3 tensor containing the direction of the rays in world space
                pixels: n_samples x 3 tensor containing the color of the pixels
        """
        # extra code for testing
        if bounds is not None:
            idx = np.random.randint(bounds[0], bounds[1], n_samples)
        else:
            if indices is None:
                # get n_samples random indices from flattened image coords
                indices = np.random.randint(0, len(self), n_samples)
            else:
                n_samples = len(indices)
                idx = indices

        # find which image the idx belongs to
        # im_nums = np.searchsorted(self.sizes_cumsum, idx) - 1
        # y = idx - self.sizes_cumsum[im_nums]
        # x = y % self.im_width[im_nums]
        # y = y // self.im_width[im_nums]
            
        idx = indices
        im_nums = idx // self.size
        y = idx - im_nums * self.size
        x = y % self.im_width
        y = y // self.im_width

        uv = torch.tensor([x + 0.5, y + 0.5], dtype=torch.float64).T

        rays_o, rays_d = pixel_to_ray(self.K, self.c2w[im_nums], uv)
        pixels = self.ims[im_nums, y, x] / 255

        return rays_o, rays_d, pixels
    
    def sample_along_rays(self, rays_o, rays_d, perturb=True, near=2.0, far=6.0, t_width=1):
        """
            Inputs:
                rays_o: n_samples x 3 tensor containing the origin of the rays in world space
                rays_d: n_samples x 3 tensor containing the direction of the rays in world space
                perturb: boolean representing whether to perturb the rays

            Outputs:
                samples: n_samples x 3 tensor containing the samples along the rays
        """
        n_samples = rays_o.shape[0]
        t = torch.linspace(near, far, n_samples)

        if perturb:
            t += torch.rand_like(t) * t_width

        samples = rays_o + rays_d * t.unsqueeze(-1)
        return samples

    # def __getitem__(self, idx):
    #     im_num = idx//len(self.ims)
    #     im = self.ims[im_num]
    #     y = idx//self.im_width[im_num]
    #     x = idx%self.im_width[im_num]
    #     uv = torch.tensor([x, y], dtype=torch.float32)
    #     rays = pixel_to_ray(self.K[im_num].unsqueeze(0), self.c2w[im_num].unsqueeze(0), uv.unsqueeze(0))
    #     pixel = im[y, x]/255
    #     return *rays, pixel

def volrender(sigmas, rgbs, step_size):
    """
        Inputs:
            sigmas: N x i x 1 tensor containing densities along the ray
            rgbs: N x i x 3 tensor containing colors along the ray
            step_size: float value containing delta

        Outputs:
            rendered_colors: N x 3 tensor containing the final expected color for each ray
    """

    alpha = 1 - torch.exp(-sigmas * step_size) # Nxix1

    T = torch.cumprod(1 - alpha, dim=1)
    T = torch.cat([torch.ones((T.shape[0], 1, 1)), T[:, :-1]], dim=1) # Nxix1

    colors = T * alpha * rgbs # Nxix3

    rendered_colors = torch.sum(colors, dim=1) # Nx3
    return rendered_colors