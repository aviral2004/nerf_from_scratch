# Visualize Cameras, Rays and Samples
import viser, time
import numpy as np

from main import *

data = np.load(f"lego_200x200.npz")

# Training images: [100, 200, 200, 3]
images_train = data["images_train"] / 255.0

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = data["c2ws_train"]

# Validation images: 
images_val = data["images_val"] / 255.0

# Cameras for the validation images: [10, 4, 4]
# (camera-to-world transformation matrix): [10, 200, 200, 3]
c2ws_val = data["c2ws_val"]

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float

# --- You Need to Implement These ------
dataset = RayDataset(images_train, c2ws_train, focal)

# This will check that your uvs aren't flipped
uvs_start = 0
uvs_end = 40_000
# sample_uvs = dataset.uvs[uvs_start:uvs_end] # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
# # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
# assert np.all(images_train[0, sample_uvs[:,1], sample_uvs[:,0]] == dataset.pixels[uvs_start:uvs_end])

# # # Uncoment this to display random rays from the first image
indices = torch.randint(low=0, high=40_000, size=(100,))

# # # Uncomment this to display random rays from the top left corner of the image
# indices_x = torch.randint(low=100, high=200, size=(100,))
# indices_y = torch.randint(low=0, high=100, size=(100,))
# indices = indices_x + (indices_y * 200)

# data = {"rays_o": dataset.rays_o[indices], "rays_d": dataset.rays_d[indices]}
rays_o, rays_d, pixels = dataset.sample_rays(100, bounds=(uvs_start, uvs_end), indices=indices) # Should expect (B, 3)
points = dataset.sample_along_rays(rays_o, rays_d, perturb=False, )
H, W = images_train.shape[1:3]
K = dataset.K.numpy()

points = points.numpy()
rays_o = rays_o.numpy()
rays_d = rays_d.numpy()
# ---------------------------------------

server = viser.ViserServer(share=True)
for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
  server.add_camera_frustum(
    f"/cameras/{i}",
    fov=2 * np.arctan2(H / 2, K[0, 0]),
    aspect=W / H,
    scale=0.15,
    wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
    position=c2w[:3, 3],
    image=image
  )
for i, (o, d) in enumerate(zip(rays_o, rays_d)):
  positions = np.stack((o, o + d * 6.0))
  server.add_spline_catmull_rom(
      f"/rays/{i}", positions=positions,
  )
server.add_point_cloud(
    f"/samples",
    colors=np.zeros_like(points).reshape(-1, 3),
    points=points.reshape(-1, 3),
    point_size=0.03,
)
time.sleep(1000)