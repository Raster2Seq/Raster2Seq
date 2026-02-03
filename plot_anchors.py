import json
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from tqdm import tqdm

ckpt_path = "output/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/checkpoint0449.pth"
ckpt = torch.load(ckpt_path)
image_root = "data/coco_s3d_bw/test/"  
pred_root = "s3d_test_preds/s3d_bw_poly2seq_l512_sem_bs32_coo20_cls1_anchor_deccatsrc_smoothing_numcls19_pts_fromnorder1399_convertv3_t2/jsons/"
image_paths = glob.glob("data/coco_s3d_bw/test/*.png")
output_root = "s3d_anchor_viz"
os.makedirs(output_root, exist_ok=True)

IMG_SIZE = 256
# anchors = torch.sigmoid(ckpt['ema']['query_embed.weight']) * IMG_SIZE

# List of 2D points (example)
# anchors = anchors.detach().cpu().numpy()
# np.savetxt("anchors.csv", anchors, delimiter=",", header="x,y", comments="")

# Extract x and y coordinates
# x_coords = [point[0] for point in anchors]
# y_coords = [point[1] for point in anchors]

# # Create a scatter plot
# fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='markers'))

# # Add labels and title
# fig.update_layout(
#     title="2D Points Plot",
#     xaxis_title="X-axis",
#     yaxis_title="Y-axis",
#     showlegend=False,
#     template="plotly_white"

# )

# # Show the plot
# # fig.show()

# # Save the plot to a file
# fig.write_image("anchors_s3d_plot.png")  # Save as PNG

for i, image_path in tqdm(enumerate(image_paths)):
    image_id = os.path.basename(image_path).split('.')[0]
    pred_path = f"{pred_root}/{image_id}_pred.json"

    with open(pred_path, 'r') as f:
        data = json.load(f) # ['annotations']
        room_polys = [np.array(x['segmentation']) for x in data]
        room_ids = [x['category_id'] for x in data]
        room_anchors = [np.array(x['anchors']) for x in data]

    # Load image and convert to numpy array
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # # Create Plotly figure with image background
    # fig = go.Figure()
    # fig.add_trace(go.Image(z=img_np))
    # fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(color='red', size=8), name='Anchors'))

    # fig.update_layout(
    #     title="2D Anchors on Image",
    #     xaxis=dict(visible=False),
    #     yaxis=dict(visible=False),
    #     showlegend=False,
    #     margin=dict(l=0, r=0, t=40, b=0)
    # )

    # fig.write_image("anchors_s3d_on_image_plot.png")  # Save as PNG

    # Overlay polygons on the image using Plotly
    fig = go.Figure()
    fig.add_trace(go.Image(z=img_np))
    # fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(color='red', size=8), name='Anchors'))

    # Add polygons and plot each corner as yellow star
    for poly, anchor in zip(room_polys, room_anchors):
        poly = np.array(poly)
        anchor = np.array(anchor)
        # Close the polygon by repeating the first point
        x_poly = np.append(poly[:,0], poly[0,0])
        y_poly = np.append(poly[:,1], poly[0,1])
        fig.add_trace(go.Scatter(x=x_poly, y=y_poly, mode='lines', line=dict(color='blue', width=2), name='Room'))
        # Plot corners as yellow stars
        fig.add_trace(go.Scatter(x=poly[:,0], y=poly[:,1], mode='markers', marker=dict(color='red', size=12), name='Corners'))
        # Plot anchors
        fig.add_trace(go.Scatter(x=anchor[:,0], y=anchor[:,1], mode='markers', marker=dict(symbol='star',color='yellow', size=12), name='Anchors'))

    fig.update_layout(
        title="2D Anchors and Room Polygons on Image",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_image(f"{output_root}/viz_{image_id}.png")  # Save as PNG