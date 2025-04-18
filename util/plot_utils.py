"""
Utilities for floorplan visualization.
"""
import torch
import math
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.patches as mpatches
from PIL import ImageColor

import cv2 
import numpy as np
from imageio import imsave

from shapely.geometry import LineString
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch


colors_12 = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58230",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd7b4"
]

semantics_cmap = {
    0: '#e6194b',
    1: '#3cb44b',
    2: '#ffe119',
    3: '#0082c8',
    4: '#f58230',
    5: '#911eb4',
    6: '#46f0f0',
    7: '#f032e6',
    8: '#d2f53c',
    9: '#fabebe',
    10: '#008080',
    11: '#e6beff',
    12: '#aa6e28',
    13: '#fffac8',
    14: '#800000',
    15: '#aaffc3',
    16: '#808000',
    17: '#ffd7b4'
}

semantics_label = {
    0: 'Living Room',
    1: 'Kitchen',
    2: 'Bedroom',
    3: 'Bathroom',
    4: 'Balcony',
    5: 'Corridor',
    6: 'Dining room',
    7: 'Study',
    8: 'Studio',
    9: 'Store room',
    10: 'Garden',
    11: 'Laundry room',
    12: 'Office',
    13: 'Basement',
    14: 'Garage',
    15: 'Misc.',
    16: 'Door',
    17: 'Window'
}


BLUE = '#6699cc'
GRAY = '#999999'
DARKGRAY = '#333333'
YELLOW = '#ffcc33'
GREEN = '#339933'
RED = '#ff3333'
BLACK = '#000000'


def plot_floorplan_with_regions(regions, corners=None, edges=None, scale=256):
    """Draw floorplan map where different colors indicate different rooms
    """
    colors = colors_12

    regions = [(region * scale / 256).round().astype(np.int32) for region in regions]

    # define the color map
    room_colors = [colors[i % len(colors)] for i in range(len(regions))]

    colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
    colorMap = np.asarray(colorMap)
    if len(regions) > 0:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
            np.uint8)
    else:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
            np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
    alpha_channels[1:len(regions) + 1] = 150

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_map = np.zeros([scale, scale]).astype(np.int32)
    # sort regions
    if len(regions) > 1:
        avg_corner = [region.mean(axis=0) for region in regions]
        ind = np.argsort(np.square(np.array(avg_corner)).sum(axis=1), axis=0)
        regions = [regions[_idx] for _idx in ind] # np.array(regions)[ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = (0,0,0,255)
    lineColor = (0,0,0,255)

    for region in regions:
        for i, point in enumerate(region):
            if i == len(region)-1:
                cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=5)
            else:    
                cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=5)

    for region in regions:
        for i, point in enumerate(region):
            cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
            cv2.circle(image, tuple(point), color=(255, 255, 255, 0), radius=6, thickness=-1)

    return image


def plot_score_map(corner_map, scores):
    """Draw score map overlaid on the density map
    """
    score_map = np.zeros([356, 356, 3])
    score_map[100:, 50:306] = corner_map
    cv2.putText(score_map, 'room_prec: '+str(round(scores['room_prec']*100, 1)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (252, 252, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'room_rec: '+str(round(scores['room_rec']*100, 1)), (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (252, 252, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'corner_prec: '+str(round(scores['corner_prec']*100, 1)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'corner_rec: '+str(round(scores['corner_rec']*100, 1)), (190, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'angles_prec: '+str(round(scores['angles_prec']*100, 1)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(score_map, 'angles_rec: '+str(round(scores['angles_rec']*100, 1)), (190, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return score_map


def plot_room_map(preds, room_map, room_id=0, im_size=256):
    """Draw room polygons overlaid on the density map
    """
    centroid_x = int(np.mean(preds[:, 0]))
    centroid_y = int(np.mean(preds[:, 1]))

    # Get text size to create a background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text = str(room_id)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)


    for i, corner in enumerate(preds):
        if i == len(preds)-1:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[0][0]), round(preds[0][1])), (252, 252, 0), 2)
        else:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[i+1][0]), round(preds[i+1][1])), (252, 252, 0), 2)
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        # cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw white background box with transparency
        overlay = room_map.copy()
        cv2.rectangle(overlay, 
                    (centroid_x - text_width//2 - 1, centroid_y - text_height//2 - 1),
                    (centroid_x + text_width//2 + 1, centroid_y + text_height//2 + 1),
                    (0, 0, 0), 
                    -1)  # Filled rectangle
        cv2.addWeighted(overlay, 0.7, room_map, 0.3, 0, room_map)  # 70% opacity

        # Draw text
        cv2.putText(room_map, 
                    text, 
                    (centroid_x - text_width//2, centroid_y + text_height//2), 
                    font, 
                    font_scale, 
                    (0, 255, 0),
                    thickness)
        
    return room_map


def plot_anno(img, annos, save_path, transformed=False, draw_poly=True, draw_bbx=True, thickness=2):
    """Visualize annotation
    """
    img = np.repeat(np.expand_dims(img,2), 3, axis=2)
    num_inst = len(annos)

    bbx_color = (0, 255, 0)
    # poly_color = (0, 255, 0)
    for j in range(num_inst):
        
        if draw_bbx:
            bbox = annos[j]['bbox']
            if transformed: 
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[2]), round(bbox[3]))
            else:
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[0]+bbox[2]), round(bbox[1]+bbox[3]))
            # Blue color in BGR
            img = cv2.rectangle(img, start_point, end_point, bbx_color, thickness)

        if draw_poly:
            verts = annos[j]['segmentation'][0]
            if isinstance(verts, list):
                verts = np.array(verts)
            verts = verts.reshape(-1,2)

            for i, corner in enumerate(verts):
                if i == len(verts)-1:
                    cv2.line(img, (round(corner[0]), round(corner[1])), (round(verts[0][0]), round(verts[0][1])), (0, 252, 252), 1)
                else:
                    cv2.line(img, (round(corner[0]), round(corner[1])), (round(verts[i+1][0]), round(verts[i+1][1])), (0, 252, 252), 1)
                cv2.circle(img, (round(corner[0]), round(corner[1])), 2, (255, 0, 0), 2)
                cv2.putText(img, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, (0, 255, 0), 1, cv2.LINE_AA)

    imsave(save_path, img)


def plot_coords(ax, ob, color=BLACK, zorder=1, alpha=1, linewidth=1):
    x, y = ob.xy
    ax.plot(x, y, color=color, zorder=zorder, alpha=alpha, linewidth=linewidth, solid_joinstyle='miter')


def plot_corners(ax, ob, color=BLACK, zorder=1, alpha=1):
    x, y = ob.xy
    ax.scatter(x, y, color=color, marker='o')

def get_angle(p1, p2):
    """Get the angle of this line with the horizontal axis.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def filled_arc(e1, e2, direction, radius, ax, color):
    """Draw arc for door
    """
    angle = get_angle(e1,e2)
    if direction == 'counterclock':
        theta1 = angle
        theta2 = angle + 90.0
    else:
        theta1 = angle - 90.0
        theta2 = angle
    circ = mpatches.Wedge(e1, radius, theta1, theta2, fill=True, color=color, linewidth=1, ec='#000000')
    ax.add_patch(circ)


def plot_semantic_rich_floorplan(polygons, file_name, prec=None, rec=None):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for (poly, poly_type) in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)
            if poly_type != 16 and poly_type != 17:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=10)

    # Iterate over all predicted polygons (rooms, doors, windows)
    for (poly, poly_type) in polygons:
        if poly_type == 'outqwall':  # unclear what is this?
            pass
        elif poly_type == 16:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == 17:  # Window
            polygons_windows.append([poly, poly_type])
        else: # regular room
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor='#FFFFFF', alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0.5, linewidth=1, capstyle='round', edgecolor='#000000FF')
            ax.add_patch(patch)
            ax.text(np.mean(poly[:, 0]), np.mean(poly[:, 1]), 
                    semantics_label[poly_type], 
                    fontsize=6, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7)
                    )


    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for (poly, poly_type, door_size) in polygons_doors:

        door_size_y = np.abs(poly[0,1]-poly[1,1])
        door_size_x = np.abs(poly[0,0]-poly[1,0])
        if door_size_y > door_size_x:
            if poly[1,1] > poly[0,1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'clock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'clock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'counterclock', door_size/2, ax, 'white')

        else:
            if poly[1,0] > poly[0,0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'counterclock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'counterclock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'clock', door_size/2, ax, 'white')


    # Draw windows
    for (line, line_type) in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor='#FFFFFF', alpha=1.0, linewidth=1, linestyle='dashed')
        ax.add_patch(patch)

    title = ''
    if prec is not None:
        title = 'prec: ' + str(round(prec * 100, 1)) + ', rec: ' + str(round(rec * 100, 1))
    plt.title(file_name.split('/')[-1] + ' ' + title)
    plt.axis('equal')
    plt.axis('off')

    print(f'>>> {file_name}')
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    fig.savefig(file_name, dpi=fig.dpi)


def plot_semantic_rich_floorplan_tight(polygons, file_name, prec=None, rec=None, plot_text=True, is_bw=False, door_window_index=[16,17], img_w=256, img_h=256):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)
    """

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)


    # Set figure size to exactly 256x256 pixels
    dpi = 100  # Standard screen DPI
    figsize = (img_w/dpi, img_h/dpi)  # Convert pixels to inches

    # Create square figure with fixed size
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Set equal aspect ratio and the limits to exactly match the coordinate space
    ax.set_aspect('equal')
    ax.set_xlim(0, img_w - 1) # 255
    ax.set_ylim(0, img_h - 1) # 255

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for (poly, poly_type) in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)

            if poly_type not in door_window_index:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=10)
    
    # Iterate over all predicted polygons (rooms, doors, windows)
    for (poly, poly_type) in polygons:
        if poly_type == 'outqwall':  # unclear what is this?
            pass
        elif poly_type == door_window_index[0]:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == door_window_index[1]:  # Window
            polygons_windows.append([poly, poly_type])
        else: # regular room
            if len(poly) < 3:
                continue
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor='#FFFFFF', alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            if not is_bw:
                patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], alpha=0.5, linewidth=1, capstyle='round', edgecolor='#000000FF')
                ax.add_patch(patch)
            if plot_text:
                ax.text(np.mean(poly[:, 0]), np.mean(poly[:, 1]), semantics_label[poly_type], size=6, horizontalalignment='center', verticalalignment='center')

    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for (poly, poly_type, door_size) in polygons_doors:

        door_size_y = np.abs(poly[0,1]-poly[1,1])
        door_size_x = np.abs(poly[0,0]-poly[1,0])
        if door_size_y > door_size_x:
            if poly[1,1] > poly[0,1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'clock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'clock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'counterclock', door_size/2, ax, 'white')

        else:
            if poly[1,0] > poly[0,0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'counterclock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'counterclock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'clock', door_size/2, ax, 'white')


    # Draw windows
    for (line, line_type) in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor='#FFFFFF', alpha=1.0, linewidth=1, linestyle='dashed')
        ax.add_patch(patch)

    if plot_text:
        title = ''
        if prec is not None:
            title = 'prec: ' + str(round(prec * 100, 1)) + ', rec: ' + str(round(rec * 100, 1))
        plt.title(file_name.split('/')[-1] + ' ' + title)

    # plt.axis('equal')
    plt.axis('off')

    print(f'>>> {file_name}')
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    if is_bw:
        plt.set_cmap(get_cmap('gray'))
    
    fig.savefig(file_name, dpi=dpi, bbox_inches='tight', pad_inches=0)


def plot_semantic_rich_floorplan_nicely(polygons, 
                                      file_name, 
                                      prec=None, 
                                      rec=None, 
                                      plot_text=True, 
                                      is_bw=False, 
                                      door_window_index=[16,17], 
                                      img_w=256, 
                                      img_h=256,
                                      semantics_label_mapping=semantics_label,
                                      ):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)
    """

    # Set figure size to exactly 256x256 pixels
    dpi = 150  # Standard screen DPI
    figsize = (img_w/dpi, img_h/dpi)  # Convert pixels to inches

    # Create square figure with fixed size
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    # Set equal aspect ratio and the limits to exactly match the coordinate space
    # ax.set_aspect('equal')
    # ax.set_xlim(0, img_w - 1)
    # ax.set_ylim(0, img_h - 1)

    # Disable autoscaling
    ax.autoscale(False)
    
    # Disable adjusting automatically
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for (poly, poly_type) in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)

            if poly_type not in door_window_index:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=2)
    
    # Iterate over all predicted polygons (rooms, doors, windows)
    for (poly, poly_type) in polygons:
        if poly_type == door_window_index[0]:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == door_window_index[1]:  # Window
            polygons_windows.append([poly, poly_type])
        else: # regular room
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor='#FFFFFF', alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            if not is_bw:
                patch = PolygonPatch(polygon, facecolor=semantics_cmap[poly_type], 
                                     alpha=0.5, linewidth=1, 
                                     capstyle='round', edgecolor='#000000FF')
                ax.add_patch(patch)
            if plot_text:
                ax.text(np.mean(poly[:, 0]), np.mean(poly[:, 1]), 
                        semantics_label_mapping[poly_type], 
                        fontsize=6, 
                        ha='center', 
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7)
                        )

    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for (poly, poly_type, door_size) in polygons_doors:

        door_size_y = np.abs(poly[0,1]-poly[1,1])
        door_size_x = np.abs(poly[0,0]-poly[1,0])
        if door_size_y > door_size_x:
            if poly[1,1] > poly[0,1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'clock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'clock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'counterclock', door_size/2, ax, 'white')

        else:
            if poly[1,0] > poly[0,0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, 'counterclock', door_size, ax, 'white')
            else:
                filled_arc(e1, e2, 'counterclock', door_size/2, ax, 'white')
                filled_arc(e2, e1, 'clock', door_size/2, ax, 'white')


    # Draw windows
    for (line, line_type) in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor='#FFFFFF', alpha=1.0, linewidth=1, linestyle='dashed')
        ax.add_patch(patch)

    if plot_text:
        title = ''
        if prec is not None:
            title = 'prec: ' + str(round(prec * 100, 1)) + ', rec: ' + str(round(rec * 100, 1))
        plt.title(file_name.split('/')[-1] + ' ' + title)

    print(f'>>> {file_name}')
    plt.axis('equal')
    plt.axis('off')
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    if is_bw:
        plt.set_cmap(get_cmap('gray'))
    fig.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


# def plot_semantic_rich_floorplan_opencv(polygons, file_name, img_w=256, img_h=256, door_window_index=[16,17], semantics_label_mapping=semantics_label, is_bw=False):
#     """
#     Plot semantically-rich floorplan using OpenCV.

#     Args:
#         polygons (list): A list of polygons, where each polygon is a list of (x, y) coordinates.
#         file_name (str): Path to save the output image.
#         img_w (int): Width of the image.
#         img_h (int): Height of the image.
#         is_bw (bool): If True, use black and white colors only.
#     """
#     # Create a blank black image
#     if is_bw:
#         image = np.zeros((img_h, img_w), dtype=np.uint8)  # Grayscale image
#     else:
#         image = np.zeros((img_h, img_w, 3), dtype=np.uint8)  # RGB image

#     # Define colors
#     door_color = (200, 200, 200) if is_bw else (255, 0, 0)  # Light gray for BW, Blue for RGB
#     window_color = (150, 150, 150) if is_bw else (0, 0, 255)  # Dark gray for BW, Red for RGB
#     text_color = (0, 0, 0) if is_bw else (0, 0, 0)  # White text for both modes

#     # Create an overlay for transparency
#     overlay = image.copy()

#     # Draw polygons
#     for poly, poly_type in polygons:
#         points = np.array(poly, dtype=np.int32)  # Convert to NumPy array
#         if poly_type == door_window_index[0]:  # Door
#             cv2.polylines(overlay, [points], isClosed=True, color=door_color[:3], thickness=2)
#         elif poly_type == door_window_index[1]:  # Window
#             cv2.polylines(overlay, [points], isClosed=True, color=window_color[:3], thickness=2)
#         else:  # Regular room
#             rgb_color = ImageColor.getcolor(semantics_cmap[poly_type], "RGB")
#             cv2.fillPoly(overlay, [points], color=(rgb_color[2], rgb_color[1], rgb_color[0]))

#             # Calculate the centroid of the polygon for the label
#             M = cv2.moments(points)
#             if M["m00"] != 0:  # Avoid division by zero
#                 centroid_x = int(M["m10"] / M["m00"])
#                 centroid_y = int(M["m01"] / M["m00"])
#                 # Get the text size for the label
#                 label = semantics_label_mapping[poly_type]
#                 font_scale = 0.5
#                 text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
#                 text_width, text_height = text_size[0], text_size[1]

#                 # # Define the top-left and bottom-right corners of the rectangle
#                 # top_left = (centroid_x - text_width // 2 - font_scale, centroid_y - text_height // 2 - font_scale)
#                 # bottom_right = (centroid_x + text_width // 2 + font_scale, centroid_y + text_height // 2 + font_scale)

#                 # # Draw a white rectangle as the background for the text
#                 # cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)

#                 # Draw the label at the centroid
#                 cv2.putText(
#                     overlay,
#                     semantics_label_mapping[poly_type],  # Label text
#                     (centroid_x, centroid_y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale,  # Font scale
#                     text_color[:3],
#                     1,  # Thickness
#                     cv2.LINE_AA,
#                 )

#     # Blend the overlay with the original image
#     alpha = 0.7  # Transparency level
#     cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

#     # Save the image
#     cv2.imwrite(file_name, image)
#     print(f"Saved floorplan to {file_name}")


def plot_semantic_rich_floorplan_opencv(polygons, file_name, img_w=256, img_h=256, 
                                       door_window_index=[16,17], 
                                       semantics_label_mapping=semantics_label, 
                                       is_bw=False,
                                       ):
    """
    Plot semantically-rich floorplan using OpenCV with improved quality.
    
    Args:
        polygons (list): A list of polygons, where each polygon is a list of (x, y) coordinates.
        file_name (str): Path to save the output image.
        img_w (int): Width of the output image.
        img_h (int): Height of the output image.
        door_window_index (list): Indices for door and window types.
        semantics_label_mapping (dict): Mapping from polygon type to semantic label.
        is_bw (bool): If True, use black and white colors only.
        line_thickness (int): Thickness of lines for polygons and doors/windows.
        text_padding (int): Padding around text labels.
        font_scale (float): Scale factor for text size.
        room_alpha (float): Transparency for room colors (0.0-1.0).
        anti_aliasing (bool): Whether to use anti-aliasing for lines.
    """
    line_thickness=2
    text_padding=5
    font_scale=1.0
    room_alpha=0.6
    # Create a white background image (more conventional for floorplans)
    if is_bw:
        image = np.ones((img_h, img_w), dtype=np.uint8) * 255  # White grayscale image
    else:
        image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White RGB image
    
    # Create a separate layer for room colors
    overlay = image.copy()
    
    # Track polygons for each type for proper layering
    room_polygons = []
    door_polygons = []
    window_polygons = []
    
    # Sort polygons by type
    for poly, poly_type in polygons:
        if len(poly) < 2:  # Skip invalid polygons
            continue
            
        points = np.array(poly, dtype=np.int32)
        
        if poly_type == door_window_index[0]:  # Door
            door_polygons.append((points, poly_type))
        elif poly_type == door_window_index[1]:  # Window
            window_polygons.append((points, poly_type))
        else:  # Room
            room_polygons.append((points, poly_type))
    
    # Draw rooms first (bottom layer)
    for points, poly_type in room_polygons:
        # Fill room with color
        if not is_bw:
            # Get RGB color from semantics_cmap and convert from RGB to BGR for OpenCV
            rgb_color = ImageColor.getcolor(semantics_cmap[poly_type], "RGB")
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            
            cv2.fillPoly(overlay, [points], color=bgr_color)
        else:
            # Use light gray for rooms in BW mode
            cv2.fillPoly(overlay, [points], color=(240, 240, 240))
        
        # Draw room outline
        line_type = cv2.LINE_AA
        cv2.polylines(image, [points], isClosed=True, 
                     color=(0, 0, 0), thickness=line_thickness, 
                     lineType=line_type)
    
    # Blend overlay with transparency
    cv2.addWeighted(overlay, room_alpha, image, 1 - room_alpha, 0, image)
    
    # Draw doors with proper styling
    for points, _ in door_polygons:
        if len(points) >= 2:
            # For doors, we can improve by drawing arcs to represent swing
            # Here we draw them as thick lines with distinctive color
            door_color = (100, 100, 100) if is_bw else (0, 0, 255)  # Gray for BW, Red for RGB
            line_type = cv2.LINE_AA
            cv2.polylines(image, [points], isClosed=False, 
                         color=door_color, thickness=line_thickness*2,
                         lineType=line_type)
    
    # Draw windows with dashed styling
    for points, _ in window_polygons:
        if len(points) >= 2:
            window_color = (150, 150, 150) if is_bw else (255, 0, 0)  # Gray for BW, Blue for RGB
            
            # Create dashed line effect for windows
            if len(points) == 2:
                # For a simple line window
                pt1, pt2 = points[0], points[1]
                dash_length = 5
                
                # Calculate line parameters
                length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                if length > 0:
                    num_dashes = max(2, int(length / (2 * dash_length)))
                    
                    for i in range(num_dashes):
                        start_ratio = i / num_dashes
                        end_ratio = (i + 0.5) / num_dashes
                        
                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
                        
                        line_type = cv2.LINE_AA
                        cv2.line(image, (start_x, start_y), (end_x, end_y), 
                                window_color, thickness=line_thickness,
                                lineType=line_type)
            else:
                # For multi-point windows
                line_type = cv2.LINE_AA
                cv2.polylines(image, [points], isClosed=True, 
                             color=window_color, thickness=line_thickness,
                             lineType=line_type)
    
    # Add room labels
    for points, poly_type in room_polygons:
        # Calculate the centroid for text placement
        M = cv2.moments(points)
        if M["m00"] != 0:  # Avoid division by zero
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            # Get room label
            label = semantics_label_mapping[poly_type]
            
            # Get text size for centering and background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, 1)[0]
            
            # Calculate text background rectangle
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            
            # Create background for text
            rect_top_left = (text_x - text_padding, text_y - text_size[1] - text_padding)
            rect_bottom_right = (text_x + text_size[0] + text_padding, text_y + text_padding)
            
            # Draw semi-transparent white background for text
            background = image.copy()
            cv2.rectangle(background, rect_top_left, rect_bottom_right, 
                         (255, 255, 255), -1)
            
            # Blend the background
            cv2.addWeighted(background, 0.7, image, 0.3, 0, image)
            
            # Draw the text
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                1,  # Thickness
                cv2.LINE_AA,  # Anti-aliased text
            )
    
    # Add border around the image for better framing
    cv2.rectangle(image, (0, 0), (img_w-1, img_h-1), (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save with high quality
    if is_bw:
        cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print(f"Saved improved floorplan to {file_name}")
    
    return image  # Return the image for optional further processing or visualization