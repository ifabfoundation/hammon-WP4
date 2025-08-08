import glob
from util.default_params import default_params
import skimage.io
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os.path
import skimage.io
import time
import json
import multiprocessing

import skimage.io
from Panos.Pano_rectification import simon_rectification
from Panos.Pano_project import project_face, stitch_tiles, render_imgs
import matplotlib.pyplot as plt

from Panos.Pano_visualization import R_heading, draw_all_vp_and_hl_color, draw_all_vp_and_hl_bi, \
    draw_zenith_on_top_color, draw_zenith_on_top_bi, draw_sphere_zenith, R_roll, R_pitch
from Panos.Pano_zp_hvp import calculate_consensus_zp
from Panos.Pano_consensus_vis import draw_consensus_zp_hvps, draw_consensus_rectified_sphere, \
    draw_center_hvps_rectified_sphere, draw_center_hvps_on_panorams
from vanishing_points_utils import Pano_hvp
from Panos.Pano_histogram import calculate_histogram
from Panos.Pano_project import project_facade_for_refine
import tempfile
import shutil