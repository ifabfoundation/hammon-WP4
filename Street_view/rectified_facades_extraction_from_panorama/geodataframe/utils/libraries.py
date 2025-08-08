import overpy
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, linemerge, nearest_points
from shapely.validation import make_valid
from shapely.errors import TopologicalError
import networkx as nx
import math
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from s3_library.S3Client import S3Client