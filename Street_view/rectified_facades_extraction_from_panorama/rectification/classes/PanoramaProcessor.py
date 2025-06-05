from util.libraries import *

class PanoramaProcessor:
    """
    Class for processing panoramic images and extracting rectified facades.
    """
    
    def __init__(self, root='Pano_new', country_city='New', plot_redundant=True, save_directly=False, new_count=5, geodataframe_path=None, images_base_path=None):
        """
        Initialize the panorama processor.
        
        Args:
            root (str): Main directory
            country_city (str): Subdirectory for country/city
            plot_redundant (bool): Whether to display redundant plots
            save_directly (bool): Whether to save images directly
            new_count (int): Counter for new images
            geodataframe_path (str): Path to the GeoDataFrame file to read for image selection
            images_base_path (str): Base path for images that will be combined with PATH, RUN, and FOTO from the geodataframe
        """
        self.root = root
        self.country_city = country_city
        self.plot_redundant = plot_redundant
        self.save_directly = save_directly
        self.new_count = new_count
        self.geodataframe_path = geodataframe_path
        self.images_base_path = images_base_path
        
        # Setting up directories
        self.img_folder = os.path.join(root, country_city, 'images/')
        self.inter_dir = os.path.join(root, 'Pano_hl_z_vp/')
        self.rendering_output_folder = os.path.join(root, country_city, 'Rendering')
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.rendering_output_folder):
            os.makedirs(self.rendering_output_folder)
    
    def get_image_list(self):
        """
        Gets the list of panoramic images to process based on GeoDataFrame if provided,
        otherwise falls back to all images in the directory.
        
        Returns:
            list: List of image paths
        """
        if self.geodataframe_path and self.images_base_path:
            # Import geopandas here to ensure it's only required when using this feature
            import geopandas as gpd
            
            # Read the geodataframe
            try:
                gdf = gpd.read_file(self.geodataframe_path)
                
                # Construct image paths from base path and columns PATH, RUN, and FOTO
                image_list = []
                for _, row in gdf.iterrows():
                    # Construct the path using the base path and the columns from the geodataframe
                    if 'PATH' in row and 'RUN' in row and 'FOTO' in row:
                        img_path = os.path.join(self.images_base_path, str(row['PATH']), str(row['RUN']), str(row['FOTO']))
                        image_list.append(img_path)
                    else:
                        print(f"Warning: Row is missing required columns PATH, RUN, or FOTO")
                
                print(f"Found {len(image_list)} images from geodataframe")
                return image_list
            except Exception as e:
                print(f"Error reading geodataframe: {e}")
                print("Falling back to reading all images in the directory")
        
        # Fallback to the original behavior if geodataframe path is not provided or if there was an error
        image_list = glob.glob(self.img_folder + '*.jpg')
        image_list.sort()
        return image_list
    
    def setup_temp_folders(self, task='hahaha', thread_num=1):
        """
        Configures temporary folders and removes old files.
        
        Args:
            task (str): Task name
            thread_num (int): Thread number
            
        Returns:
            str, str: Paths to temporary folders
        """
        thread = str(thread_num) + '/'
        tmp_folder = os.path.join(self.root, self.country_city, 'tmp', task, thread)
        tmp_folder_ifab = os.path.join(self.root, self.country_city, 'tmp_ifab', task, thread)
        
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        
        # Remove old temporary files
        remove_list = glob.glob(tmp_folder + '*.jpg')
        for i in remove_list:
            os.remove(i)
            
        return tmp_folder, tmp_folder_ifab
    
    def process_tiles(self, panorama_img, tmp_folder, tmp_folder_ifab):
        """
        Generates and processes sections (tiles) of the panoramic image.
        
        Args:
            panorama_img (numpy.ndarray): Panoramic image
            tmp_folder (str): Temporary folder
            tmp_folder_ifab (str): IFAB temporary folder
            
        Returns:
            list: List of tile paths
            list: Horizon data
            list: Horizontal vanishing point data
            list: Vanishing point group data
            list: Zenith data
            list: Zenith group data
            list: Line data
            list: Zenith data in homogeneous coordinates
            list: Vanishing point data in homogeneous coordinates
            list: Line data in homogeneous coordinates
        """
        # Generate tiles from the panoramic image
        tilelist = render_imgs(panorama_img, tmp_folder, tmp_folder_ifab, self.save_directly)
        
        if not self.save_directly:
            # Use already saved images
            tilelist = glob.glob(tmp_folder + '*.jpg')
            tilelist.sort()
        
        # Initialize lists for rectification data
        hl = []
        hvps = []
        hvp_groups = []
        z = []
        z_group = []
        ls = []
        z_homo = []
        hvp_homo = []
        ls_homo = []
        
        # Process each tile
        for i in range(len(tilelist)):
            [tmp_hl, tmp_hvps, tmp_hvp_groups, tmp_z, tmp_z_group, tmp_ls, 
             tmp_z_homo, tmp_hvp_homo, tmp_ls_homo, params] = simon_rectification(
                 tilelist[i], i, self.inter_dir, self.root, self.new_count)
            
            hl.append(tmp_hl)
            hvps.append(tmp_hvps)
            hvp_groups.append(tmp_hvp_groups)
            z.append(tmp_z)
            z_group.append(tmp_z_group)
            ls.append(tmp_ls)
            z_homo.append(tmp_z_homo)
            hvp_homo.append(tmp_hvp_homo)
            ls_homo.append(tmp_ls_homo)
        
        return tilelist, hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params
    
    def calculate_zenith_points(self, z_homo, hvp_homo, im):
        """
        Calculates zenith points from all perspectives.
        
        Args:
            z_homo (list): Zenith points in homogeneous coordinates
            hvp_homo (list): Horizontal vanishing points in homogeneous coordinates
            im (PIL.Image): Panoramic image
            
        Returns:
            numpy.ndarray: Zenith points
            list: Transformed horizontal vanishing points
        """
        # Calculate zenith points from all perspectives
        zenith_points = np.array([R_heading(np.pi / 2 * (i - 1)).dot(zenith) for i, zenith in enumerate(z_homo)])
        points2 = np.array([R_heading(np.pi / 2 * (i - 1)).dot(np.array([0., 0., 1.])) for i in range(len(z_homo))])
        hv_points = [(R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T for i, hv_p in enumerate(hvp_homo)]
        
        if self.plot_redundant:
            draw_all_vp_and_hl_color(zenith_points, hv_points, im.copy(), self.root)
            draw_all_vp_and_hl_bi(zenith_points, hv_points, im.copy(), self.root)
            draw_sphere_zenith(zenith_points, hv_points, self.root)
        
        return zenith_points, hv_points
    
    def calculate_consensus(self, zenith_points, ls_homo, im, params):
        """
        Calculates the consensus zenith and horizontal vanishing points.
        
        Args:
            zenith_points (numpy.ndarray): Zenith points
            ls_homo (list): Lines in homogeneous coordinates
            im (PIL.Image): Panoramic image
            params: Parameters for processing
            
        Returns:
            numpy.ndarray: Best zenith point
            list: Rectified vanishing points
            float: Pitch
            float: Roll
        """
        # Calculate the consensus zenith
        [zenith_consensus, best_zenith] = calculate_consensus_zp(zenith_points, method='svd')
        
        # Transform consensus zenith points to original coordinates
        zenith_consensus_org = np.array([R_heading(-np.pi / 2 * (i - 1)).dot(zenith) 
                                        for i, zenith in enumerate(zenith_consensus)])
        
        # Calculate horizontal vanishing points from the consensus zenith point
        result_list = []
        for i in range(len(zenith_consensus_org)):
            result = Pano_hvp.get_all_hvps(ls_homo[i], zenith_consensus_org[i], params)
            result_list.append(result)
        
        hvps_consensus_org = []
        for i in range(len(result_list)):
            hvps_consensus_org.append(result_list[i])
        
        hvps_consensus_uni = [(R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T 
                             for i, hv_p in enumerate(hvps_consensus_org)]
        
        if self.plot_redundant:
            draw_consensus_zp_hvps(best_zenith, hvps_consensus_uni, im.copy(), self.root)
        
        # Calculate pitch and roll
        pitch = np.arctan(best_zenith[2] / best_zenith[1])
        roll = -np.arctan(best_zenith[0] / np.sign(best_zenith[1]) * 
                         np.hypot(best_zenith[1], best_zenith[2]))
        
        # Rectify horizontal vanishing points
        hvps_consensus_rectified = [R_roll(-roll).dot(R_pitch(-pitch).dot(vp.T)).T 
                                   for vp in hvps_consensus_uni]
        
        if self.plot_redundant:
            draw_consensus_rectified_sphere(hvps_consensus_rectified, self.root)
        
        # Calculate histogram of horizontal vanishing points
        final_hvps_rectified = calculate_histogram(hvps_consensus_rectified, self.root, self.plot_redundant)
        
        if self.plot_redundant:
            draw_center_hvps_rectified_sphere(np.array(final_hvps_rectified), self.root)
            draw_center_hvps_on_panorams(best_zenith, np.array(final_hvps_rectified), 
                                         im.copy(), pitch, roll, self.root)
        
        return best_zenith, final_hvps_rectified, pitch, roll
    
    def process_single_panorama(self, im_path):
        """
        Process a single panoramic image.
        
        Args:
            im_path (str): Path to the panoramic image
        """
        print(im_path)
        im = Image.open(im_path)
        rendering_img_base = os.path.join(self.rendering_output_folder, 
                                         os.path.splitext(os.path.basename(im_path))[0])
        
        # Setup temporary folders
        tmp_folder, tmp_folder_ifab = self.setup_temp_folders()
        
        # Load the panoramic image
        panorama_img = skimage.io.imread(im_path)
        
        # Process the tiles
        tilelist, hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params = self.process_tiles(
            panorama_img, tmp_folder, tmp_folder_ifab)
        
        # Remove temporary files
        remove_list = glob.glob(tmp_folder + '*.jpg')
        for i in remove_list:
            os.remove(i)
        
        # Calculate zenith points
        zenith_points, hv_points = self.calculate_zenith_points(z_homo, hvp_homo, im)
        
        # Calculate consensus and rectify
        best_zenith, final_hvps_rectified, pitch, roll = self.calculate_consensus(
            zenith_points, ls_homo, im, params)
        
        # Render rectified facades
        project_facade_for_refine(np.array(final_hvps_rectified), im.copy(), pitch, roll, 
                                 im_path, self.root, tmp_folder, rendering_img_base, str(self.new_count))
        print(100)  # Completion indicator
    
    def process_all_panoramas(self):
        """
        Process all panoramic images in the image folder.
        With error handling to continue processing even if an image fails.
        """
        image_list = self.get_image_list()
        print(f"Trovate {len(image_list)} immagini da elaborare")
        
        # Process each image with error handling
        successful = 0
        failed = 0
        
        for i, im_path in enumerate(image_list):
            try:
                print(f"\n[{i+1}/{len(image_list)}] Elaborazione di: {im_path}")
                self.process_single_panorama(im_path)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"ERRORE durante l'elaborazione di {im_path}: {str(e)}")
                print(f"Continuo con l'immagine successiva...")
                continue
        
        print(f"\nRiepilogo elaborazione:")
        print(f"- Immagini elaborate con successo: {successful}")
        print(f"- Immagini fallite: {failed}")
        print(f"- Totale: {len(image_list)}")
