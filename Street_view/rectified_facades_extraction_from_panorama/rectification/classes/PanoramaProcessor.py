from util.libraries import *

class PanoramaProcessor:
    """
    Class for processing panoramic images and extracting rectified facades.
    """
    
    def __init__(self, root='Pano_new', country_city='New', plot_redundant=True, save_directly=False, new_count=5, geodataframe_path=None, images_base_path=None, s3_client=None):
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
        self.s3_client = s3_client
        
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
                gdf = self.s3_client.read_geodataframe('data', self.geodataframe_path)
                
                # Construct image paths from base path and columns PATH, RUN, and FOTO
                image_list = []
                for _, row in gdf.iterrows():
                    # Construct the path using the base path and the columns from the geodataframe
                    if 'PATH' in row and 'RUN' in row and 'FOTO' in row:
                        img_path = f"{self.images_base_path}{str(row['PATH'])}/pano/{str(row['RUN'])}/{str(row['FOTO'])}"
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
    
    def setup_temp_folders(self, task='1', thread_num=1):
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
        
        # Check if there are files with the prefix
        files_tmp = self.s3_client.list_files('data', tmp_folder)
        files_tmp_ifab = self.s3_client.list_files('data', tmp_folder_ifab)
        
        if 'Contents' in files_tmp:
            self.s3_client.delete_files_with_prefix('data', tmp_folder)
        if 'Contents' in files_tmp_ifab:
            self.s3_client.delete_files_with_prefix('data', tmp_folder_ifab)
            
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
        tilelist = render_imgs(panorama_img, tmp_folder, tmp_folder_ifab, self.save_directly, self.s3_client)
        
        if not self.save_directly:
            # Use already saved images
            s3_files = self.s3_client.list_files('data', tmp_folder)
            # Filter only jpg
            tilelist = [file for file in s3_files if file.endswith('.jpg')]
            tilelist.sort()

        local_tmp_dir = tempfile.mkdtemp(prefix="s3_tiles_")

        # Scarica i file da S3 nella directory temporanea
        local_tilelist = []
        for i, s3_path in enumerate(tilelist):
            # Estrai il nome del file dal percorso S3
            file_name = os.path.basename(s3_path)
            local_path = os.path.join(local_tmp_dir, file_name)
            
            try:
                # Scarica il file da S3
                self.s3_client.download_file('data', s3_path, local_path)
                local_tilelist.append(local_path)
            except Exception as e:
                print(f"DEBUG - Errore durante il download del tile {i+1}: {str(e)}")
            
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
        params = None  # Inizializza params qui per evitare l'errore
        
        for i in range(len(local_tilelist)):
            try:
                print(f"STEP - Processamento tile {i+1}/{len(local_tilelist)}")
                [tmp_hl, tmp_hvps, tmp_hvp_groups, tmp_z, tmp_z_group, tmp_ls, 
                tmp_z_homo, tmp_hvp_homo, tmp_ls_homo, params] = simon_rectification(
                    local_tilelist[i], i, self.inter_dir, self.root, self.new_count)
                
                hl.append(tmp_hl)
                hvps.append(tmp_hvps)
                hvp_groups.append(tmp_hvp_groups)
                z.append(tmp_z)
                z_group.append(tmp_z_group)
                ls.append(tmp_ls)
                z_homo.append(tmp_z_homo)
                hvp_homo.append(tmp_hvp_homo)
                ls_homo.append(tmp_ls_homo)
                print(f"STEP - Tile {i+1} processato con successo")
            except Exception as e:
                print(f"STEP - Errore durante il processamento del tile {i+1}: {str(e)}")

        try:
            shutil.rmtree(local_tmp_dir)
            print(f"STEP - Directory temporanea eliminata: {local_tmp_dir}")
        except Exception as e:
            print(f"STEP - Errore durante l'eliminazione della directory temporanea: {str(e)}")
        
        # Verifica che ci siano dati validi
        if len(tilelist) == 0 or params is None:
            raise ValueError("Nessun tile processato o parametri non inizializzati correttamente")
            
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
            draw_all_vp_and_hl_color(zenith_points, hv_points, im.copy(), self.root, self.s3_client)
            draw_all_vp_and_hl_bi(zenith_points, hv_points, im.copy(), self.root, self.s3_client)
            #draw_sphere_zenith(zenith_points, hv_points, self.root, self.s3_client)
        
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
        
        #if self.plot_redundant:
            #draw_consensus_rectified_sphere(hvps_consensus_rectified, self.root)
        
        # Calculate histogram of horizontal vanishing points
        final_hvps_rectified = calculate_histogram(hvps_consensus_rectified, self.root, self.plot_redundant)
        
        if self.plot_redundant:
            #draw_center_hvps_rectified_sphere(np.array(final_hvps_rectified), self.root)
            draw_center_hvps_on_panorams(best_zenith, np.array(final_hvps_rectified), 
                                         im.copy(), pitch, roll, self.root, self.s3_client)
        
        return best_zenith, final_hvps_rectified, pitch, roll
    
    def process_single_panorama(self, im_path):
        """
        Process a single panoramic image.
        
        Args:
            im_path (str): Path to the panoramic image
        """
        print(im_path)
        try:
            print(f"STEP - Tentativo di caricamento immagine da S3: {im_path}")
            pil_image = self.s3_client.read_image('geolander.streetview', im_path)
            
            panorama_img = np.array(pil_image)
            
            rendering_img_base = os.path.join(self.rendering_output_folder, 
                                             os.path.splitext(os.path.basename(im_path))[0])
            
            # Setup temporary folders
            print(f"STEP - Configurazione cartelle temporanee")
            tmp_folder, tmp_folder_ifab = self.setup_temp_folders()
            
            # Process the tiles
            print(f"STEP - Inizio processamento tile")
            tilelist, hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params = self.process_tiles(
                panorama_img, tmp_folder, tmp_folder_ifab)
            
            # Remove temporary files
            self.s3_client.delete_files_with_prefix('data', tmp_folder)
            
            # Calculate zenith points
            print(f"STEP - Calcolo punti zenit")
            zenith_points, hv_points = self.calculate_zenith_points(z_homo, hvp_homo, pil_image)
            
            # Calculate consensus and rectify
            print(f"STEP - Calcolo consensus e rectificazione")
            best_zenith, final_hvps_rectified, pitch, roll = self.calculate_consensus(
                zenith_points, ls_homo, pil_image, params)
            
            # Render rectified facades
            print(f"STEP - Rendering facades")
            project_facade_for_refine(np.array(final_hvps_rectified), pil_image.copy(), pitch, roll, 
                                     im_path, self.root, tmp_folder, rendering_img_base, str(self.new_count), self.s3_client)
            print(f"STEP - Rendering facades completato")
        except Exception as e:
            print(f"ERRORE durante l'elaborazione di {im_path}: {str(e)}")
            raise  # Rilanciamo l'eccezione per essere catturata dal metodo chiamante
    
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
