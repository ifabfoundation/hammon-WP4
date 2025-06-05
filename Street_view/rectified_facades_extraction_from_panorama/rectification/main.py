import numpy as np
from classes.PanoramaProcessor import PanoramaProcessor


def main():
    """
    Main function that configures and starts the panorama processor.
    """
    # Configuration
    root = 'Pano_new'
    country_city = 'New'
    plot_redundant = True
    save_directly = False
    new_count = 5
    
    # New parameters for the GeoDataFrame
    geodataframe_path = '../Enhanced_facades_for_extraction_639_facades.geojson'
    images_base_path = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Documents - Hammon\WP4\data\PANO_new'

    # Initialize and start the processor
    processor = PanoramaProcessor(
        root=root,
        country_city=country_city,
        plot_redundant=plot_redundant,
        save_directly=save_directly,
        new_count=new_count,
        geodataframe_path=geodataframe_path,
        images_base_path=images_base_path
    )
    
    processor.process_all_panoramas()


if __name__ == "__main__":
    # Uncomment to use a fixed seed for reproducibility
    # np.random.seed(1)
    main()
