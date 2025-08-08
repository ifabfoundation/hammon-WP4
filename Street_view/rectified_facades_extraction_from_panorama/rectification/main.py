import numpy as np
from classes.PanoramaProcessor import PanoramaProcessor


def main():
    """
    Main function that configures and starts the panorama processor.
    """
    # Configuration
    root = 'rectification/'
    country_city = 'results/'
    plot_redundant = True
    save_directly = False
    new_count = 5
    
    # New parameters for the GeoDataFrame
    geodataframe_path = 'geodataframe/results/Enhanced_facades_for_extraction_718_facades.geojson'
    images_base_path = '2025/sudest2/pano/'

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
