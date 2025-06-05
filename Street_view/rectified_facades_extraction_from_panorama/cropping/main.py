from BuildingExtractor import BuildingFacadeExtractor

def main():
    """
    Main execution function.
    """
    # Create extractor with custom configuration (esempio di utilizzo diretto)
    extractor = BuildingFacadeExtractor(
        geojson_path='../Enhanced_facades_for_extraction_639_facades.geojson',
        root_dir='../rectification/Pano_new',
        country_city='New',
        images_base_folder=r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Documents - Hammon\WP4\data\PANO_new',
        save_cropped_buildings=True,
        output_dir='./'
    )

    # Extract all buildings
    extractor.extract_all_buildings()


if __name__ == "__main__":
    main()