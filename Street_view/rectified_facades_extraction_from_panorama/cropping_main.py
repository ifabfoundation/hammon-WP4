from cropping.BuildingExtractor import BuildingFacadeExtractor
from s3_library.S3Client import S3Client


def main():
    s3_client = S3Client()

    for zone in s3_client.list_all_sub_folders('data', 'geodataframe/results'):
        file_names = s3_client.list_files('data', 'geodataframe/results/' + zone)
        geodataframe_path = [name for name in file_names if name.endswith('.geojson')][0]

        print(geodataframe_path)
        print(zone)

        # Create extractor with custom configuration (esempio di utilizzo diretto)
        extractor = BuildingFacadeExtractor(
            geojson_path=geodataframe_path,
            root_dir='rectification/',
            country_city='results/',
            save_cropped_buildings=True,
            output_dir='cropping/',
            s3_client=s3_client,
            zone_name=zone
        )

        # Extract all buildings
        extractor.extract_all_buildings()

if __name__ == "__main__":
    main()
