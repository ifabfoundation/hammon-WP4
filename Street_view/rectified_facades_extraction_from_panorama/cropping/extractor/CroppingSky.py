from cropping.utils.libraries import *

class SkyCropper:
    """
    Class for detecting and cropping sky from building facade images.
    This is intended to be used as a final step in the building extraction pipeline.
    """
    
    def __init__(self, sky_offset: int = 20):
        """
        Initialize the SkyCropper with configuration parameters.
        
        Args:
            sky_offset (int): Additional offset above the detected roof line
        """
        self.sky_offset = sky_offset
    
    def compute_horizontal_gradient_sum(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the sum of horizontal gradient for each row of the image.
        
        Args:
            image (np.ndarray): Input image (grayscale)
        
        Returns:
            np.ndarray: Array of gradient sums for each row
        """
        # Compute gradient in y-direction (vertical)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Take absolute value of gradient
        abs_sobel_y = np.abs(sobel_y)
        
        # Sum across each row to get a 1D array of gradient magnitudes
        gradient_sum = np.sum(abs_sobel_y, axis=1)
        
        return gradient_sum

    def find_roof_line(self, gradient_sum: np.ndarray, window_size: int = 11, 
                   threshold_factor: float = 1.5) -> int:
        """
        Find the roof line by detecting significant changes in gradient sums.
        
        Args:
            gradient_sum (np.ndarray): Array of gradient sums for each row
            window_size (int): Size of window for local maxima detection
            threshold_factor (float): Factor to determine significant gradient changes
        
        Returns:
            int: Row index representing the roof line
        """
        # Smooth the gradient sum to reduce noise
        smoothed = cv2.GaussianBlur(gradient_sum.reshape(-1, 1), (1, window_size), 0).flatten()
        
        # Compute first derivative of the smoothed gradient sum
        derivative = np.diff(smoothed)
        
        # Find local maxima in the derivative (indicating rapid gradient changes)
        # We're only interested in the top half of the image where roof-sky boundaries typically occur
        top_half = len(derivative) // 2
        peaks = []
        
        for i in range(1, top_half - 1):
            if derivative[i] > derivative[i-1] and derivative[i] > derivative[i+1]:
                peaks.append((i, derivative[i]))
        
        # Sort peaks by magnitude (highest gradient change first)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest significant peak (roof line)
        if peaks:
            return peaks[0][0]
        else:
            # Fallback: return 20% from the top if no significant peak found
            return int(len(gradient_sum) * 0.2)

    def crop_sky_from_image(self, image: np.ndarray, offset: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Crop the sky from a building facade image.
        
        Args:
            image (np.ndarray): Input image (BGR or RGB)
            offset (int): Additional offset above the detected roof line
        
        Returns:
            Tuple[np.ndarray, int]: Cropped image and the crop line position
        """
        # Convert to grayscale if image is color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute gradient sum for each row
        gradient_sum = self.compute_horizontal_gradient_sum(gray)
        
        # Find roof line
        roof_line = self.find_roof_line(gradient_sum)
        
        # Apply offset (make sure we don't go out of bounds)
        if offset is None:
            offset = self.sky_offset
        crop_line = max(0, roof_line - offset)
        
        # Crop the image
        cropped_image = image[crop_line:, :]
        
        return cropped_image, crop_line

    def visualize_gradient_and_crop(self, image: np.ndarray, crop_line: int) -> np.ndarray:
        """
        Create a visualization of the gradient analysis and cropping result.
        
        Args:
            image (np.ndarray): Original image
            crop_line (int): The detected crop line
        
        Returns:
            np.ndarray: Visualization image
        """
        # Convert to grayscale if image is color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Compute gradient sum
        gradient_sum = self.compute_horizontal_gradient_sum(gray)
        
        # Normalize for visualization
        normalized = gradient_sum / np.max(gradient_sum) * image.shape[1] * 0.3
        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw gradient profile
        for i, mag in enumerate(normalized):
            cv2.line(vis_image, (0, i), (int(mag), i), (0, 255, 0), 1)
        
        # Draw crop line
        cv2.line(vis_image, (0, crop_line), (vis_image.shape[1], crop_line), (0, 0, 255), 2)
        
        return vis_image

    def process_and_display_image(self, image_path: str, offset: Optional[int] = None, 
                             show_visualization: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an image file, crop the sky, and optionally display results.
        
        Args:
            image_path (str): Path to the image file
            offset (int): Additional offset above the detected roof line
            show_visualization (bool): Whether to display visualization
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Original and cropped images
        """
        # Load image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Crop sky
        cropped, crop_line = self.crop_sky_from_image(original, offset)
        
        if show_visualization:
            # Create visualization
            vis_image = self.visualize_gradient_and_crop(original, crop_line)
            
            # Display results
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Gradient Analysis")
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Cropped Image")
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return original, cropped

    def batch_process_images(self, image_paths: list, output_dir: str, offset: Optional[int] = None) -> List[np.ndarray]:
        """
        Process multiple images and save the cropped results.
        
        Args:
            image_paths (list): List of paths to image files
            output_dir (str): Directory to save cropped images
            offset (int): Additional offset above the detected roof line
            
        Returns:
            List[np.ndarray]: List of cropped images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        processed_images = []
        
        for image_path in image_paths:
            try:
                # Load image
                original = cv2.imread(image_path)
                if original is None:
                    print(f"Could not load image from {image_path}")
                    continue
                
                # Crop sky
                cropped, _ = self.crop_sky_from_image(original, offset)
                
                # Save cropped image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"cropped_{filename}")
                cv2.imwrite(output_path, cropped)
                
                # Add to result list
                processed_images.append(cropped)
                
                print(f"Processed {filename} -> {output_path}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return processed_images
            
    def process_building_image(self, image: np.ndarray, output_dir: str, filename: str) -> np.ndarray:
        """
        Process a building image to remove the sky and return the processed image.
        This method is intended to be called from BuildingExtractor as the final step.
        
        Args:
            image (np.ndarray): Input image from BuildingProcessor
            output_dir (str): Directory to save the final image
            filename (str): Filename for the output image
            
        Returns:
            np.ndarray: Processed image without sky
        """
        try:
            # Crop sky from the image
            cropped, crop_line = self.crop_sky_from_image(image)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save final image
            output_path = os.path.join(output_dir, f"final_{filename}")
            cv2.imwrite(output_path, cropped)
            
            print(f"Saved final processed image to: {output_path}")
            return cropped
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return image  # Return original if processing fails
