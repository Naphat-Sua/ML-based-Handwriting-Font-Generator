from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._g_l_y_f import Glyph
import bezier
import cv2
from scipy.spatial import ConvexHull
import logging
import warnings
from datetime import datetime

class FontStyleController:
    def __init__(self):
        self.style_params = {
            'weight': 400,  # Font weight (100-900)
            'width': 100,   # Font width percentage
            'slant': 0,     # Slant angle in degrees
            'spacing': 0,   # Letter spacing adjustment
            'baseline': 0   # Baseline shift
        }
        
    def adjust_weight(self, image, weight_factor):
        """Adjust the weight (thickness) of character strokes."""
        kernel_size = int(abs(weight_factor - 400) / 100)
        if kernel_size == 0:
            return image
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if weight_factor > 400:
            return cv2.dilate(image, kernel, iterations=1)
        else:
            return cv2.erode(image, kernel, iterations=1)
    
    def adjust_width(self, image, width_factor):
        """Adjust the width of characters."""
        if width_factor == 100:
            return image
            
        height, width = image.shape
        new_width = int(width * (width_factor / 100))
        return cv2.resize(image, (new_width, height))
    
    def adjust_slant(self, image, angle):
        """Apply slant to characters."""
        if angle == 0:
            return image
            
        height, width = image.shape
        tan_angle = np.tan(np.radians(angle))
        shift_matrix = np.float32([[1, tan_angle, 0], [0, 1, 0]])
        return cv2.warpAffine(image, shift_matrix, (width, height))
    
    def apply_style(self, image):
        """Apply all style adjustments to an image."""
        result = image.copy()
        result = self.adjust_weight(result, self.style_params['weight'])
        result = self.adjust_width(result, self.style_params['width'])
        result = self.adjust_slant(result, self.style_params['slant'])
        return result

class TTFExporter:
    def __init__(self):
        self.font = TTFont()
        self.em_size = 1000
        self.ascent = 800
        self.descent = -200
        
    def setup_font_tables(self, font_name):
        """Initialize basic font tables."""
        # Setup required tables
        self.font.setupGlyphOrder(['.notdef'])
        self.font.setupHead(fontRevision=1.0, flags=[])
        self.font.setupMaxp()
        self.font.setupPost()
        self.font.setupOS2(usWeightClass=400)
        self.font.setupName(
            familyName=font_name,
            styleName='Regular',
            uniqueFontIdentifier=f'{font_name}-Regular',
            fullName=f'{font_name} Regular',
            version='Version 1.0',
            psName=f'{font_name}-Regular'
        )
        
    def convert_contours(self, image):
        """Convert image to font contours."""
        # Find contours in the image
        contours, _ = cv2.findContours(
            image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to TTF format
        pen = TTGlyphPen(None)
        for contour in contours:
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to bezier curves
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue
                
            # Create bezier curves
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Move to first point
            pen.moveTo(hull_points[0])
            
            # Add curves
            for i in range(1, len(hull_points)):
                pen.lineTo(hull_points[i])
            
            # Close contour
            pen.closePath()
            
        return pen.glyph()
    
    def add_character(self, char, image):
        """Add a character to the font."""
        # Convert image to proper format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Create glyph
        glyph = self.convert_contours(image)
        
        # Add to font
        glyph_name = f'uni{ord(char):04X}'
        self.font['glyf'][glyph_name] = glyph
        self.font['cmap'].tables[0].cmap[ord(char)] = glyph_name
        
    def save_font(self, output_path):
        """Save the font to a TTF file."""
        try:
            self.font.save(output_path)
            return True
        except Exception as e:
            logging.error(f"Error saving font: {e}")
            return False

class FontMetadataManager:
    def __init__(self):
        self.metadata = {
            'creation_date': None,
            'modification_date': None,
            'version': '1.0',
            'style_settings': {},
            'character_metrics': {},
            'license': 'OFL',
            'author': '',
            'description': ''
        }
        
    def update_metadata(self, **kwargs):
        """Update font metadata."""
        for key, value in kwargs.items():
            if key in self.metadata:
                self.metadata[key] = value
        
        self.metadata['modification_date'] = datetime.now().isoformat()
        
    def add_character_metrics(self, char, metrics):
        """Add metrics for a specific character."""
        self.metadata['character_metrics'][char] = metrics
        
    def save_metadata(self, output_path):
        """Save metadata to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def load_metadata(self, input_path):
        """Load metadata from JSON file."""
        try:
            with open(input_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")

class QualityController:
    def __init__(self):
        self.quality_checks = {
            'min_size': 20,
            'max_size': 800,
            'min_contrast': 0.3,
            'min_strokes': 1,
            'max_complexity': 100
        }
        
    def analyze_character(self, image):
        """Analyze character quality and metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['size'] = image.shape
        metrics['mean_value'] = np.mean(image)
        metrics['std_value'] = np.std(image)
        
        # Stroke analysis
        edges = cv2.Canny(image, 100, 200)
        metrics['stroke_count'] = len(cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
        # Complexity analysis
        metrics['complexity'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        return metrics
    
    def check_quality(self, image, metrics):
        """Check if character meets quality standards."""
        issues = []
        
        if metrics['size'][0] < self.quality_checks['min_size']:
            issues.append("Character too small")
        if metrics['size'][0] > self.quality_checks['max_size']:
            issues.append("Character too large")
        if metrics['std_value'] < self.quality_checks['min_contrast']:
            issues.append("Insufficient contrast")
        if metrics['stroke_count'] < self.quality_checks['min_strokes']:
            issues.append("No clear strokes detected")
        if metrics['complexity'] > self.quality_checks['max_complexity']:
            issues.append("Character too complex")
            
        return len(issues) == 0, issues

def main():
    # Initialize components
    style_controller = FontStyleController()
    ttf_exporter = TTFExporter()
    metadata_manager = FontMetadataManager()
    quality_controller = QualityController()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create font processing pipeline
        def process_character(char, image):
            # Apply style
            styled_image = style_controller.apply_style(image)
            
            # Check quality
            metrics = quality_controller.analyze_character(styled_image)
            quality_ok, issues = quality_controller.check_quality(styled_image, metrics)
            
            if not quality_ok:
                raise ValueError(f"Quality issues detected: {', '.join(issues)}")
            
            # Add to font
            ttf_exporter.add_character(char, styled_image)
            
            # Update metadata
            metadata_manager.add_character_metrics(char, metrics)
            
            return styled_image
        
        # Process all characters
        def generate_font(characters, font_name, author):
            # Initialize font
            ttf_exporter.setup_font_tables(font_name)
            
            # Update metadata
            metadata_manager.update_metadata(
                author=author,
                creation_date=datetime.now().isoformat(),
                style_settings=style_controller.style_params
            )
            
            # Process each character
            processed_chars = {}
            for char, image in characters.items():
                try:
                    processed_chars[char] = process_character(char, image)
                    logging.info(f"Processed character: {char}")
                except Exception as e:
                    logging.error(f"Error processing character {char}: {e}")
            
            # Save font files
            output_dir = Path(f'output/{font_name}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ttf_path = output_dir / f'{font_name}.ttf'
            ttf_exporter.save_font(ttf_path)
            
            metadata_path = output_dir / f'{font_name}_metadata.json'
            metadata_manager.save_metadata(metadata_path)
            
            return ttf_path, metadata_path
            
        return generate_font
        
    except Exception as e:
        logging.error(f"Error initializing font generation system: {e}")
        return None

if __name__ == "__main__":
    # Get font generator
    font_generator = main()
    
    if font_generator:
        logging.info("Font generation system initialized successfully")
    else:
        logging.error("Failed to initialize font generation system")
