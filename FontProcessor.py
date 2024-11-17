import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
from pathlib import Path
import json

class FontProcessor:
    def __init__(self):
        self.output_size = (64, 64)
        self.templates_path = Path('templates')
        self.output_path = Path('generated_font')
        
    def preprocess_image(self, image):
        """Preprocess a character image for the model."""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
            
        # Resize to standard size
        image = image.resize(self.output_size, Image.LANCZOS)
        
        # Normalize pixel values
        image_array = np.array(image) / 255.0
        
        return image_array

    def postprocess_image(self, array):
        """Convert model output back to image."""
        # Scale back to 0-255 range
        array = (array * 255).astype(np.uint8)
        
        # Create PIL Image
        image = Image.fromarray(array)
        
        return image

class FontExporter:
    def __init__(self):
        self.char_size = 64
        self.padding = 8
        self.chars_per_row = 13
        
    def create_font_sheet(self, generated_chars):
        """Create a sheet with all generated characters."""
        num_chars = len(generated_chars)
        num_rows = (num_chars + self.chars_per_row - 1) // self.chars_per_row
        
        sheet_width = self.chars_per_row * (self.char_size + self.padding)
        sheet_height = num_rows * (self.char_size + self.padding)
        
        sheet = Image.new('L', (sheet_width, sheet_height), 'white')
        draw = ImageDraw.Draw(sheet)
        
        for idx, (char, img) in enumerate(generated_chars.items()):
            row = idx // self.chars_per_row
            col = idx % self.chars_per_row
            
            x = col * (self.char_size + self.padding)
            y = row * (self.char_size + self.padding)
            
            sheet.paste(img, (x, y))
            
            # Draw character label
            draw.text((x, y + self.char_size - 12), char, fill='gray')
            
        return sheet

    def export_font_data(self, generated_chars, metadata):
        """Export font data and metadata."""
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save individual characters
        char_data = {}
        for char, img in generated_chars.items():
            filename = f"{ord(char):04x}.png"
            img.save(self.output_path / filename)
            char_data[char] = {
                'unicode': ord(char),
                'file': filename
            }
        
        # Save metadata
        metadata.update({
            'char_size': self.char_size,
            'num_chars': len(generated_chars),
            'characters': char_data
        })
        
        with open(self.output_path / 'font_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create and save font sheet
        sheet = self.create_font_sheet(generated_chars)
        sheet.save(self.output_path / 'font_sheet.png')

class QualityChecker:
    def __init__(self):
        self.min_stroke_width = 2
        self.max_stroke_width = 10
        self.min_coverage = 0.1
        self.max_coverage = 0.4
        
    def check_image_quality(self, image):
        """Check if a character image meets quality standards."""
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Calculate image statistics
        coverage = np.mean(image < 128)  # For dark strokes on light background
        
        # Perform quality checks
        issues = []
        
        if coverage < self.min_coverage:
            issues.append("Stroke too light or missing")
        elif coverage > self.max_coverage:
            issues.append("Stroke too heavy or image too dark")
            
        # Check stroke consistency
        if len(issues) == 0:
            return True, None
        else:
            return False, issues

def generate_sample_text(font_path, text, size=32):
    """Generate sample text using the created font."""
    # Create image with white background
    width = len(text) * size
    height = int(size * 1.5)
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(str(font_path), size)
        draw.text((10, 10), text, font=font, fill='black')
        return image
    except Exception as e:
        print(f"Error generating sample text: {e}")
        return None

# Helper functions for font generation
def process_collected_samples(samples_dir):
    """Process collected handwriting samples."""
    processor = FontProcessor()
    samples = {}
    
    for file in os.listdir(samples_dir):
        if file.endswith('.png'):
            char = file[0].upper()
            image = Image.open(os.path.join(samples_dir, file))
            samples[char] = processor.preprocess_image(image)
            
    return samples

def generate_font_package(generated_chars, author_name, font_name):
    """Create a complete font package with metadata."""
    exporter = FontExporter()
    metadata = {
        'name': font_name,
        'author': author_name,
        'creation_date': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    exporter.export_font_data(generated_chars, metadata)
