import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os
import json

class HandwritingCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Font Generator")
        
        # Set up the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for drawing
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(self.main_frame, 
                              width=self.canvas_width, 
                              height=self.canvas_height,
                              bg='white',
                              bd=2,
                              relief='solid')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.current_character = 'A'
        self.collected_samples = {}
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Character selection
        ttk.Label(self.main_frame, text="Current Character:").grid(row=1, column=0, pady=5)
        self.char_var = tk.StringVar(value='A')
        self.char_entry = ttk.Entry(self.main_frame, textvariable=self.char_var, width=5)
        self.char_entry.grid(row=1, column=1, pady=5)
        
        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(self.button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Save", command=self.save_character).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Generate Font", command=self.generate_font).pack(side=tk.LEFT, padx=5)
        
        # Progress tracking
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=3, column=0, columnspan=2, pady=10)
        self.progress_var = tk.StringVar(value="Characters collected: 0/52")
        ttk.Label(self.progress_frame, textvariable=self.progress_var).pack()

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                     width=3, smooth=True, capstyle=tk.ROUND)
            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_character(self):
        # Get the current character
        char = self.char_var.get().upper()
        if not char or len(char) != 1:
            tk.messagebox.showerror("Error", "Please enter a single character")
            return

        # Create image from canvas
        image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Get canvas contents
        bbox = self.canvas.bbox('all')
        if bbox:
            # Scale and center the drawing
            x1, y1, x2, y2 = bbox
            padding = 20
            width = x2 - x1 + 2 * padding
            height = y2 - y1 + 2 * padding
            
            # Create a new image with the character centered
            char_image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
            draw = ImageDraw.Draw(char_image)
            
            # Copy canvas contents
            for item in self.canvas.find_all():
                coords = self.canvas.coords(item)
                draw.line(coords, fill='black', width=3)
            
            # Save processed image
            char_image = char_image.resize((64, 64), Image.LANCZOS)
            
            # Store in collected samples
            self.collected_samples[char] = np.array(char_image)
            
            # Update progress
            self.update_progress()
            
            # Clear canvas for next character
            self.clear_canvas()
            
            # Increment character
            next_char = chr(ord(char) + 1) if char < 'Z' else 'A'
            self.char_var.set(next_char)

    def update_progress(self):
        count = len(self.collected_samples)
        self.progress_var.set(f"Characters collected: {count}/52")

    def generate_font(self):
        if len(self.collected_samples) < 26:
            tk.messagebox.showwarning("Warning", "Please collect at least all uppercase letters before generating font")
            return
            
        # Create directory for saving samples
        os.makedirs('collected_samples', exist_ok=True)
        
        # Save collected samples
        for char, image_array in self.collected_samples.items():
            image = Image.fromarray(image_array)
            image.save(f'collected_samples/{char}.png')
        
        # Initialize and train the VAE model
        model = HandwritingVAE(latent_dim=128)
        train_data = list(self.collected_samples.values())
        train_labels = list(self.collected_samples.keys())
        
        dataset = HandwritingDataset(train_data, train_labels)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train the model
        train_model(model, train_loader, num_epochs=100, device=device)
        
        # Save the trained model
        torch.save(model.state_dict(), 'handwriting_font_model.pth')
        
        # Generate font samples
        font_generator = FontGenerator('handwriting_font_model.pth')
        os.makedirs('generated_font', exist_ok=True)
        
        for char in self.collected_samples.keys():
            # Generate character
            z = torch.randn(1, 128).to(device)
            generated_char = font_generator.generate_character(z)
            
            # Save generated character
            char_image = Image.fromarray((generated_char[0, 0] * 255).astype(np.uint8))
            char_image.save(f'generated_font/{char}.png')
        
        tk.messagebox.showinfo("Success", "Font generated successfully! Check the 'generated_font' folder.")

# Main application class
class FontGeneratorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.app = HandwritingCanvas(self.root)
        
    def run(self):
        self.root.mainloop()

# Run the application
if __name__ == "__main__":
    app = FontGeneratorApp()
    app.run()
