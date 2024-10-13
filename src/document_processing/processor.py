import cv2
import pytesseract
from PIL import Image
import numpy as np

class DocumentProcessor:
    def __init__(self):
        # Ensure Tesseract is in your PATH or specify the path here
        # pytesseract.pytesseract.tesseract_cmd = r'path/to/tesseract'
        pass

    def extract_text(self, image_path):
        """Extract text from an image using Tesseract OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

    def analyze_layout(self, image_path):
        """Perform basic layout analysis using OpenCV."""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find contours in the image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze contours to identify potential text blocks, images, etc.
        layout = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 1000:  # Adjust this threshold as needed
                layout.append({
                    'type': 'text' if w > h else 'image',  # Simple heuristic
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })

        return layout

    def process_document(self, image_path):
        """Process a document image, extracting text and analyzing layout."""
        text = self.extract_text(image_path)
        layout = self.analyze_layout(image_path)
        
        return {
            'text': text,
            'layout': layout
        }

# Usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    result = processor.process_document("path/to/your/document/image.jpg")
    print(result)
