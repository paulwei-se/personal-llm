import cv2
import pytesseract
from PIL import Image
import numpy as np
import PyPDF2
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_document(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Ensure Tesseract is in your PATH or specify the path here
        # pytesseract.pytesseract.tesseract_cmd = r'path/to/tesseract'

    def extract_text(self, file_path):
        """Extract text from an image or PDF file."""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            return self.extract_text_from_image(file_path)

    def extract_text_from_image(self, image_path):
        """Extract text from an image using Tesseract OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def analyze_layout(self, file_path):
        """Perform basic layout analysis on an image or first page of a PDF."""
        try:
            if file_path.lower().endswith('.pdf'):
                # For PDFs, we'll analyze the first page only
                image = self.pdf_to_image(file_path)
                if image is None:
                    self.logger.warning(f"Could not extract image from PDF: {file_path}")
                    return []
            else:
                image = cv2.imread(file_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        except Exception as e:
            self.logger.error(f"Error in analyze_layout for {file_path}: {str(e)}")
            return []

    def pdf_to_image(self, pdf_path):
        """Convert the first page of a PDF to an image."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page = reader.pages[0]
                
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                            data = xObject[obj].get_data()
                            mode = 'RGB' if xObject[obj]['/ColorSpace'] == '/DeviceRGB' else 'P'
                            img = Image.frombytes(mode, size, data)
                            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # If no image found, return None
                self.logger.warning(f"No image found in the first page of PDF: {pdf_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error in pdf_to_image for {pdf_path}: {str(e)}")
            return None

    def process_document(self, file_path):
        """Process a document file, extracting text and analyzing layout."""
        try:
            text = self.extract_text(file_path)
            layout = self.analyze_layout(file_path)
            
            return {
                'text': text,
                'layout': layout
            }
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'text': '',
                'layout': []
            }

# Usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    result = processor.process_document("path/to/your/document/file.pdf")
    print(result)