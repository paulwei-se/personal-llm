import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from document_processing.processor import DocumentProcessor

def test_document_processor():
    processor = DocumentProcessor()
    
    # Process the sample document
    sample_doc_path = os.path.join(os.path.dirname(__file__), 'sample_documents', 'sample_doc2.png')
    result = processor.process_document(sample_doc_path)
    
    # Print the extracted text
    print("Extracted Text:")
    print(result['text'])
    
    # Print the layout analysis
    print("\nLayout Analysis:")
    for item in result['layout']:
        print(f"Type: {item['type']}, Position: ({item['x']}, {item['y']}), Size: {item['width']}x{item['height']}")

if __name__ == "__main__":
    test_document_processor()
