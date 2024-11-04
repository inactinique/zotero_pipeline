import io
import logging
import requests
import PyPDF2
from pyzotero import zotero
from typing import Dict, Tuple

class ZoteroPDFProcessor:
    def __init__(self, library_id: str, library_type: str, api_key: str):
        """
        Initialize the ZoteroPDFProcessor with Zotero credentials
        
        Args:
            library_id (str): Your Zotero library ID
            library_type (str): Either 'user' or 'group'
            api_key (str): Your Zotero API key
        """
        self.zot = zotero.Zotero(library_id, library_type, api_key)
        self.logger = logging.getLogger(__name__)
        
        # Set up logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def process_collection(self, collection_name: str) -> list:
        """
        Process all items in a given collection
        
        Args:
            collection_name (str): Name of the collection to process
            
        Returns:
            list: List of tuples containing (text, metadata) for each processed PDF
        """
        try:
            # Get all collections
            collections = self.zot.collections()
            
            # Find the target collection
            target_collection = None
            for collection in collections:
                if collection['data']['name'] == collection_name:
                    target_collection = collection
                    break
            
            if not target_collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            # Get items in the collection
            items = self.zot.collection_items(target_collection['key'])
            
            # Process each item
            results = []
            for item in items:
                text, metadata = self.extract_pdf_text(item)
                if text:  # Only add if text was successfully extracted
                    results.append((text, metadata))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing collection {collection_name}: {str(e)}")
            return []

    def extract_pdf_text(self, item: Dict) -> Tuple[str, Dict]:
        """
        Download and extract text from a PDF file
        Returns the text and associated metadata
        """
        try:
            # Get PDF attachment
            attachments = self.zot.children(item['key'])
            
            # Debug information
            print(f"Processing item: {item['data'].get('title', 'Unknown')}")
            print(f"Found {len(attachments)} attachments")
            
            # Find PDF attachment
            pdf_attachment = None
            for att in attachments:
                print(f"Attachment type: {att['data'].get('contentType', 'Unknown')}")
                if att['data'].get('contentType') == 'application/pdf':
                    pdf_attachment = att
                    break
            
            if not pdf_attachment:
                self.logger.warning(f"No PDF found for item: {item['data'].get('title', 'Unknown')}")
                return "", {}
            
            try:
                # Try to get the file using multiple methods
                pdf_content = None
                
                # Method 1: Try using attachment_simple (most reliable method)
                try:
                    attachment_link = self.zot.attachment_simple(pdf_attachment['key'])
                    response = requests.get(attachment_link)
                    if response.status_code == 200:
                        pdf_content = response.content
                except Exception as e1:
                    print(f"Method 1 failed: {str(e1)}")
                    
                    # Method 2: Try direct file access
                    try:
                        pdf_content = self.zot.file(pdf_attachment['key'])
                    except Exception as e2:
                        print(f"Method 2 failed: {str(e2)}")
                        
                        # Method 3: Try to construct download URL manually
                        try:
                            download_url = f"https://api.zotero.org/{self.zot.library_type}/{self.zot.library_id}/items/{pdf_attachment['key']}/file"
                            headers = {'Authorization': f'Bearer {self.zot.api_key}'}
                            response = requests.get(download_url, headers=headers)
                            if response.status_code == 200:
                                pdf_content = response.content
                        except Exception as e3:
                            print(f"Method 3 failed: {str(e3)}")
                
                if pdf_content is None:
                    raise Exception("Could not download PDF content using any method")
                
                # Convert bytes to BytesIO object
                pdf_file = io.BytesIO(pdf_content)
                
            except Exception as e:
                self.logger.error(f"Error downloading PDF: {str(e)}")
                return "", {}
            
            # Extract text
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                text = " ".join(page.extract_text() for page in reader.pages)
            except Exception as e:
                self.logger.error(f"Error extracting text from PDF: {str(e)}")
                return "", {}
            
            # Get metadata
            metadata = {
                'title': item['data'].get('title', ''),
                'creators': item['data'].get('creators', []),
                'date': item['data'].get('date', ''),
                'tags': item['data'].get('tags', []),
                'itemKey': item['key']
            }
            
            return text, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {item.get('data', {}).get('title', 'Unknown')}: {str(e)}")
            return "", {}