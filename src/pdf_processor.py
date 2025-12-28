# PDF processing module for extracting text and metadata from research papers

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

import fitz
from config import UPLOADS_DIR, PROCESSED_DIR

@dataclass
class DocumentMetadata:
    doc_id: str
    filename: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    num_pages: int = 0
    file_path: str = ""
    
    # Converts object to dictionary representation
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "num_pages": self.num_pages,
            "file_path": self.file_path
        }
    
    @classmethod
    # Creates object instance from dictionary data
    def from_dict(cls, data: Dict) -> "DocumentMetadata":
        return cls(**data)

@dataclass
class ProcessedDocument:
    metadata: DocumentMetadata
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    

class PDFProcessor:
    
    # Initializes the class with configuration parameters
    def __init__(self):
        self.section_patterns = [
            r"^(abstract|introduction|background|related work|methodology|"
            r"methods|results|discussion|conclusion|references|acknowledgments)",
            r"^\d+\.?\s+(introduction|background|methodology|results|discussion|conclusion)",
        ]
    
    # Generates output based on input
    def generate_doc_id(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        return file_hash
    
    # Extract Text From Pdf
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        doc = fitz.open(pdf_path)
        full_text = []
        num_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            text = self._clean_text(text)
            
            if text.strip():
                full_text.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(full_text), num_pages
    
    #  Clean Text
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        text = re.sub(r'^.{0,50}(arXiv|preprint|submitted|accepted).{0,50}$', '', 
                     text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()
    
    # Extract Metadata
    def extract_metadata(self, text: str, pdf_path: str) -> DocumentMetadata:
        filename = os.path.basename(pdf_path)
        doc_id = self.generate_doc_id(pdf_path)
        
        title = self._extract_title(text)
        
        abstract = self._extract_abstract(text)
        
        authors = self._extract_authors(text)
        
        return DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            title=title,
            authors=authors,
            abstract=abstract,
            file_path=pdf_path
        )
    
    #  Extract Title
    def _extract_title(self, text: str) -> str:
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('[Page'):
                continue
            if len(line) > 10 and not any(
                line.lower().startswith(x) for x in 
                ['abstract', 'arxiv', 'http', 'doi:', 'keywords', 'author']
            ):
                if line.count('.') <= 1 and not re.match(r'^\d', line):
                    return line[:200]
        
        return "Untitled Document"
    
    #  Extract Abstract
    def _extract_abstract(self, text: str) -> str:
        abstract_match = re.search(
            r'abstract[:\s]*\n(.*?)(?=\n\s*(?:introduction|keywords|\d+\.?\s*introduction|1\s))',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:2000]
        
        return ""
    
    #  Extract Authors
    def _extract_authors(self, text: str) -> List[str]:
        lines = text.split('\n')[:20]
        
        authors = []
        for line in lines:
            if '@' in line or any(x in line.lower() for x in ['university', 'institute', 'department']):
                continue
            if ',' in line and len(line) < 200:
                potential_names = [n.strip() for n in line.split(',')]
                for name in potential_names:
                    if re.match(r'^[A-Z][a-z]+(\s+[A-Z]\.?|\s+[A-Z][a-z]+){1,3}$', name):
                        authors.append(name)
        
        return authors[:10]
    
    # Extract Sections
    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        current_section = "preamble"
        current_content = []
        
        for line in text.split('\n'):
            is_section = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.lower().strip())
                if match:
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = match.group(1) if match.lastindex else line.strip().lower()
                    current_content = []
                    is_section = True
                    break
            
            if not is_section:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    # Processes the input data
    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        full_text, num_pages = self.extract_text_from_pdf(pdf_path)
        
        metadata = self.extract_metadata(full_text, pdf_path)
        metadata.num_pages = num_pages
        
        sections = self.extract_sections(full_text)
        
        return ProcessedDocument(
            metadata=metadata,
            full_text=full_text,
            sections=sections
        )
    
    # Processes the input data
    def process_directory(self, directory: str = None) -> List[ProcessedDocument]:
        if directory is None:
            directory = UPLOADS_DIR
        
        documents = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                print(f"Processing: {pdf_path.name}")
                doc = self.process_pdf(str(pdf_path))
                documents.append(doc)
                print(f"  ✓ Extracted {len(doc.full_text)} characters, {doc.metadata.num_pages} pages")
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path.name}: {e}")
        
        return documents

if __name__ == "__main__":
    processor = PDFProcessor()
    documents = processor.process_directory()
    
    for doc in documents:
        print(f"\nDocument: {doc.metadata.title}")
        print(f"  ID: {doc.metadata.doc_id}")
        print(f"  Pages: {doc.metadata.num_pages}")
        print(f"  Sections: {list(doc.sections.keys())}")
