"""
PDF Processor: Extracts text from research paper PDFs.
Handles various PDF formats and preserves document structure.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

import fitz  # PyMuPDF
from config import UPLOADS_DIR, PROCESSED_DIR


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    doc_id: str
    filename: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    num_pages: int = 0
    file_path: str = ""
    
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
    def from_dict(cls, data: Dict) -> "DocumentMetadata":
        return cls(**data)


@dataclass
class ProcessedDocument:
    """A processed document with extracted text and metadata."""
    metadata: DocumentMetadata
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    

class PDFProcessor:
    """
    Extracts text and metadata from research paper PDFs.
    """
    
    def __init__(self):
        self.section_patterns = [
            r"^(abstract|introduction|background|related work|methodology|"
            r"methods|results|discussion|conclusion|references|acknowledgments)",
            r"^\d+\.?\s+(introduction|background|methodology|results|discussion|conclusion)",
        ]
    
    def generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file content hash."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        return file_hash
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, num_pages)
        """
        doc = fitz.open(pdf_path)
        full_text = []
        num_pages = len(doc)  # Get page count before closing
        
        for page_num, page in enumerate(doc):
            # Extract text with better formatting
            text = page.get_text("text")
            
            # Clean up the text
            text = self._clean_text(text)
            
            if text.strip():
                full_text.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(full_text), num_pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers that appear alone
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Remove header/footer artifacts (common patterns)
        text = re.sub(r'^.{0,50}(arXiv|preprint|submitted|accepted).{0,50}$', '', 
                     text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()
    
    def extract_metadata(self, text: str, pdf_path: str) -> DocumentMetadata:
        """
        Extract metadata from the document text.
        
        Args:
            text: Full document text
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentMetadata object
        """
        filename = os.path.basename(pdf_path)
        doc_id = self.generate_doc_id(pdf_path)
        
        # Try to extract title (usually first significant line)
        title = self._extract_title(text)
        
        # Try to extract abstract
        abstract = self._extract_abstract(text)
        
        # Try to extract authors (simplified)
        authors = self._extract_authors(text)
        
        return DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            title=title,
            authors=authors,
            abstract=abstract,
            file_path=pdf_path
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract the paper title from text."""
        lines = text.split('\n')
        
        # Skip page marker and find first substantial line
        for line in lines:
            line = line.strip()
            if line.startswith('[Page'):
                continue
            # Title is usually longer than 10 chars and doesn't start with common metadata
            if len(line) > 10 and not any(
                line.lower().startswith(x) for x in 
                ['abstract', 'arxiv', 'http', 'doi:', 'keywords', 'author']
            ):
                # Title usually doesn't have too many periods or special chars
                if line.count('.') <= 1 and not re.match(r'^\d', line):
                    return line[:200]  # Limit title length
        
        return "Untitled Document"
    
    def _extract_abstract(self, text: str) -> str:
        """Extract the abstract from text."""
        # Look for abstract section
        abstract_match = re.search(
            r'abstract[:\s]*\n(.*?)(?=\n\s*(?:introduction|keywords|\d+\.?\s*introduction|1\s))',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Clean up and limit length
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:2000]
        
        return ""
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names (simplified extraction)."""
        # This is a simplified extraction - real papers vary greatly in format
        lines = text.split('\n')[:20]  # Authors usually in first 20 lines
        
        authors = []
        for line in lines:
            # Look for lines that might contain author names
            # Usually contains email or affiliation markers
            if '@' in line or any(x in line.lower() for x in ['university', 'institute', 'department']):
                continue
            # Check for comma-separated names pattern
            if ',' in line and len(line) < 200:
                potential_names = [n.strip() for n in line.split(',')]
                for name in potential_names:
                    # Basic name pattern (2-4 words, capitalized)
                    if re.match(r'^[A-Z][a-z]+(\s+[A-Z]\.?|\s+[A-Z][a-z]+){1,3}$', name):
                        authors.append(name)
        
        return authors[:10]  # Limit to 10 authors
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract document sections.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        current_section = "preamble"
        current_content = []
        
        for line in text.split('\n'):
            # Check if this line is a section header
            is_section = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.lower().strip())
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = match.group(1) if match.lastindex else line.strip().lower()
                    current_content = []
                    is_section = True
                    break
            
            if not is_section:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a PDF file and extract all information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument object with all extracted information
        """
        # Extract text
        full_text, num_pages = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(full_text, pdf_path)
        metadata.num_pages = num_pages
        
        # Extract sections
        sections = self.extract_sections(full_text)
        
        return ProcessedDocument(
            metadata=metadata,
            full_text=full_text,
            sections=sections
        )
    
    def process_directory(self, directory: str = None) -> List[ProcessedDocument]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Path to directory (defaults to UPLOADS_DIR)
            
        Returns:
            List of ProcessedDocument objects
        """
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
    # Test the PDF processor
    processor = PDFProcessor()
    documents = processor.process_directory()
    
    for doc in documents:
        print(f"\nDocument: {doc.metadata.title}")
        print(f"  ID: {doc.metadata.doc_id}")
        print(f"  Pages: {doc.metadata.num_pages}")
        print(f"  Sections: {list(doc.sections.keys())}")
