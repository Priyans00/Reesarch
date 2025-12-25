"""
Text Chunker: Splits documents into semantically meaningful chunks.
Optimized for research papers with section-aware chunking.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import uuid

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class TextChunk:
    """A chunk of text from a document."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TextChunk":
        return cls(**data)


class TextChunker:
    """
    Splits documents into chunks suitable for embedding and retrieval.
    Uses sentence-aware splitting to preserve semantic coherence.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk (in words, approximately)
            chunk_overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum chunk size in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Sentence splitting pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First, normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens (roughly words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)
    
    def _create_chunk(
        self,
        sentences: List[str],
        doc_id: str,
        section: str = "",
        chunk_index: int = 0
    ) -> TextChunk:
        """Create a TextChunk from a list of sentences."""
        text = ' '.join(sentences)
        chunk_id = f"{doc_id}_{chunk_index}_{uuid.uuid4().hex[:8]}"
        
        metadata = {
            "section": section,
            "chunk_index": chunk_index,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
        
        return TextChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            metadata=metadata
        )
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str = ""
    ) -> List[TextChunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            doc_id: Document ID for tracking
            section: Section name for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            # If no sentences found, treat whole text as one chunk
            if len(text) >= self.min_chunk_size:
                return [self._create_chunk([text], doc_id, section, 0)]
            return []
        
        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunk = self._create_chunk(
                    current_sentences, doc_id, section, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Keep overlap sentences
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_sentences:
            text_content = ' '.join(current_sentences)
            if len(text_content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_sentences, doc_id, section, chunk_index
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_document(
        self,
        full_text: str,
        doc_id: str,
        sections: Optional[Dict[str, str]] = None,
        title: str = "",
        abstract: str = ""
    ) -> List[TextChunk]:
        """
        Chunk a full document, optionally using section information.
        
        Args:
            full_text: Complete document text
            doc_id: Document ID
            sections: Optional dictionary of section name -> section text
            title: Document title (prepended to chunks for context)
            abstract: Document abstract
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        # If we have sections, chunk each section separately
        if sections:
            # Add title context to first chunk of each section
            title_context = f"Paper: {title}\n\n" if title else ""
            
            for section_name, section_text in sections.items():
                # Skip very short sections or reference sections
                if len(section_text) < self.min_chunk_size:
                    continue
                if section_name.lower() in ['references', 'bibliography', 'acknowledgments']:
                    continue
                
                # Add section header to text
                section_text_with_header = f"[{section_name.upper()}]\n{section_text}"
                
                chunks = self.chunk_text(
                    section_text_with_header,
                    doc_id,
                    section=section_name
                )
                
                # Add title context to first chunk of section
                if chunks and title_context:
                    chunks[0].text = title_context + chunks[0].text
                    chunks[0].metadata["has_title_context"] = True
                
                all_chunks.extend(chunks)
        else:
            # Chunk the full text without section awareness
            all_chunks = self.chunk_text(full_text, doc_id, section="full_document")
            
            # Add title to first chunk
            if all_chunks and title:
                all_chunks[0].text = f"Paper: {title}\n\n{all_chunks[0].text}"
                all_chunks[0].metadata["has_title_context"] = True
        
        # Add abstract as a separate chunk if available
        if abstract and len(abstract) >= self.min_chunk_size:
            abstract_chunk = TextChunk(
                chunk_id=f"{doc_id}_abstract_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=f"Paper: {title}\n\n[ABSTRACT]\n{abstract}" if title else f"[ABSTRACT]\n{abstract}",
                metadata={
                    "section": "abstract",
                    "chunk_index": -1,
                    "is_abstract": True,
                    "word_count": len(abstract.split()),
                    "char_count": len(abstract)
                }
            )
            all_chunks.insert(0, abstract_chunk)
        
        return all_chunks
    
    def chunk_documents(
        self,
        documents: List
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of ProcessedDocument objects
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(
                full_text=doc.full_text,
                doc_id=doc.metadata.doc_id,
                sections=doc.sections,
                title=doc.metadata.title,
                abstract=doc.metadata.abstract
            )
            all_chunks.extend(chunks)
            print(f"Created {len(chunks)} chunks from: {doc.metadata.filename}")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


class SemanticChunker(TextChunker):
    """
    Advanced chunker that uses semantic boundaries.
    Tries to keep related content together.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Patterns that indicate semantic boundaries
        self.boundary_patterns = [
            r'\n\n+',  # Paragraph breaks
            r'\n\s*[-•]\s+',  # Bullet points
            r'\n\s*\d+\.\s+',  # Numbered lists
            r'\n\s*[A-Z][^.]*:\s*\n',  # Section-like headers
        ]
    
    def _find_best_split_point(self, text: str, target_pos: int, window: int = 200) -> int:
        """
        Find the best point to split text near target_pos.
        Prefers semantic boundaries like paragraph breaks.
        """
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_text = text[start:end]
        
        # Look for paragraph break
        para_match = re.search(r'\n\n+', search_text)
        if para_match:
            return start + para_match.end()
        
        # Look for sentence end
        sent_match = re.search(r'[.!?]\s+', search_text)
        if sent_match:
            return start + sent_match.end()
        
        # Fall back to target position
        return target_pos
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str = ""
    ) -> List[TextChunk]:
        """
        Semantically-aware text chunking.
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # First try paragraph-based splitting
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return super().chunk_text(text, doc_id, section)
        
        chunks = []
        current_paragraphs = []
        current_chars = 0
        chunk_index = 0
        
        # Target chunk size in characters (roughly chunk_size * 5 chars per token)
        target_chars = self.chunk_size * 5
        overlap_chars = self.chunk_overlap * 5
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_chars + para_len > target_chars and current_paragraphs:
                # Create chunk
                chunk_text = '\n\n'.join(current_paragraphs)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        [chunk_text], doc_id, section, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Calculate overlap
                overlap_text = ""
                for p in reversed(current_paragraphs):
                    if len(overlap_text) + len(p) <= overlap_chars:
                        overlap_text = p + '\n\n' + overlap_text if overlap_text else p
                    else:
                        break
                
                current_paragraphs = [overlap_text] if overlap_text else []
                current_chars = len(overlap_text)
            
            current_paragraphs.append(para)
            current_chars += para_len
        
        # Last chunk
        if current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    [chunk_text], doc_id, section, chunk_index
                )
                chunks.append(chunk)
        
        return chunks


if __name__ == "__main__":
    # Test the chunker
    test_text = """
    This is the first paragraph of the document. It contains some introductory 
    information about the topic at hand. We will explore various aspects in detail.
    
    The second paragraph delves deeper into the methodology. We used a novel approach
    that combines multiple techniques. The results were quite promising and showed
    significant improvements over baseline methods.
    
    In our experiments, we evaluated the system on several benchmarks. The metrics
    we used include precision, recall, and F1 score. Our system achieved state-of-the-art
    performance on most benchmarks.
    
    The final paragraph discusses future work and conclusions. There are many avenues
    for further research. We believe this work opens up new possibilities in the field.
    """
    
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(test_text, doc_id="test_doc", section="test")
    
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.metadata['chunk_index']} ---")
        print(f"Length: {len(chunk.text)} chars")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
