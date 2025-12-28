# Text chunking module for splitting documents into semantic chunks

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import uuid

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    
    # Converts object to dictionary representation
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    # Creates object instance from dictionary data
    @classmethod
    def from_dict(cls, data: Dict) -> "TextChunk":
        return cls(**data)

class TextChunker:
    
    # Initializes the class with configuration parameters
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
        )
    
    # Splits text into sentences using regex pattern
    def _split_into_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text)
        
        sentences = self.sentence_pattern.split(text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    # Estimates token count from text word count
    def _estimate_tokens(self, text: str) -> int:
        words = len(text.split())
        return int(words * 1.3)
    
    # Creates TextChunk object from list of sentences with metadata
    def _create_chunk(
        self,
        sentences: List[str],
        doc_id: str,
        section: str = "",
        chunk_index: int = 0
    ) -> TextChunk:
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
    
    # Splits text into overlapping chunks preserving sentence boundaries
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str = ""
    ) -> List[TextChunk]:
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            if len(text) >= self.min_chunk_size:
                return [self._create_chunk([text], doc_id, section, 0)]
            return []
        
        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunk = self._create_chunk(
                    current_sentences, doc_id, section, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
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
        
        if current_sentences:
            text_content = ' '.join(current_sentences)
            if len(text_content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_sentences, doc_id, section, chunk_index
                )
                chunks.append(chunk)
        
        return chunks
    
    # Chunks entire document with optional section-aware processing
    def chunk_document(
        self,
        full_text: str,
        doc_id: str,
        sections: Optional[Dict[str, str]] = None,
        title: str = "",
        abstract: str = ""
    ) -> List[TextChunk]:
        all_chunks = []
        
        if sections:
            title_context = f"Paper: {title}\n\n" if title else ""
            
            for section_name, section_text in sections.items():
                if len(section_text) < self.min_chunk_size:
                    continue
                if section_name.lower() in ['references', 'bibliography', 'acknowledgments']:
                    continue
                
                section_text_with_header = f"[{section_name.upper()}]\n{section_text}"
                
                chunks = self.chunk_text(
                    section_text_with_header,
                    doc_id,
                    section=section_name
                )
                
                if chunks and title_context:
                    chunks[0].text = title_context + chunks[0].text
                    chunks[0].metadata["has_title_context"] = True
                
                all_chunks.extend(chunks)
        else:
            all_chunks = self.chunk_text(full_text, doc_id, section="full_document")
            
            if all_chunks and title:
                all_chunks[0].text = f"Paper: {title}\n\n{all_chunks[0].text}"
                all_chunks[0].metadata["has_title_context"] = True
        
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
    
    # Chunks multiple documents and returns combined list of chunks
    def chunk_documents(
        self,
        documents: List
    ) -> List[TextChunk]:
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
    
    # Initializes the class with configuration parameters
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.boundary_patterns = [
            r'\n\n+',
            r'\n\s*[-•]\s+',
            r'\n\s*\d+\.\s+',
            r'\n\s*[A-Z][^.]*:\s*\n',
        ]
    
    # Finds optimal text split point near target position using semantic boundaries
    def _find_best_split_point(self, text: str, target_pos: int, window: int = 200) -> int:
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_text = text[start:end]
        
        para_match = re.search(r'\n\n+', search_text)
        if para_match:
            return start + para_match.end()
        
        sent_match = re.search(r'[.!?]\s+', search_text)
        if sent_match:
            return start + sent_match.end()
        
        return target_pos
    
    # Chunks text using semantic boundaries like paragraph breaks
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str = ""
    ) -> List[TextChunk]:
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return super().chunk_text(text, doc_id, section)
        
        chunks = []
        current_paragraphs = []
        current_chars = 0
        chunk_index = 0
        
        target_chars = self.chunk_size * 5
        overlap_chars = self.chunk_overlap * 5
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_chars + para_len > target_chars and current_paragraphs:
                chunk_text = '\n\n'.join(current_paragraphs)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        [chunk_text], doc_id, section, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
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
        
        if current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    [chunk_text], doc_id, section, chunk_index
                )
                chunks.append(chunk)
        
        return chunks
