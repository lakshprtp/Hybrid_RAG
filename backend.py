import os
import sys
import io
import uuid
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import pypdf
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tavily import TavilyClient

# Gemini SDK (New modern google-genai with fallback)
try:
    from google import genai
    from google.genai import types as genai_types
    USE_NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_GENAI = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG-Backend")

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'")
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"').strip("'")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_knowledge_base").strip().strip('"').strip("'")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip().strip('"').strip("'")
# Clean model name if passed with 'models/' prefix for google-genai
if USE_NEW_GENAI and EMBEDDING_MODEL.startswith("models/"):
    EMBEDDING_MODEL = EMBEDDING_MODEL.replace("models/", "")
if "text-embedding-004" in EMBEDDING_MODEL:
    EMBEDDING_MODEL = "gemini-embedding-001"

GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite").strip().strip('"').strip("'")
if USE_NEW_GENAI and GENERATION_MODEL.startswith("models/"):
    GENERATION_MODEL = GENERATION_MODEL.replace("models/", "")
if "1.5-flash" in GENERATION_MODEL or "2.5-flash" in GENERATION_MODEL:
    GENERATION_MODEL = "gemini-3.1-flash-lite"

LOCAL_QDRANT_PATH = "./qdrant_storage"
VECTOR_DIMENSION = 768  # text-embedding-004 outputs 768 dimensions

# Initialize FastAPI App
app = FastAPI(
    title="Gemini + Qdrant + Tavily RAG API",
    description="High-performance Retrieval-Augmented Generation system powered by Qdrant Vector DB, Gemini Embeddings, and Tavily Search.",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
qdrant_client: Optional[QdrantClient] = None
tavily_client: Optional[TavilyClient] = None
genai_client: Optional[Any] = None


# ==========================================
# Client Initializations & Helpers
# ==========================================

def get_qdrant_client() -> QdrantClient:
    """Initialize or return the Qdrant client."""
    global qdrant_client
    if qdrant_client is None:
        if QDRANT_URL:
            logger.info(f"Connecting to remote Qdrant at {QDRANT_URL}")
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, check_compatibility=False)
        else:
            logger.info(f"Initializing local embedded Qdrant storage at {LOCAL_QDRANT_PATH}")
            Path(LOCAL_QDRANT_PATH).mkdir(parents=True, exist_ok=True)
            qdrant_client = QdrantClient(path=LOCAL_QDRANT_PATH, check_compatibility=False)
    return qdrant_client


def ensure_collection():
    """Ensure the Qdrant collection exists with the required vector dimension."""
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            logger.info(f"Creating collection '{COLLECTION_NAME}' (dim={VECTOR_DIMENSION}, metric=COSINE)")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")

        # Ensure payload index on thread_id exists for fast filtered searches
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="thread_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise


def get_genai_client(api_key: Optional[str] = None):
    """Initialize or return Gemini GenAI client with whitespace/quote sanitized key."""
    global genai_client
    key = api_key or GEMINI_API_KEY
    if key:
        key = key.strip().strip('"').strip("'")
    if not key or key == "your_gemini_api_key_here":
        return None

    if USE_NEW_GENAI:
        if genai_client is None or getattr(genai_client, "_api_key", None) != key:
            genai_client = genai.Client(api_key=key)
            genai_client._api_key = key
        return genai_client
    else:
        genai.configure(api_key=key)
        return genai


def get_tavily_client(api_key: Optional[str] = None) -> Optional[TavilyClient]:
    """Get or initialize Tavily client with sanitized key."""
    global tavily_client
    key = api_key or TAVILY_API_KEY
    if key:
        key = key.strip().strip('"').strip("'")
    if key and key != "your_tavily_api_key_here":
        if tavily_client is None or getattr(tavily_client, "api_key", None) != key:
            tavily_client = TavilyClient(api_key=key)
        return tavily_client
    return None


@app.on_event("startup")
def startup_event():
    try:
        ensure_collection()
    except Exception as e:
        logger.warning(f"Could not auto-create collection at startup: {e}")
    if GEMINI_API_KEY:
        get_genai_client()


# ==========================================
# Embedding & Search Services
# ==========================================

def compute_gemini_embedding(text: str, custom_key: Optional[str] = None, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    """Generate embedding for a single text chunk using Gemini."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        return [0.0] * VECTOR_DIMENSION

    client = get_genai_client(custom_key)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API Key is not configured. Please set GEMINI_API_KEY in Render environment variables or .env."
        )

    last_error = None
    candidate_models = []
    for m in [EMBEDDING_MODEL, "gemini-embedding-001", "models/gemini-embedding-001", "gemini-embedding-2-preview", "gemini-embedding-2", "text-embedding-004"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    try:
        if USE_NEW_GENAI:
            for em_model in candidate_models:
                for cfg in [
                    {"output_dimensionality": VECTOR_DIMENSION, "task_type": task_type},
                    {"task_type": task_type},
                    None
                ]:
                    try:
                        kwargs = {"model": em_model, "contents": clean_text}
                        if cfg:
                            kwargs["config"] = cfg
                        response = client.models.embed_content(**kwargs)
                        if response.embeddings:
                            return response.embeddings[0].values
                    except Exception as err:
                        last_error = err
                        continue

            # Fallback to legacy google.generativeai if available
            try:
                import google.generativeai as legacy_genai
                effective_key = (custom_key or GEMINI_API_KEY).strip().strip('"').strip("'")
                legacy_genai.configure(api_key=effective_key)
                for leg_m in ["models/text-embedding-004", "models/embedding-001", "text-embedding-004"]:
                    try:
                        resp = legacy_genai.embed_content(
                            model=leg_m,
                            content=clean_text,
                            task_type="retrieval_document"
                        )
                        emb = resp.get("embedding", [])
                        if emb:
                            return emb
                    except Exception as leg_err:
                        last_error = leg_err
                        continue
            except Exception:
                pass

            raise RuntimeError(f"All Gemini embedding models failed. Cause: {last_error}")
        else:
            response = client.embed_content(
                model=f"models/{EMBEDDING_MODEL}" if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL,
                content=clean_text,
                task_type="retrieval_document"
            )
            emb = response.get("embedding", [])
            return emb if emb else getattr(response, "embedding", [])
    except Exception as e:
        logger.error(f"Gemini embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")


def compute_batch_embeddings(texts: List[str], custom_key: Optional[str] = None, batch_size: int = 40) -> List[List[float]]:
    """Compute embeddings in batches, strictly filtering out any empty or whitespace-only chunks."""
    client = get_genai_client(custom_key)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API Key is missing. Configure GEMINI_API_KEY in Render environment variables or .env."
        )

    sanitized_texts = [t.strip() for t in texts if t and len(t.strip()) > 0]
    if not sanitized_texts:
        return []

    candidate_models = []
    for m in [EMBEDDING_MODEL, "gemini-embedding-001", "models/gemini-embedding-001", "gemini-embedding-2-preview", "gemini-embedding-2", "text-embedding-004"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    embeddings = []
    for i in range(0, len(sanitized_texts), batch_size):
        batch = sanitized_texts[i:i + batch_size]
        batch_success = False
        if USE_NEW_GENAI:
            for em_model in candidate_models:
                for cfg in [
                    {"output_dimensionality": VECTOR_DIMENSION, "task_type": "RETRIEVAL_DOCUMENT"},
                    {"task_type": "RETRIEVAL_DOCUMENT"},
                    None
                ]:
                    try:
                        kwargs = {"model": em_model, "contents": batch}
                        if cfg:
                            kwargs["config"] = cfg
                        response = client.models.embed_content(**kwargs)
                        if response.embeddings:
                            embeddings.extend([e.values for e in response.embeddings])
                            batch_success = True
                            break
                    except Exception as ex:
                        logger.warning(f"Batch embedding with {em_model} failed: {ex}")
                        continue
                if batch_success:
                    break

        if not batch_success:
            logger.warning(f"Batch embedding fallback for batch {i//batch_size}")
            for t in batch:
                embeddings.append(compute_gemini_embedding(t, custom_key, task_type="RETRIEVAL_DOCUMENT"))

    return embeddings


def search_qdrant(
    query_text: str,
    top_k: int = 8,
    custom_gemini_key: Optional[str] = None,
    thread_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search Qdrant for top_k relevant document chunks, optionally filtered by thread_id."""
    client = get_qdrant_client()
    ensure_collection()

    query_vector = compute_gemini_embedding(query_text, custom_key=custom_gemini_key, task_type="RETRIEVAL_QUERY")
    
    query_filter = None
    if thread_id:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="thread_id",
                    match=qmodels.MatchValue(value=thread_id)
                )
            ]
        )

    fetch_limit = max(top_k, 8)

    try:
        if hasattr(client, "query_points"):
            query_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=fetch_limit,
                with_payload=True
            ).points
        else:
            query_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=fetch_limit,
                with_payload=True
            )

        results = []
        for point in query_results:
            payload = point.payload or {}
            results.append({
                "id": str(point.id),
                "score": round(float(point.score), 4),
                "text": payload.get("text", ""),
                "filename": payload.get("filename", "unknown"),
                "chunk_id": payload.get("chunk_id", 0),
                "total_chunks": payload.get("total_chunks", 1),
                "thread_id": payload.get("thread_id", ""),
                "timestamp": payload.get("timestamp", "")
            })

        # Metadata / Identity query check:
        # If the user asks about the candidate, name, author, contact, or overview,
        # ensure Chunk 0 (which contains the document header/name) is always present!
        query_lower = query_text.lower()
        identity_keywords = ["name", "candidate", "who", "author", "person", "whose", "contact", "email", "phone", "profile", "summary"]
        if any(k in query_lower for k in identity_keywords) and thread_id:
            has_header = any(r.get("chunk_id") == 0 for r in results)
            if not has_header:
                try:
                    header_filter = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="thread_id", match=qmodels.MatchValue(value=thread_id))
                        ]
                    )
                    header_points, _ = client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=header_filter,
                        limit=12,
                        with_payload=True
                    )
                    for hp in header_points:
                        if hp.payload and hp.payload.get("chunk_id") == 0:
                            results.insert(0, {
                                "id": str(hp.id),
                                "score": 0.999,
                                "text": hp.payload.get("text", ""),
                                "filename": hp.payload.get("filename", "unknown"),
                                "chunk_id": hp.payload.get("chunk_id", 0),
                                "total_chunks": hp.payload.get("total_chunks", 1),
                                "thread_id": hp.payload.get("thread_id", ""),
                                "timestamp": hp.payload.get("timestamp", "")
                            })
                            break
                except Exception as ex:
                    logger.warning(f"Header retrieval note: {ex}")

        return results[:max(top_k, 8)]
    except Exception as e:
        logger.error(f"Error querying Qdrant: {e}")
        return []


def search_tavily(query: str, max_results: int = 4, search_depth: str = "basic", custom_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search the web via Tavily Search API."""
    client = get_tavily_client(custom_key)
    if not client:
        logger.warning("Tavily client is not configured (missing TAVILY_API_KEY). Skipping web search.")
        return []
    
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False
        )
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": round(float(item.get("score", 0.0)), 4)
            })
        return results
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


# ==========================================
# Text Processing & Chunking
# ==========================================

def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks respecting natural boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunk = text[start:].strip()
            if len(chunk) > 3:
                chunks.append(chunk)
            break

        break_pos = text.rfind("\n\n", start + chunk_size // 2, end)
        if break_pos == -1:
            break_pos = text.rfind(". ", start + chunk_size // 2, end)
            if break_pos != -1:
                break_pos += 1
        if break_pos == -1:
            break_pos = text.rfind("\n", start + chunk_size // 2, end)
        if break_pos == -1:
            break_pos = text.rfind(" ", start + chunk_size // 2, end)

        if break_pos == -1 or break_pos <= start:
            break_pos = end

        chunk = text[start:break_pos].strip()
        if len(chunk) > 3:
            chunks.append(chunk)
        
        start = max(start + 1, break_pos - chunk_overlap)

    return chunks


def extract_text_from_file(
    filename: str,
    content_bytes: bytes,
    pdf_password: Optional[str] = None,
    custom_gemini_key: Optional[str] = None
) -> str:
    """Extract readable text from PDF, TXT, MD, CSV, etc. with PDF password support and Gemini Multimodal OCR."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            if reader.is_encrypted:
                decrypted = False
                # Try empty password first (some PDFs have blank owner passwords)
                try:
                    res = reader.decrypt("")
                    if res != pypdf.PasswordType.NOT_DECRYPTED and res != 0:
                        decrypted = True
                except Exception:
                    pass

                # If still encrypted and user provided a password, try it
                if not decrypted and pdf_password:
                    try:
                        res = reader.decrypt(pdf_password.strip())
                        if res != pypdf.PasswordType.NOT_DECRYPTED and res != 0:
                            decrypted = True
                    except Exception:
                        pass

                if not decrypted:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"File '{filename}' is password-protected (encrypted). "
                            "UIDAI e-Aadhaar cards and locked PDFs require a password to open. "
                            "(UIDAI e-Aadhaar password format: First 4 letters of your name in CAPITAL + 4-digit Year of Birth, e.g. 'LAKS2001'). "
                            "Please enter the password in the 'PDF Password' field before clicking Index, or remove password protection first."
                        )
                    )

            pages_text = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    clean_page = page_text.strip()
                    if clean_page:
                        pages_text.append(f"[Page {i+1}]\n{clean_page}")
                except Exception:
                    continue

            extracted = "\n\n".join(pages_text).strip()
            # If readable text was extracted, return it
            if extracted and len(extracted) > 20:
                return extracted

            # If pypdf extracted nothing (scanned image / photo in PDF), attempt Gemini Multimodal OCR
            logger.info(f"PDF '{filename}' had no digital text layer ({len(extracted)} chars). Attempting Gemini Multimodal OCR extraction...")
            genai_client = get_genai_client(custom_gemini_key)
            if genai_client and USE_NEW_GENAI:
                try:
                    ocr_response = genai_client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=[
                            genai_types.Part.from_bytes(data=content_bytes, mime_type="application/pdf"),
                            (
                                "You are an accurate document OCR system. Extract all readable text, titles, candidate names, "
                                "labels, ID numbers, addresses, dates, and tables from this document verbatim. "
                                "Output ONLY the extracted text content."
                            )
                        ]
                    )
                    ocr_text = getattr(ocr_response, "text", "") or ""
                    if ocr_text.strip() and len(ocr_text.strip()) > 10:
                        logger.info(f"Gemini OCR successfully extracted {len(ocr_text)} characters from scanned PDF '{filename}'.")
                        return ocr_text.strip()
                except Exception as ocr_err:
                    logger.warning(f"Gemini Multimodal OCR failed for {filename}: {ocr_err}")

            if not extracted:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' contains only scanned images or photos with no extractable digital text. Please upload a searchable PDF or text file."
                )
            return extracted
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"PDF extraction error for {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF '{filename}': {str(e)}")

    # For non-PDF text formats, decode with standard text encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content_bytes.decode("utf-8", errors="ignore")


# ==========================================
# Pydantic Request/Response Models
# ==========================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query or question")
    search_mode: str = Field("hybrid", description="Mode: 'hybrid', 'documents', or 'web'")
    top_k_docs: int = Field(4, ge=1, le=10, description="Number of document chunks to retrieve")
    max_web_results: int = Field(4, ge=1, le=10, description="Number of Tavily web search results")
    tavily_search_depth: str = Field("basic", description="Search depth: 'basic' or 'advanced'")
    thread_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    custom_gemini_key: Optional[str] = None
    custom_tavily_key: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    mode_used: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    stats: Dict[str, Any]


# ==========================================
# REST API Endpoints
# ==========================================

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB Limit


@app.get("/api/status")
def get_system_status():
    """Return health check, configuration status, and exact real-time collection statistics."""
    gemini_ready = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
    tavily_ready = bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here")
    
    qdrant_status = "error"
    doc_count = 0
    try:
        client = get_qdrant_client()
        ensure_collection()
        try:
            count_res = client.count(collection_name=COLLECTION_NAME, exact=True)
            doc_count = count_res.count
        except Exception:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            doc_count = info.points_count or 0
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"unreachable: {str(e)}"

    return {
        "status": "healthy",
        "gemini_api_configured": gemini_ready,
        "tavily_api_configured": tavily_ready,
        "qdrant_status": qdrant_status,
        "collection_name": COLLECTION_NAME,
        "total_chunks_indexed": doc_count,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "sdk": "google-genai" if USE_NEW_GENAI else "google-generativeai"
    }


def index_files(
    files_data: List[tuple],
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    custom_gemini_key: Optional[str] = None,
    thread_id: Optional[str] = None,
    pdf_password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest files (PDF, TXT, MD), chunk text, compute Gemini embeddings,
    and index them into Qdrant vector database (tagged with thread_id).
    Strictly enforces a 2MB max file size limit.
    """
    client_genai = get_genai_client(custom_gemini_key)
    if not client_genai:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API Key is missing. Please provide it or set GEMINI_API_KEY in .env."
        )

    # Validate 2MB size limit upfront for all files
    for filename, content in files_data:
        if len(content) > MAX_FILE_SIZE_BYTES:
            size_mb = round(len(content) / (1024 * 1024), 2)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{filename}' ({size_mb}MB) exceeds the 2MB size limit. Please upload files under 2MB."
            )

    client = get_qdrant_client()
    ensure_collection()

    total_chunks_added = 0
    processed_files = []

    for filename, content in files_data:
        raw_text = extract_text_from_file(
            filename,
            content,
            pdf_password=pdf_password,
            custom_gemini_key=custom_gemini_key
        )
        
        if not raw_text.strip():
            logger.warning(f"File {filename} contained no extractable text.")
            continue

        raw_chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 3]
        if not chunks:
            continue

        logger.info(f"Generating embeddings for {len(chunks)} chunks from '{filename}' (thread: {thread_id})...")
        embeddings = compute_batch_embeddings(chunks, custom_key=custom_gemini_key)

        points = []
        upload_time = time.strftime("%Y-%m-%d %H:%M:%S")
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "filename": filename,
                "chunk_id": idx,
                "total_chunks": len(chunks),
                "text": chunk,
                "thread_id": thread_id or "default",
                "timestamp": upload_time
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

        for i in range(0, len(points), 100):
            batch_points = points[i:i + 100]
            client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

        total_chunks_added += len(chunks)
        processed_files.append({
            "filename": filename,
            "chunks_count": len(chunks),
            "size_bytes": len(content)
        })

    return {
        "message": f"Successfully processed {len(processed_files)} file(s). Added {total_chunks_added} chunks into Qdrant.",
        "processed_files": processed_files,
        "total_chunks": total_chunks_added,
        "thread_id": thread_id
    }


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(400),
    chunk_overlap: int = Form(100),
    custom_gemini_key: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    pdf_password: Optional[str] = Form(None)
):
    """
    Upload documents (PDF, TXT, MD), chunk text, compute Gemini embeddings,
    and index them into Qdrant vector database via FastAPI endpoint.
    """
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append((f.filename, content))
    return index_files(
        files_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        custom_gemini_key=custom_gemini_key,
        thread_id=thread_id,
        pdf_password=pdf_password
    )


@app.get("/api/documents")
def list_documents(thread_id: Optional[str] = None):
    """List distinct document names and chunk counts currently indexed in Qdrant (filtered by thread_id if specified)."""
    client = get_qdrant_client()
    ensure_collection()
    
    try:
        scroll_filter = None
        if thread_id:
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="thread_id",
                        match=qmodels.MatchValue(value=thread_id)
                    )
                ]
            )

        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        doc_stats = {}
        for point in scroll_res:
            p = point.payload or {}
            fn = p.get("filename", "unknown")
            ts = p.get("timestamp", "unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {"filename": fn, "chunks": 0, "uploaded_at": ts}
            doc_stats[fn]["chunks"] += 1

        return {
            "total_documents": len(doc_stats),
            "total_chunks": len(scroll_res),
            "thread_id": thread_id,
            "documents": list(doc_stats.values())
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return {"total_documents": 0, "total_chunks": 0, "documents": []}


@app.delete("/api/documents")
def clear_all_documents(thread_id: Optional[str] = None):
    """
    Delete indexed documents for a specific thread or reset the entire collection.
    Guarantees immediate purge in Qdrant with wait=True.
    """
    client = get_qdrant_client()
    try:
        if thread_id:
            del_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="thread_id",
                        match=qmodels.MatchValue(value=thread_id)
                    )
                ]
            )
            # 1. Scroll point IDs for this thread to delete by specific IDs
            try:
                points_to_delete, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=del_filter,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False
                )
                if points_to_delete:
                    ids = [p.id for p in points_to_delete]
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=ids,
                        wait=True
                    )
                    logger.info(f"Purged {len(ids)} points by ID for thread '{thread_id}'.")
            except Exception as scroll_err:
                logger.warning(f"ID-based deletion scroll note: {scroll_err}")

            # 2. Also delete using filter with wait=True to guarantee no orphan points remain
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=del_filter,
                wait=True
            )
            logger.info(f"Purged all points with filter thread_id='{thread_id}'.")
            return {"message": f"Documents for thread '{thread_id}' cleared successfully."}
        else:
            client.delete_collection(collection_name=COLLECTION_NAME)
            ensure_collection()
            return {"message": f"Collection '{COLLECTION_NAME}' cleared and reset successfully."}
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def clean_rag_answer(text: str) -> str:
    """Strip repetitive boilerplate filler phrases like 'Based on the provided document(s)', etc."""
    import re
    cleaned = text.strip()
    pattern = r'^(?:[\*\_]*(?:Based on|According to|From|As per)\s+(?:the\s+)?(?:provided|uploaded|attached|given)?\s*(?:(?:document|file|context|information|text|excerpt)[s]?\s*)+[\*\_]*[,\s:\-]*)+'
    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^[,\s:\-\*\_]+', '', cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


@app.post("/api/query", response_model=QueryResponse)
def execute_rag_query(request: QueryRequest):
    """
    RAG Query Endpoint:
    1. Retrieves relevant document context from Qdrant Vector DB (Gemini embeddings).
    2. Retrieves live web results from Tavily Search API.
    3. Synthesizes a grounded, source-attributed answer with Gemini LLM.
    """
    start_time = time.time()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    client_genai = get_genai_client(request.custom_gemini_key)
    if not client_genai:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API Key is missing. Please configure it in .env or the UI."
        )

    doc_sources: List[Dict[str, Any]] = []
    web_sources: List[Dict[str, Any]] = []

    # 1. Document Retrieval via Qdrant
    if request.search_mode in ["hybrid", "documents"]:
        try:
            doc_sources = search_qdrant(
                query,
                top_k=request.top_k_docs,
                custom_gemini_key=request.custom_gemini_key,
                thread_id=request.thread_id
            )
        except Exception as e:
            logger.error(f"Document search failed: {e}")

    # If in Documents Only mode, enforce strict grounding if no documents found
    if request.search_mode == "documents" and not doc_sources:
        thread_docs = list_documents(thread_id=request.thread_id)
        if thread_docs.get("total_documents", 0) == 0:
            msg = (
                f"⚠️ **No documents have been indexed in this chat thread yet.**\n\n"
                f"Please upload your document (PDF, TXT, MD) using the **Document Ingestion** section in the left sidebar to ask questions about it.\n\n"
                f"*(Note: In multi-thread mode, uploaded documents are private to each chat. If you uploaded a document in a previous chat, switch to it from the sidebar or re-upload it here).* "
            )
        else:
            msg = (
                f"⚠️ I could not find information about **'{query}'** in your uploaded documents for this chat.\n\n"
                f"*(In **Documents Only** mode, responses are strictly grounded in your uploaded files. To search the internet for this topic, please switch to **Hybrid** or **Web Search** mode).* "
            )
        return QueryResponse(
            query=query,
            answer=msg,
            mode_used=request.search_mode,
            document_sources=[],
            web_sources=[],
            stats={
                "total_latency_seconds": round(time.time() - start_time, 3),
                "generation_time_seconds": 0.0,
                "doc_chunks_retrieved": 0,
                "web_results_retrieved": 0
            }
        )

    # 2. Web Retrieval via Tavily
    if request.search_mode in ["hybrid", "web"]:
        try:
            web_sources = search_tavily(
                query,
                max_results=request.max_web_results,
                search_depth=request.tavily_search_depth,
                custom_key=request.custom_tavily_key
            )
        except Exception as e:
            logger.error(f"Web search failed: {e}")

    # Build Prompt with Context & Conversation History
    context_sections = []
    
    if request.chat_history:
        history_text = "### [RECENT CONVERSATION HISTORY]:\n"
        for turn in request.chat_history[-6:]:
            role_label = "User" if turn.get("role") == "user" else "Assistant"
            history_text += f"{role_label}: {turn.get('content', '')}\n"
        context_sections.append(history_text)

    if doc_sources:
        doc_context = "### [DOCUMENT KNOWLEDGE BASE (QDRANT VECTOR DB)]:\n"
        for i, doc in enumerate(doc_sources):
            doc_context += f"Source [{i+1}] (File: {doc['filename']}, Chunk: {doc['chunk_id']}, Score: {doc['score']}):\n"
            doc_context += f"{doc['text']}\n\n"
        context_sections.append(doc_context)

    if web_sources:
        web_context = "### [LIVE WEB SEARCH RESULTS (TAVILY)]:\n"
        for i, web in enumerate(web_sources):
            web_context += f"Web Source [{i+1}] (Title: {web['title']}, URL: {web['url']}):\n"
            web_context += f"{web['content']}\n\n"
        context_sections.append(web_context)

    full_context = "\n\n".join(context_sections)

    if request.search_mode == "documents":
        prompt = f"""You are a strict Document Q&A assistant operating in 'Documents Only' mode.

CRITICAL MANDATORY INSTRUCTIONS:
1. Answer DIRECTLY, naturally, and concisely. NEVER begin your answer with filler phrases like "Based on the provided document(s)", "According to the provided document", "From the document", etc. State the facts directly.
2. You MUST answer the user's question using EXCLUSIVELY the facts explicitly stated in the provided [DOCUMENT KNOWLEDGE BASE] below.
3. You are STRICTLY FORBIDDEN from answering using your pre-trained general knowledge, external assumptions, or outside facts.
4. If the answer to the user's question is NOT explicitly contained in or directly supported by the document excerpts below, you MUST REFUSE to answer.
Your entire response in that case MUST be:
"I could not find information about '{query}' in your uploaded documents. 

In 'Documents Only' mode, answers are strictly limited to the content of your uploaded files. To search the internet for this question, please switch to **Hybrid** or **Web Search** mode."
5. Do NOT answer the question and then add a disclaimer footnote. If the document does not contain the answer, strictly refuse.

==============================================
[DOCUMENT KNOWLEDGE BASE]:
==============================================
{full_context}
==============================================

USER QUESTION:
{query}

ANSWER:"""
    elif not full_context:
        prompt = f"""You are a helpful and knowledgeable AI assistant.
The user asked: {query}

Note: No specific document context or web search results were found for this query.
Please answer the query clearly to the best of your ability, while informing the user that no specific indexed documents or web search results were retrieved."""
    else:
        prompt = f"""You are an advanced RAG (Retrieval-Augmented Generation) assistant powered by Qdrant Vector DB, Gemini Embeddings, and Tavily Search.

Analyze the retrieved context and conversation history below and answer the user's question accurately, thoroughly, and objectively.

CRITICAL INSTRUCTIONS:
1. Answer DIRECTLY, naturally, and concisely. NEVER begin your answer with boilerplate filler phrases like "Based on the provided document(s)", "According to the document", "Based on the context", etc. State the facts directly.
2. Ground your answer in the provided context and conversation history whenever applicable.
3. Clearly cite your sources:
   - For internal documents, cite like: [Doc: filename (Chunk #)]
   - For web search results, cite like: [Web: Title](URL)
4. If the user asks a follow-up question, use the recent conversation history to provide a coherent, continuous response.
5. If the document context and web search information conflict, highlight the discrepancy clearly.
6. If the provided context does not fully answer the question, state what is known from the context and what remains uncertain.
7. Use clear Markdown headings, bullet points, and code blocks where helpful.
8. CRITICAL SOURCE ATTRIBUTION:
   At the very end of your response, on its own line, you MUST declare which source type was actually used to answer the question:
   <!--SOURCES_USED: ["documents"]-->   (if answered using ONLY the internal document excerpts)
   <!--SOURCES_USED: ["web"]-->         (if answered using ONLY the web search results)
   <!--SOURCES_USED: ["both"]-->        (if your answer synthesized facts from BOTH documents and web search)
   <!--SOURCES_USED: ["none"]-->        (if neither context source contained relevant facts)

==============================================
CONTEXT & CONVERSATION HISTORY:
==============================================
{full_context}
==============================================

USER QUESTION:
{query}

ANSWER:"""

    generation_start = time.time()
    answer = ""
    candidate_models = [GENERATION_MODEL, "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.7-flash", "gemini-3.8-flash"]
    for cand_model in candidate_models:
        try:
            if USE_NEW_GENAI:
                response = client_genai.models.generate_content(
                    model=cand_model,
                    contents=prompt
                )
                answer = response.text if response.text else "No response generated by model."
            else:
                model = client_genai.GenerativeModel(cand_model)
                response = model.generate_content(prompt)
                answer = response.text if response.text else "No response generated by model."
            if answer and not answer.startswith("⚠️"):
                break
        except Exception as ex:
            logger.warning(f"Generation with {cand_model} failed: {ex}")
            continue

    if not answer:
        answer = "⚠️ Failed to generate response from all available Gemini models."
    generation_time = round(time.time() - generation_start, 3)

    # Filter sources dynamically in Hybrid mode
    if request.search_mode == "hybrid" and answer:
        import re
        tag_match = re.search(r'<!--SOURCES_USED:\s*\[(.*?)\]-->', answer, re.IGNORECASE)
        clean_answer = re.sub(r'<!--SOURCES_USED:.*?-->', '', answer).strip()
        
        sources_detected = []
        if tag_match:
            sources_detected = [s.strip().strip('"\'').lower() for s in tag_match.group(1).split(',') if s.strip()]

        # Heuristic fallback if tag was omitted or malformed
        if not sources_detected:
            has_doc_citation = "[Doc:" in clean_answer or any(d.get("filename") in clean_answer for d in doc_sources[:3] if d.get("filename") != "unknown")
            has_web_citation = "[Web:" in clean_answer or any(w.get("url") in clean_answer for w in web_sources[:3] if w.get("url"))
            if has_doc_citation and has_web_citation:
                sources_detected = ["both"]
            elif has_doc_citation:
                sources_detected = ["documents"]
            elif has_web_citation:
                sources_detected = ["web"]
            else:
                max_doc_score = max([d.get("score", 0.0) for d in doc_sources], default=0.0)
                if max_doc_score >= 0.65 and not web_sources:
                    sources_detected = ["documents"]
                elif max_doc_score < 0.45 and web_sources:
                    sources_detected = ["web"]
                else:
                    sources_detected = ["both"]

        if "documents" in sources_detected and "web" not in sources_detected and "both" not in sources_detected:
            web_sources = []
            logger.info("Hybrid mode: Answer sourced exclusively from documents. Web sources suppressed.")
        elif "web" in sources_detected and "documents" not in sources_detected and "both" not in sources_detected:
            doc_sources = []
            logger.info("Hybrid mode: Answer sourced exclusively from web search. Document sources suppressed.")
        elif "none" in sources_detected:
            doc_sources = []
            web_sources = []
            logger.info("Hybrid mode: Neither source contained relevant information.")

        answer = clean_answer

    # Strip boilerplate filler like "Based on the provided document(s)..."
    answer = clean_rag_answer(answer)

    total_time = round(time.time() - start_time, 3)

    return QueryResponse(
        query=query,
        answer=answer,
        mode_used=request.search_mode,
        document_sources=doc_sources,
        web_sources=web_sources,
        stats={
            "total_latency_seconds": total_time,
            "generation_time_seconds": generation_time,
            "doc_chunks_retrieved": len(doc_sources),
            "web_results_retrieved": len(web_sources)
        }
    )


# Serve Static UI Frontend if index.html exists
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "message": "RAG Backend API is running.",
            "docs": "/docs",
            "status": "/api/status"
        }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", 8000))
    print(f"🚀 Starting RAG Backend server at http://{host}:{port} ...")
    uvicorn.run("backend:app", host=host, port=port, reload=True)
