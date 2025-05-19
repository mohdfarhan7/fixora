import os
import re
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "OMT312630 (3).pdf")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# Get OpenAI API key from environment variable
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Function to reconstruct page_map (from main.py) ---
def extract_text_and_images(pdf_path, image_dir="images"):
    try:
        os.makedirs(image_dir, exist_ok=True)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        doc = fitz.open(pdf_path)
        page_map = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            texts = []
            for block in blocks:
                for line in block.get("lines", []):
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    texts.append(line_text.strip())
            text = "\n".join([t for t in texts if t])
            images = []
            labels = {}
            caption_blocks = [
                b[4].strip() for b in page.get_text("blocks")
                if b[1] > 0.75 * page.rect.height and isinstance(b[4], str) and b[4].strip()
            ]
            caption_text_combined = " ".join(caption_blocks)
            caption_matches = re.findall(r"(\d+)\s*[–—-]\s*([^0-9]+?)(?=\s*\d\s*[–—-]|$)", caption_text_combined)
            
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_ext = base_image["ext"]
                    image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    image_path = os.path.join(image_dir, image_filename)
                    images.append(image_path)
                except Exception as e:
                    logger.error(f"Error processing image on page {page_num + 1}: {str(e)}")
                    continue

            for label, caption in caption_matches:
                labels[label.strip()] = {
                    "image": images[0] if images else None,
                    "caption": caption.strip()
                }
            page_map[page_num + 1] = {
                "text": text,
                "images": images,
                "labels": labels
            }
        return page_map
    except Exception as e:
        logger.error(f"Error in extract_text_and_images: {str(e)}")
        raise

# --- Load FAISS index and page_map on startup ---
@app.on_event("startup")
async def startup_event():
    global qa_chain, page_map
    try:
        logger.info("Starting application initialization...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"BASE_DIR: {BASE_DIR}")
        logger.info(f"PDF_PATH: {PDF_PATH}")
        logger.info(f"IMAGE_DIR: {IMAGE_DIR}")
        logger.info(f"INDEX_PATH: {INDEX_PATH}")

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

        embedding = OpenAIEmbeddings()
        vectordb = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
        page_map = extract_text_and_images(PDF_PATH)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            return_source_documents=True
        )
        logger.info("Application initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def extract_labels_from_text(text):
    return re.findall(r"\((\d+)\)", text)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Fixora API is running"}

@app.get("/query")
def query_manual(user_prompt: str = Query(..., description="Your question")):
    try:
        full_prompt = (
            "You must only answer using the provided document. "
            "If the answer is not in the document, say: 'Answer not found in the manual.'\n\n"
            f"Question: {user_prompt}"
        )
        result = qa_chain({"query": full_prompt})
        answer_text = result["result"]

        # Find source page
        matched_page = None
        sources = result.get("source_documents", [])
        if sources:
            metadata = sources[0].metadata
            if "page" in metadata:
                matched_page = metadata["page"] + 1

        images = []
        if matched_page:
            page_data = page_map.get(matched_page)
            if page_data:
                images = [
                    f"/images/{os.path.basename(img)}"
                    for img in page_data.get("images", [])
                ]

        return {
            "status": "success",
            "message": "Response retrieved successfully",
            "result": answer_text,
            "images": images
        }
    except Exception as e:
        logger.error(f"Error in query_manual: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve images statically
@app.get("/images/{image_name}")
def get_image(image_name: str):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Error serving image {image_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 