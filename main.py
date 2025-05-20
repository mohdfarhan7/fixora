 re
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# ====== CONFIGURATION ======
PDF_PATH = "OMT312630 (3).pdf"
IMAGE_DIR = "images"
INDEX_PATH = "faiss_index"
os.environ["OPENAI_API_KEY"] = "YOUR OPEN AI API KEY"

def extract_text_and_images(pdf_path, image_dir="images"):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_map = {}

    for page_num in range(len(doc)):
        page = doc[page_num]

        # --- Properly extract all text ---
        blocks = page.get_text("dict")["blocks"]
        texts = []
        for block in blocks:
            for line in block.get("lines", []):
                line_text = " ".join([span["text"] for span in line["spans"]])
                texts.append(line_text.strip())
        text = "\n".join([t for t in texts if t])

        images = []
        labels = {}

        # --- Get captions from lower part of page ---
        caption_blocks = [
            b[4].strip() for b in page.get_text("blocks")
            if b[1] > 0.75 * page.rect.height and isinstance(b[4], str) and b[4].strip()
        ]
        caption_text_combined = " ".join(caption_blocks)

        # Match label-caption pairs: "1 — Description"
        caption_matches = re.findall(r"(\d+)\s*[–—-]\s*([^0-9]+?)(?=\s*\d\s*[–—-]|$)", caption_text_combined)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(image_dir, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append(image_path)

        # Assign labels to images (first image fallback)
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

def build_or_load_vectorstore(pdf_path):
    embedding = OpenAIEmbeddings()
    if os.path.exists(INDEX_PATH):
        # Allow dangerous deserialization since we trust our own index files
        return FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
    else:
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load_and_split()
        vectordb = FAISS.from_documents(pages, embedding)
        vectordb.save_local(INDEX_PATH)
        return vectordb

def extract_labels_from_text(text):
    return re.findall(r"\((\d+)\)", text)

def main():
    print("📄 Extracting text, images, and labels...")
    page_map = extract_text_and_images(PDF_PATH)

    print("🔍 Loading or creating vector store...")
    vectordb = build_or_load_vectorstore(PDF_PATH)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        return_source_documents=True
    )

    print("\n🤖 John Deere Manual QA Bot (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        full_prompt = (
            "You must only answer using the provided document. "
            "If the answer is not in the document, say: 'Answer not found in the manual.'\n\n"
            f"Question: {query}"
        )

        result = qa_chain({"query": full_prompt})
        answer_text = result["result"]
        print(f"\nBot: {answer_text}")

        # Find source page
        matched_page = None
        sources = result.get("source_documents", [])
        if sources:
            metadata = sources[0].metadata
            if "page" in metadata:
                matched_page = metadata["page"] + 1

        mentioned_labels = extract_labels_from_text(answer_text)

        if matched_page:
            page_data = page_map.get(matched_page)
            if page_data:
                labels = page_data.get("labels", {})
                images_on_page = page_data.get("images", [])

                if mentioned_labels:
                    matched_imgs = [
                        (label, labels[label]) for label in mentioned_labels if label in labels
                    ]
                    if matched_imgs:
                        print(f"(Matched labeled images from page {matched_page}):")
                        for label, info in matched_imgs:
                            print(f"  → ({label}) {info['image']} — {info.get('caption', '')}")
                    else:
                        print(f"(No labeled image matches found, showing all images from page {matched_page}):")
                        for img in images_on_page:
                            print(f"  → {img}")
                elif images_on_page:
                    print(f"(Images from page {matched_page}):")
                    for img in images_on_page:
                        print(f"  → {img}")
                else:
                    print(f"(No images found on page {matched_page}.)")
        else:
            print("(No matching page found for image lookup.)")

if __name__ == "__main__":
    main() 
