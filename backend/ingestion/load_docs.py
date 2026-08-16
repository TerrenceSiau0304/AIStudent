from langchain_core.documents import Document
# from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import os
import pickle

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR/"data"

# def process_pdfs_recursively(root_folder, image_output_dir="pdfImage/"):
#     """
#     Recursively walks root_folder, finds all PDFs, and partitions each one.
#     Returns a dict mapping filepath -> list of elements.
#     """
#     os.makedirs(image_output_dir, exist_ok=True)
#     all_results = {}
#     passedFile = 0
#     failedFile = 0

#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.lower().endswith(".pdf"):
#                 filepath = os.path.join(dirpath, filename)
#                 print(f"Processing: {filepath}")

#                 try:
#                     raw_pdf_elements = partition_pdf(
#                         filename=filepath,
#                         extract_images_in_pdf=False,
#                         infer_table_structure=False,
#                         chunking_strategy="by_title",
#                         max_characters=5000,
#                         new_after_n_chars=4800,
#                         combine_text_under_n_chars=3000,
#                         image_output_dir_path=image_output_dir,
#                         languages=["eng"],
#                     )
#                     all_results[filepath] = raw_pdf_elements
#                     passedFile += 1
#                 except Exception as e:
#                     print(f"Failed to process {filepath}: {e}")
#                     failedFile += 1

#     print(passedFile, failedFile)
#     return all_results


def build_document(chunk: dict):
    """
    Convert dictionary-based document data into LangChain
    Document objects.

    The input dictionary is expected to contain file paths as
    keys and collections of document elements as values. Each
    element is converted into a Document object, with the
    corresponding file path stored as metadata.

    Args:
        chunk (dict):
            Dictionary containing document elements grouped by
            their source file path.

    Returns:
        list[Document]:
            A list of Document objects containing the document
            content and its source file path.
    """
    docs = []
    for filepath, elements in chunk.items():
        for element in elements:
            doc = Document(page_content=str(element),
                          metadata={
                              "source":filepath
                          })
            docs.append(doc)
    return docs


def save_document(filename: str, documents: dict):
    """
    Save document data to a pickle file.

    The file is stored inside the application's configured
    data directory.

    Args:
        filename (str):
            Name of the file used to store the documents.

        documents (dict):
            Document data that will be serialized and saved.
    """
    path = DATA_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(documents, f)


def load_document(filename: str):
    """
    Load document data from a pickle file.

    The function reads a previously saved pickle file from
    the application's data directory and deserializes its
    contents back into a Python object.

    Args:
        filename (str):
            Name of the pickle file containing the documents.

    Returns:
        Any:
            The Python object stored in the pickle file.
    """
    path = DATA_DIR / filename
    with open(path, "rb") as f:
        results= pickle.load(f)

    return results
