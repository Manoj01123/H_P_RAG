# # from fastapi import FastAPI, Form
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# # from fastapi.requests import Request
# # from rag_retrieval import query_rag
# # import uvicorn
# # import os
# # import requests
# # import logging
# #
# # import sys
# # print(sys.executable)
# # print(sys.path)
# #
# #
# # # Initialize FastAPI app
# # app = FastAPI()
# # templates = Jinja2Templates(directory="templates")
# #
# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# #
# # # Helper function to download files from S3
# # # def download_from_s3(url, local_path):
# # #     logging.info(f"Downloading model from S3: {url}")
# # #     response = requests.get(url, stream=True)
# # #     with open(local_path, "wb") as file:
# # #         for chunk in response.iter_content(chunk_size=8192):
# # #             file.write(chunk)
# # #     logging.info(f"Model downloaded and saved to {local_path}")
# #
# # def download_from_s3(url, local_path):
# #     if os.path.exists(local_path):
# #         print(f"{local_path} already exists. Skipping download.")
# #         return
# #
# #     try:
# #         print(f"Downloading model from {url}...")
# #         response = requests.get(url, stream=True)
# #         response.raise_for_status()  # Raise an HTTPError if the request returned an unsuccessful status code
# #         with open(local_path, "wb") as file:
# #             for chunk in response.iter_content(chunk_size=8192):
# #                 file.write(chunk)
# #         print(f"Downloaded model to {local_path}.")
# #     except Exception as e:
# #         print(f"Failed to download model: {e}")
# #         raise
# #
# #
# # # Load model once globally
# # model_initialized = False
# # local_path = "model.gguf"
# # s3_url = "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf"
# #
# # if not os.path.exists(local_path):
# #     download_from_s3(s3_url, local_path)
# #
# # @app.get("/", response_class=HTMLResponse)
# # def get_home(request: Request):
# #     return templates.TemplateResponse("index.html", {"request": request, "answer": None})
# #
# # @app.post("/query", response_class=HTMLResponse)
# # def query_rag_endpoint(request: Request, question: str = Form(...)):
# #     global model_initialized
# #     if not model_initialized:
# #         logging.info("Initializing RAG model...")
# #         model_initialized = True
# #
# #     response = query_rag(question)
# #     return templates.TemplateResponse(
# #         "index.html",
# #         {"request": request, "answer": response["result"], "question": question}
# #     )
# #
# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8080)
# #
#
#
#
# from fastapi import FastAPI, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.requests import Request
# from rag_retrieval import query_rag  # Ensure this is updated to handle the model path dynamically
# import uvicorn
# import os
# import requests
# import logging
#
# # Print system info for debugging (optional, can be removed in production)
# import sys
# print(sys.executable)
# print(sys.path)
#
# # Initialize FastAPI app
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
#
# # S3 model configuration
# S3_URL = os.getenv("MODEL_S3_URL", "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf")
# LOCAL_PATH = "/tmp/model.gguf"  # Use ephemeral storage for Render
#
# def download_from_s3(url, local_path):
#     """Download the model file from S3 if it doesn't already exist."""
#     if os.path.exists(local_path):
#         logging.info(f"{local_path} already exists. Skipping download.")
#         return
#
#     try:
#         logging.info(f"Downloading model from {url}...")
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise an error for bad status codes
#         with open(local_path, "wb") as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         logging.info(f"Model downloaded to {local_path}.")
#     except Exception as e:
#         logging.error(f"Failed to download model: {e}")
#         raise
#
# # Download the model during app startup
# try:
#     download_from_s3(S3_URL, LOCAL_PATH)
# except Exception as e:
#     logging.error(f"Critical error during model download: {e}")
#     raise
#
# # Global flag for model initialization
# model_initialized = False
#
# @app.get("/", response_class=HTMLResponse)
# async def get_home(request: Request):
#     """Render the home page with a question form."""
#     return templates.TemplateResponse("index.html", {"request": request, "answer": None})
#
# @app.post("/query", response_class=HTMLResponse)
# async def query_rag_endpoint(request: Request, question: str = Form(...)):
#     """Handle user queries and return answers from the RAG model."""
#     global model_initialized
#     if not model_initialized:
#         logging.info("Initializing RAG model...")
#         # Pass the downloaded model path to `query_rag`
#         os.environ["MODEL_PATH"] = LOCAL_PATH  # Ensure the model path is passed dynamically
#         model_initialized = True
#
#     try:
#         response = query_rag(question)
#         return templates.TemplateResponse(
#             "index.html",
#             {"request": request, "answer": response["result"], "question": question}
#         )
#     except Exception as e:
#         logging.error(f"Error during query processing: {e}")
#         return templates.TemplateResponse(
#             "index.html",
#             {"request": request, "answer": "An error occurred while processing your query.", "question": question}
#         )
#
# if __name__ == "__main__":
#     # Use the Render-recommended PORT environment variable
#     port = int(os.getenv("PORT", 8080))
#     uvicorn.run(app, host="0.0.0.0", port=port)

import os
import requests
import logging
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from rag_retrieval import query_rag
import uvicorn

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)

# AWS S3 Model URL (Stored as an Environment Variable)
S3_URL = os.getenv("MODEL_S3_URL", "https://harrypotter07.s3.us-east-1.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf")
LOCAL_PATH = "/tmp/model.gguf"  # Store in Render's temporary storage


def download_model():
    """Download the model from AWS S3 if it doesn't already exist."""
    if os.path.exists(LOCAL_PATH):
        logging.info(f"Model already exists at {LOCAL_PATH}. Skipping download.")
        return

    try:
        logging.info(f"Downloading model from {S3_URL}...")
        response = requests.get(S3_URL, stream=True)
        response.raise_for_status()

        with open(LOCAL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logging.info(f"Model successfully downloaded to {LOCAL_PATH}.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise


# Ensure model is downloaded before running the API
download_model()


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Render the home page with a question form."""
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})


@app.post("/query", response_class=HTMLResponse)
async def query_rag_endpoint(request: Request, question: str = Form(...)):
    """Handle user queries and return answers from the RAG model."""
    try:
        response = query_rag(question)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer": response['result'], "question": question}
        )
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer": "An error occurred while processing your query.", "question": question}
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
