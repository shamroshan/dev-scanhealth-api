from fastapi import FastAPI, HTTPException 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import pickle

load_dotenv()

app = FastAPI()

openai.api_key = os.environ.get("OPENAI_KEY")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lively-forest-09162750f.5.azurestaticapps.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

@app.get("/")
async def helloapp():
    return {"message": "Hello App"}

async def startup_event():
    global index, chat_engine

    # Load local PDF files for processing
    documents_path = "./local_pdfs"
    reader = SimpleDirectoryReader(documents_path)
    
    # Create an index from the documents
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Setup the chat engine using the index
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

app.add_event_handler("startup", startup_event)

async def enhance_response(response: str, query: str) -> str:
    """Enhance the response by providing additional context or clarification."""
    prompt = (
        f"Enhance the following response based on the query:\n\n"
        f"Query: {query}\n"
        f"Response: {response}\n\n"
        "Provide a more detailed and accurate response in markdown format. Use lists, bold texts, italics, bullets, points, indentations etc. to visualize attractively. "
        "Ensure that the response is strictly derived from the medical content provided in the PDFs."
        "Adhere to these guidelines:\n"
        "1. Only provide answers that are strictly based on the medical information from the provided documents.\n"
        "2. Avoid general knowledge or assumptions. All answers should come directly from the PDF data.\n"
        "3. If the query is unrelated to the medical content in the PDFs:\n"
        "   - respond with 'I can only answer medical-related questions based on the provided data.'\n"
        "4. Provide a clear explanation for medical terminology and concepts when necessary, ensuring a structured and detailed response."
        "5. If the query asks for general medical advice not covered in the PDFs, respond with 'I can only provide information from the specific documents you’ve uploaded.'"
        "6. Keep the language professional, informative, and medically accurate. \n"
        "Please make sure to follow these rules strictly."
    )

    try:
        openai_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a specialized medical chatbot answering only from the provided documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        enhanced_content = openai_response.choices[0].message.content
        return enhanced_content.strip()
    except Exception as e:
        print(f"Error enhancing response: {e}")
        return response


@app.options("/chat")
async def options_chat_endpoint():
    return {}

@app.post("/chat")
async def chat(message: Message):
    if message.role != "user":
        raise HTTPException(status_code=400, detail="Invalid role")

    # Generate initial response from the chatbot
    response_stream = chat_engine.stream_chat(message.content)
    response_chunks = [chunk for chunk in response_stream.response_gen]
    response = "".join(response_chunks)

    # Enhance the response for better clarity
    enhanced_response = await enhance_response(response, message.content)
    return {"role": "assistant", "content": enhanced_response}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
