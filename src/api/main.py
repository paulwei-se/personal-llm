from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio
from src.document_intelligence import DocumentIntelligence

app = FastAPI()
di = DocumentIntelligence(max_workers=4)

class Query(BaseModel):
    text: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    doc_id = await di.process_and_index_document(file.filename)
    if doc_id:
        return {"doc_id": doc_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to process document")

@app.post("/search")
async def search(query: Query):
    results = await di.search(query.text)
    return results

@app.post("/answer")
async def answer_question(query: Query, doc_id: str):
    answer = await di.answer_question(query.text, doc_id)
    return answer

@app.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    summary = await di.summarize_document(doc_id)
    return {"summary": summary}

@app.get("/bias/{doc_id}")
async def detect_bias(doc_id: str):
    bias = await di.detect_bias(doc_id)
    return bias

@app.delete("/document/{doc_id}")
async def remove_document(doc_id: str):
    success = await di.remove_document(doc_id)
    if success:
        return {"message": f"Document {doc_id} removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Document not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)