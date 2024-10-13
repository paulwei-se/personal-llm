from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
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
    doc_id = di.process_and_index_document(file.filename)
    return {"doc_id": doc_id}

@app.post("/search")
async def search(query: Query):
    results = di.search(query.text)
    return results

@app.post("/answer")
async def answer_question(query: Query, doc_id: str):
    answer = di.answer_question(query.text, doc_id)
    return answer

@app.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    summary = di.summarize_document(doc_id)
    return {"summary": summary}

@app.get("/bias/{doc_id}")
async def detect_bias(doc_id: str):
    bias = di.detect_bias(doc_id)
    return bias

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
