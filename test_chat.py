import asyncio
from api.index import chat

GLOBAL_STATE = {
    "vector_db": None,
    "bm25": None,
    "chunks": ["test chunk 1", "test chunk 2"],
    "metadata": [{"source": "test.pdf", "page": 1}, {"source": "test.pdf", "page": 2}],
    "document_summaries": {"test.pdf": "Test summary"},
    "uploaded_hashes": set(),
    "uploaded_file_names": ["test.pdf"]
}

import rag.retrieval
GLOBAL_STATE["bm25"] = rag.retrieval.create_bm25(GLOBAL_STATE["chunks"])

import api.index
api.index.GLOBAL_STATE = GLOBAL_STATE

async def main():
    try:
        res = await chat({"query": "summarize the doc", "history": []})
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
