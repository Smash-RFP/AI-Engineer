from .modules.embedding_loader import load_embedding_model
from .modules.retrievals import run_retrieve


def retrieved_contexts(query, model_name, provider="huggingface", strategy="hybrid",
                      normalization_method="z_score", use_cross_encoder=True):
    
    embedding_model = load_embedding_model(model_name, provider)

    results = run_retrieve(
        query=query,
        strategy=strategy,
        normalization_method=normalization_method,
        use_cross_encoder=use_cross_encoder,
        embedding_model=embedding_model
    )
    return results

# from typing import List, Dict
# from langchain.schema import Document

# def retrieved_contexts(query, model_name, provider, normalization_method="z_score", use_cross_encoder=Truegy) -> List[Dict]:
#     docs: List[Document] = retriever.get_relevant_documents(query)

#     out = []
#     for i, d in enumerate(docs):
#         m = d.metadata or {}
#         out.append({
#             "retrieved_content": d.page_content,
#             "retrieved_source_id": m.get("source_id") or m.get("source") or m.get("file_path") or m.get("filename"),
#             "retrieved_chunk_id": m.get("chunk_id") or m.get("page") or m.get("page_number") or i,
#             # 필요하면 원메타 보존
#             "metadata": m,
#         })
#     return out