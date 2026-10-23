Retrieval comparison, 25 questions, top_k=5, retriever=bm25, 2026-10-23.

| chunking | retrieval | chunks | hit@1 | hit@5 | MRR | answer coverage@5 |
|---|---|---:|---:|---:|---:|---:|
| semantic | bm25 | 6 | 0.920 | 1.000 | 0.950 | 0.943 |
| fixed | bm25 | 9 | 0.920 | 1.000 | 0.948 | 0.940 |
