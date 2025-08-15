cd src
# 1. Generar embeddings con Universal Sentence Encoder (USE)
python generate_embedding.py --model use
# 2. Generar embeddings con all-mpnet-base-v2
python generate_embedding.py --model st1
# 3. Generar embeddings con all-MiniLM-L6-v2
python generate_embedding.py --model st2
# 4. Generar embeddings con paraphrase-mpnet-base-v2
python generate_embedding.py --model st3