from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

a = model.encode("Chiếc xe này chạy rất nhanh.")
b = model.encode("Tốc độ của chiếc xe này rất cao.")

score = cosine_similarity([a], [b])[0][0]
print(score)