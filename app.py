# File: app.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']

    embedding_1 = model.encode(text1, convert_to_tensor=True)
    embedding_2 = model.encode(text2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    similarity_score_value = similarity.item()

    response = {
        "similarity score": similarity_score_value
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
