from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_explorerVersion7 import *
import pandas as pd
import nltk
from nltk import word_tokenize


app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/generate')
def get_score():
    # multi_value(scores['Polarities Found'],scores['Text'], scores['Polarity Count-after Adversative'], 5)
    # multi_value(polarity_list, text, final_polarity, k)

    text = "her film is unrelentingly claustrophobic and unpleasant ."
    polarity4_list = findPolarity4(text)
    score4 = countPolarity4(polarity4_list, 7)
    polarity5_list = findPolarity5(text)
    score5 = countPolarity5(polarity5_list, 7)
    polarity6_list = findPolarity6(text)
    is_adversative_present = adversative_present(polarity6_list)
    polarity_count_after_adversative = update_p_after_adversative(is_adversative_present, score4, score5)
    polarity_count_after_adversative = qn_mark(text, polarity_count_after_adversative)
    score_multi = multi_value(polarity6_list, text, polarity_count_after_adversative, 5)
    score_multi = qn_mark(text, score_multi)
    final_score = new_multi(score_multi)

    # check if score is an integer
    if isinstance(final_score,int):
        return jsonify(
            {
                "code": 200,
                "text": text,
                "data": final_score
            }
        )
    return jsonify(
        {
            "code": 404,
            "text": text,
            "data": {},
            "message": "No polarity generated."
        }
    ), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)