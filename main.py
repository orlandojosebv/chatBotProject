from flask import Flask, render_template, request, jsonify
import chatbot_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['GET'])
def get_response():
    user_input = request.args.get('user_input')
    response = chatbot_model.get_chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
