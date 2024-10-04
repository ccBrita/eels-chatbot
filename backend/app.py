from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from engine import EELSEngine

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize the chatbot engine
chatbot = EELSEngine()

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint to handle chat queries from the frontend.
    """
    user_input = request.json.get('message')

    # Validate user input
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Get the response from the chatbot engine
        response = chatbot.query(user_input)
        print(response)

        # Prepare the assistant's message
        assistant_message = {'role': 'assistant', 'content': str(response)}

        # Return the assistant's message in 'output'
        return jsonify({'output': assistant_message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
