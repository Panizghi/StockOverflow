from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_greeting', methods=['GET'])
def get_greeting():
    # Mock response; integrate with NEAR SDK as needed
    return jsonify(greeting="Hello from NEAR!")

@app.route('/set_greeting', methods=['POST'])
def set_greeting():
    greeting = request.json.get('greeting', '')
    # Send greeting to NEAR blockchain; mock response here
    return jsonify(success=True, newGreeting=greeting)

if __name__ == '__main__':
    app.run(debug=True)
