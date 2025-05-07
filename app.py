import torch
from flask import Flask, request, render_template_string, jsonify
from transformers import BertTokenizer
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

# Define the model architecture (same as training)
class ToxicModel(nn.Module):
    def __init__(self, num_labels, hidden_size=256):
        super(ToxicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.metadata_fc = nn.Linear(1, 32)
        self.fc = nn.Linear(hidden_size * 2 + 32, 256)
        self.output = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, metadata):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_input = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        metadata_output = self.metadata_fc(metadata)
        combined_output = torch.cat((lstm_output, metadata_output), dim=1)
        fc_output = F.relu(self.fc(combined_output))
        output = torch.sigmoid(self.output(fc_output))
        return output

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'toxic_model.pth'  # Update path to your model
model = ToxicModel(num_labels=6)  # Modify num_labels if needed
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function with identity-related hate terms
def predict_toxicity(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    metadata = torch.tensor([[len(text.split())]], dtype=torch.float32)  # Using the comment length as metadata

    # Define expanded list of identity-related terms
    identity_terms = [
        "gay", "lesbian", "trans", "transgender", "homosexual", "bisexual", "queer",
        "black", "white", "asian", "indian", "latino", "hispanic", "arab",
        "muslim", "christian", "jew", "jewish", "atheist", "sikh", "buddhist",
        "disabled", "autistic", "deaf", "blind", "retard", "mental", "cripple",
        "refugee", "immigrant", "migrant", "ethnic", "racial", "race", "tribe",
        "gender", "sex", "nonbinary", "female", "woman", "man", "male", "female"
    ]
    lower_text = text.lower()
    identity_flag = any(term in lower_text for term in identity_terms)

    # Run the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, metadata)
        predictions = outputs.squeeze().tolist()

    # Boost identity_hate score if identity terms are found
    if identity_flag and isinstance(predictions, list) and len(predictions) == 6:
        predictions[5] = max(predictions[5], 0.95)  # identity_hate index = 5

    return predictions

# Define route for real-time prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get comment text from request
    data = request.get_json()
    text = data.get('comment', '')
    
    if not text:
        return jsonify({"error": "No comment provided"}), 400
    
    # Make prediction
    predictions = predict_toxicity(text)
    
    # Format the predictions as a dictionary (the labels are in the same order as your model's output)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = {label: prediction for label, prediction in zip(labels, predictions)}

    # Return prediction results as JSON
    return jsonify(result)

# Define the main page route (HTML for input form)
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Toxicity Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .container {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 80%;
                max-width: 600px;
                text-align: center;
            }

            h1 {
                color: #333;
                font-size: 24px;
            }

            form {
                margin-top: 20px;
            }

            textarea {
                width: 100%;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
                font-size: 16px;
                resize: vertical;
            }

            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                margin-top: 10px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 5px;
            }

            button:hover {
                background-color: #45a049;
            }

            #result {
                margin-top: 20px;
                display: none;
                text-align: left;
            }

            #result ul {
                list-style-type: none;
                padding: 0;
            }

            #result li {
                padding: 5px 0;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Toxicity Detection in Comments</h1>
            <form id="commentForm">
                <label for="comment">Enter your comment:</label>
                <textarea id="comment" name="comment" rows="5" placeholder="Type your comment here..." required></textarea>
                <button type="submit">Submit</button>
            </form>

            <div id="result">
                <h3>Prediction Results:</h3>
                <ul id="predictionList"></ul>
            </div>
        </div>

        <script>
            document.getElementById('commentForm').addEventListener('submit', function(event) {
                event.preventDefault();
                const comment = document.getElementById('comment').value;
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ comment: comment })
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    const predictionList = document.getElementById('predictionList');
                    predictionList.innerHTML = '';

                    for (const [key, value] of Object.entries(data)) {
                        const listItem = document.createElement('li');
                        listItem.textContent = `${key}: ${value.toFixed(4)}`;
                        predictionList.appendChild(listItem);
                    }

                    resultDiv.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        </script>
    </body>
    </html>
    ''')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
