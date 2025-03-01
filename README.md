# Custom ChatBot with PyTorch and Azure OpenAI API Integration

## Project Overview

This project implements a hybrid chatbot that leverages both a custom-trained PyTorch model and the Azure OpenAI API. The system first attempts to classify user intents using a custom model, then falls back to the Azure OpenAI API for more complex queries. It features a responsive React frontend and a Flask backend, making it suitable for integration into personal websites, portfolios, or any application requiring a customized chatbot.

## Features

- **Dual Intelligence System**: Combines custom intent classification with Azure OpenAI's advanced capabilities
- **Intent-Based Responses**: Accurately identifies user intents from a customizable set of intents
- **Confidence Threshold**: Intelligently routes queries to the most appropriate response system
- **Responsive UI**: Clean, modern chat interface built with React
- **Real-Time Communication**: Immediate response to user queries
- **Markdown Support**: Bot responses can include formatted markdown text
- **Mobile-Friendly Design**: Works well on both desktop and mobile devices
- **Customizable**: Easily modify the intents and responses to suit your needs

## Technical Architecture

### Backend (Python/Flask)

- **Intent Recognition**: Custom PyTorch model using DistilBERT for efficient text classification
- **API Integration**: Seamless fallback to Azure OpenAI API for complex queries
- **Web Server**: Flask application serving both the API endpoints and static frontend files
- **Data Processing**: Utilities for encoding labels and processing training data

### Frontend (React)

- **Component-Based**: Modular UI with separate components for ChatBox and ChatIcon
- **State Management**: React hooks for managing chat state and UI transitions
- **Styling**: Custom CSS for a polished user experience
- **Markdown Rendering**: Support for rich text formatting in bot responses

## How It Works

1. **Intent Recognition Flow**:
   - User sends a message through the chat interface
   - The message is sent to the Flask backend via API
   - The backend processes the message using the custom PyTorch model
   - If the model's confidence exceeds the threshold, a predefined response is returned
   - If confidence is low, the query is forwarded to Azure OpenAI API

2. **Model Training**:
   - The custom model is trained on a JSON file containing intents, patterns, and responses
   - Each intent has multiple example patterns and possible responses
   - The training process converts text to tokens, then to numerical representations
   - A DistilBERT model is fine-tuned for intent classification

3. **API Fallback**:
   - When the custom model cannot confidently classify an intent, the Azure OpenAI API is used
   - This provides flexibility to handle a wide range of queries beyond the training data

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/custom-chatbot.git
   cd custom-chatbot
   ```

2. **Set up the backend**:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file with Azure OpenAI credentials
   echo "API_KEY=your_azure_openai_api_key" > .env
   echo "ENDPOINT=your_azure_openai_endpoint" >> .env
   ```

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Set up the frontend**:
   ```bash
   cd chatbot-frontend
   npm install
   ```

5. **Run the application**:
   ```bash
   # In the root directory, start the backend
   python app.py
   
   # In another terminal, start the frontend development server
   cd chatbot-frontend
   npm run dev
   ```

## Customization

### Modifying Intents and Responses

1. Edit the `chatbot-backend/Data/intents.json` file to add or modify intents
2. Each intent should have:
   - A unique tag
   - Multiple example patterns (questions/inputs)
   - Several response options (the bot will randomly select one)

Example intent format:
```json
{
  "tag": "greeting",
  "patterns": [
    "Hi", 
    "Hello", 
    "Hey there"
  ],
  "responses": [
    "Hello! How can I help you today?",
    "Hi there! What can I assist you with?"
  ]
}
```

3. After modifying intents, retrain the model:
```bash
python train.py
```

### Styling the Chat Interface

- Modify `chatbot-frontend/src/components/ChatBox/BoxStyle.css` and `chatbot-frontend/src/components/ChatIcon/IconStyle.css` to match your website's design

## Deployment

### Backend Deployment

The project is configured for easy deployment to platforms like Heroku:
- `runtime.txt` specifies Python version
- `requirements.txt` lists all dependencies

### Frontend Deployment

- Build the frontend for production:
```bash
cd chatbot-frontend
npm run build
```
- The Flask app is configured to serve the built frontend files from the `chatbot-frontend/dist` directory

## Dependencies

### Backend
- Flask
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- dotenv
- requests

### Frontend
- React
- Marked (for Markdown rendering)
- Vite (build tool)

## Project Structure

```
├── app.py                       # Main Flask application
├── azure_api_bot.py             # Azure OpenAI API integration
├── chatbot-backend/             # Backend code
│   ├── Data/                    # Training data
│   │   └── intents.json         # Intent definitions
│   ├── dataset.py               # Data loading utilities
│   ├── main.py                  # Backend server
│   ├── model.py                 # PyTorch model definition
│   └── utils.py                 # Utility functions
├── chatbot-frontend/            # React frontend
│   ├── public/                  # Static files
│   └── src/                     # React components
│       ├── components/          # UI components
│       │   ├── ChatBox/         # Chat interface
│       │   └── ChatIcon/        # Chat button
│       ├── App.jsx              # Main React component
│       └── main.jsx             # React entry point
├── custom_bot.py                # Standalone bot implementation
├── train.py                     # Model training script
└── requirements.txt             # Python dependencies
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.