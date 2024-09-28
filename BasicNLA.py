import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a dictionary to store the conversation flow
conversation_flow = {
    'hello': 'Hi, how are you?',
    'how are you': 'I am good, thanks!',
    'what is your name': 'My name is Juh.'
}

# Define a function to process user input
def process_input(input_text):
    # Tokenize the input text
    tokens = nltk.word_tokenize(input_text)

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Check if the input matches a conversation flow
    for token in lemmatized_tokens:
        if token in conversation_flow:
            return conversation_flow[token]

    # If no match is found, return a default response
    return 'I did not understand that.'

# Define a function to start the conversation
def start_conversation():
    print('Hello, I am Juh. How can I help you?')

    while True:
        # Get user input
        user_input = input('> ')

        # Process the user input
        response = process_input(user_input)

        # Print the response
        print(response)

# Start the conversation
start_conversation()
