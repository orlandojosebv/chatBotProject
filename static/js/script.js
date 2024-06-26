function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput) {
        const chatBox = document.getElementById('chat-box');
        const userMessage = document.createElement('div');
        userMessage.textContent = "You: " + userInput;
        chatBox.appendChild(userMessage);

        fetch(`/get_response?user_input=${encodeURIComponent(userInput)}`)
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('div');
            botMessage.textContent = "Bot: " + data.response;
            chatBox.appendChild(botMessage);
            document.getElementById('user-input').value = '';
        });
    }
}
