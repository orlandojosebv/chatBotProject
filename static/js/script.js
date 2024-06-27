function sendMessage() {
    const userInput = document.getElementById('user-input');
    const userText = userInput.value.trim();
    if (userText === "") return;

    appendMessage('user', userText);
    userInput.value = "";

    fetch(`/get_response?user_input=${userText}`)
        .then(response => response.json())
        .then(data => {
            appendMessage('bot', data.response);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function appendMessage(sender, text) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    const messageTextDiv = document.createElement('div');
    messageTextDiv.classList.add('text');
    messageTextDiv.textContent = text;
    messageDiv.appendChild(messageTextDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
