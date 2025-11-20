const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const retrievedData = document.getElementById('retrieved-data');

const ALLOWED_FILE_TYPES = ['txt', 'pdf', 'csv'];

/* -----------------------------------------------------------
   1. App State
----------------------------------------------------------- */
let appState = {
    isProcessing: false,
    currentThinkingId: null
};

function updateState(updates) {
    appState = { ...appState, ...updates };
}

/* -----------------------------------------------------------
   2. Chat History (sessionStorage – cleared on refresh)
----------------------------------------------------------- */
let chatHistory = JSON.parse(sessionStorage.getItem("chatHistory") || "[]");

function saveChat() {
    sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
}

function renderChat() {
    chatMessages.innerHTML = "";

    chatHistory.forEach(msg => {
        const div = document.createElement('div');
        div.classList.add(
            "chat-message",
            msg.sender === "user" ? "user-message" : "bot-message"
        );
        div.textContent = msg.text;
        chatMessages.appendChild(div);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

renderChat();

/* -----------------------------------------------------------
   3. File Upload
----------------------------------------------------------- */
uploadBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    const file = files[0];
    const ext = file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_FILE_TYPES.includes(ext)) {
        alert(`Unsupported file type: ${ext}`);
        fileInput.value = "";
        return;
    }

    const form = new FormData();
    form.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: form
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.detail || data.message || "File upload failed");
        }

        alert(`File uploaded successfully: ${data.filename}`);

    } catch (error) {
        console.error("Upload error:", error);
        alert("File upload error: " + error.message);
    }
});

/* -----------------------------------------------------------
   4. Send Question → POST /ask
----------------------------------------------------------- */
sendBtn.addEventListener("click", async () => {

    if (appState.isProcessing) return;

    const question = userInput.value.trim();
    if (!question) return;

    updateState({ isProcessing: true });

    // Save user message
    chatHistory.push({ sender: "user", text: question });
    saveChat();
    renderChat();

    userInput.value = "";

    // Add temporary "Thinking..." message
    const thinkingId = "thinking-" + Date.now();
    updateState({ currentThinkingId: thinkingId });

    chatHistory.push({
        sender: "bot",
        id: thinkingId,
        text: "Thinking..."
    });
    saveChat();
    renderChat();

    try {
        const response = await fetch("http://127.0.0.1:8000/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ query: question })
        });

        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }

        const data = await response.json();

        if (!data.answer) {
            throw new Error("Invalid response: missing 'answer' field");
        }

        // Remove thinking message
        chatHistory = chatHistory.filter(msg => msg.id !== thinkingId);

        // Add bot answer
        chatHistory.push({
            sender: "bot",
            text: data.answer
        });

        //  NEW: Display retrieved chunks in textarea
        if (Array.isArray(data.context_chunks)) {
            const formatted = data.context_chunks
                .map((chunk, index) => `--- Fragment ${index + 1} ---\n${chunk}`)
                .join("\n\n");

            retrievedData.value = formatted;
        }

        saveChat();
        renderChat();

    } catch (err) {
        console.error("Error:", err);

        // Remove temporary message
        chatHistory = chatHistory.filter(msg => msg.id !== thinkingId);

        chatHistory.push({
            sender: "bot",
            text: "Error communicating with backend."
        });

        saveChat();
        renderChat();

    } finally {
        updateState({ isProcessing: false });
    }
});