// Configuration
let apiUrl = 'http://localhost:8080';
let isConnected = false;
let chatSessions = [];
let currentSessionId = null;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const chatHistoryList = document.getElementById('chatHistory');

// State
let emptyStateShown = true;

const userIconSVG = `<svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
const botIconSVG = `<svg stroke="currentColor" fill="none" stroke-width="1.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`;

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener('input', () => {
    adjustInputHeight();
    if (messageInput.value.trim().length > 0) {
        sendBtn.classList.add('active');
    } else {
        sendBtn.classList.remove('active');
    }
});

// Initialize
window.addEventListener('load', () => {
    const savedUrl = localStorage.getItem('apiUrl');
    if (savedUrl) apiUrl = savedUrl;
    
    loadSessions();
    checkConnection();
    
    // Always start a new chat when opening a tab, unless the top chat is already empty
    if (chatSessions.length === 0 || chatSessions[0].messages.length > 0) {
        createNewSession();
    } else {
        loadSession(chatSessions[0].id);
    }
});

function adjustInputHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

async function checkConnection() {
    try {
        const response = await fetch(`${apiUrl}/health`, { method: 'GET' });
        setConnectionStatus(response.ok);
    } catch (error) {
        setConnectionStatus(false);
    }
}

function setConnectionStatus(connected) {
    isConnected = connected;
    if (connected) {
        statusIndicator.classList.add('connected');
        statusIndicator.classList.remove('error');
        statusText.textContent = 'Đã Kết Nối';
    } else {
        statusIndicator.classList.remove('connected');
        statusIndicator.classList.add('error');
        statusText.textContent = 'Ngoại Tuyến';
    }
}

function removeEmptyState() {
    if (emptyStateShown) {
        const emptyState = chatMessages.querySelector('.empty-state');
        if (emptyState) emptyState.remove();
        emptyStateShown = false;
    }
}

// Session Management
function loadSessions() {
    const saved = localStorage.getItem('chatSessions');
    chatSessions = saved ? JSON.parse(saved) : [];
    renderHistory();
}

function saveSessions() {
    localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    renderHistory();
}

function createNewSession() {
    const id = Date.now().toString();
    const newSession = {
        id: id,
        title: 'Cuộc trò chuyện mới',
        messages: [],
        timestamp: new Date().toISOString()
    };
    chatSessions.unshift(newSession);
    saveSessions();
    loadSession(id);
}

function loadSession(id) {
    currentSessionId = id;
    const session = chatSessions.find(s => s.id === id);
    if (!session) return;

    chatMessages.innerHTML = '';
    if (session.messages.length === 0) {
        showEmptyState();
    } else {
        emptyStateShown = false;
        session.messages.forEach(msg => {
            addMessageToUI(msg.content, msg.sender, false, msg.mapData);
        });
    }
    renderHistory();
}

function showEmptyState() {
    chatMessages.innerHTML = `
        <div class="empty-state">
            <h1>Chào bạn, tôi có thể giúp gì cho bạn?</h1>
            <p>Hãy đặt câu hỏi về triệu chứng hoặc dinh dưỡng để tôi hỗ trợ.</p>
        </div>
    `;
    emptyStateShown = true;
}

function renderHistory() {
    chatHistoryList.innerHTML = '';
    chatSessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
        item.innerHTML = `
            <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="14" width="14" xmlns="http://www.w3.org/2000/svg"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            <span class="history-title">${session.title}</span>
            <button class="delete-history" onclick="event.stopPropagation(); deleteSession('${session.id}')">
                <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="14" width="14" xmlns="http://www.w3.org/2000/svg"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        `;
        item.onclick = () => loadSession(session.id);
        chatHistoryList.appendChild(item);
    });
}

function deleteSession(id) {
    chatSessions = chatSessions.filter(s => s.id !== id);
    saveSessions();
    if (currentSessionId === id) {
        if (chatSessions.length > 0) {
            loadSession(chatSessions[0].id);
        } else {
            createNewSession();
        }
    }
}

function newChat() {
    createNewSession();
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    removeEmptyState();
    addMessageToUI(message, 'user');
    
    // Update session title if it's the first message
    const session = chatSessions.find(s => s.id === currentSessionId);
    if (session && session.messages.length === 1) {
        session.title = message.substring(0, 30) + (message.length > 30 ? '...' : '');
        saveSessions();
    }

    messageInput.value = '';
    adjustInputHeight();
    sendBtn.classList.remove('active');
    sendBtn.disabled = true;

    const typingId = 'typing-' + Date.now();
    addTypingIndicator(typingId);

    try {
        const response = await fetch(`${apiUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, session_id: currentSessionId }),
        });

        removeTypingIndicator(typingId);
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const data = await response.json();
        const responseText = formatBotResponse(data);
        addMessageToUI(responseText, 'bot', true, data.map_data);
        
    } catch (error) {
        removeTypingIndicator(typingId);
        const errorMsg = '❌ Lỗi kết nối máy chủ. Hãy đảm bảo Backend đang chạy.';
        addMessageToUI(errorMsg, 'bot');
    } finally {
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

function addMessageToUI(content, sender, save = true, mapData = null) {
    if (save) {
        const session = chatSessions.find(s => s.id === currentSessionId);
        if (session) {
            session.messages.push({ content, sender, timestamp: new Date().toISOString(), mapData });
            saveSessions();
        }
    }

    const row = document.createElement('div');
    row.className = `message-row ${sender}`;

    const wrapper = document.createElement('div');
    wrapper.className = 'message-content-wrapper';

    const avatar = document.createElement('div');
    avatar.className = `avatar ${sender}-avatar`;
    avatar.innerHTML = sender === 'user' ? userIconSVG : botIconSVG;

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text markdown-body';
    textDiv.innerHTML = parseMessageContent(content);

    // Nếu có dữ liệu bản đồ, tạo khung chứa
    if (mapData && mapData.length > 0) {
        const mapId = 'map-' + Math.random().toString(36).substr(2, 9);
        const mapDiv = document.createElement('div');
        mapDiv.id = mapId;
        mapDiv.className = 'map-container';
        textDiv.appendChild(mapDiv);

        // Đợi DOM cập nhật rồi vẽ bản đồ
        setTimeout(() => {
            const map = L.map(mapId).setView([mapData[0].lat, mapData[0].lon], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap'
            }).addTo(map);

            mapData.forEach(loc => {
                L.marker([loc.lat, loc.lon]).addTo(map)
                    .bindPopup(loc.name)
                    .openPopup();
            });
        }, 100);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(textDiv);
    row.appendChild(wrapper);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatBotResponse(data) {
    if (typeof data === 'string') return data;
    if (data.response) return data.response;
    
    let parts = [];
    if (data.qa) parts.push(data.qa);
    if (data.diet) parts.push(`🥗 **Lời khuyên:** ${data.diet}`);
    
    return parts.join('\n\n') || 'Tôi không hiểu câu hỏi của bạn.';
}

function addTypingIndicator(id) {
    const row = document.createElement('div');
    row.className = `message-row bot`;
    row.id = id;
    row.innerHTML = `
        <div class="message-content-wrapper">
            <div class="avatar bot-avatar">${botIconSVG}</div>
            <div class="message-text">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function parseMessageContent(content) {
    return content
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/• (.*?)(?=<br>|$)/g, '<li>$1</li>');
}
