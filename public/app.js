document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const fileList = document.getElementById('file-list');
    const docCount = document.getElementById('doc-count');
    const chunkCount = document.getElementById('chunk-count');
    const statusBadge = document.getElementById('status-badge');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const welcomeCard = document.getElementById('welcome-card');

    let history = [];
    let isReady = false;

    // Handle file selection
    fileUpload.addEventListener('change', () => {
        const files = fileUpload.files;
        if (files.length > 0) {
            uploadBtn.disabled = false;
            fileList.innerHTML = '';
            Array.from(files).forEach(file => {
                const ext = file.name.split('.').pop().toLowerCase();
                let icon = '📄';
                if (['pdf'].includes(ext)) icon = '📕';
                if (['doc', 'docx'].includes(ext)) icon = '📘';
                if (['md', 'markdown'].includes(ext)) icon = '📝';
                
                const el = document.createElement('div');
                el.className = 'file-item';
                el.innerHTML = `<span>${icon}</span><span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${file.name}</span>`;
                fileList.appendChild(el);
            });
        } else {
            uploadBtn.disabled = true;
            fileList.innerHTML = '';
        }
    });

    // Handle upload
    uploadBtn.addEventListener('click', async () => {
        const files = fileUpload.files;
        if (!files.length) return;

        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Indexing...';
        statusBadge.textContent = '○ Indexing...';
        statusBadge.className = 'status-badge status-idle';

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');

            docCount.textContent = data.count;
            chunkCount.textContent = data.chunks;
            statusBadge.textContent = '● Active';
            statusBadge.className = 'status-badge status-ready';
            
            chatInput.disabled = false;
            sendBtn.disabled = false;
            isReady = true;
            uploadBtn.textContent = 'Upload & Index';
            fileUpload.value = '';
            
            appendMessage('bot', `Successfully indexed ${data.chunks} chunks from ${data.count} documents! You can now ask questions.`);
        } catch (err) {
            alert(err.message);
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload & Index';
            statusBadge.textContent = '○ Error';
        }
    });

    // Handle chat
    const sendMessage = async () => {
        if (!isReady) return;
        const query = chatInput.value.trim();
        if (!query) return;

        chatInput.value = '';
        chatInput.style.height = 'auto';
        appendMessage('user', query);
        sendBtn.disabled = true;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, history })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Chat failed');

            history.push({ role: 'user', content: query });
            history.push({ role: 'assistant', content: data.response });

            appendMessage('bot', data.response, data.sources);
        } catch (err) {
            appendMessage('bot', `Error: ${err.message}`);
        } finally {
            sendBtn.disabled = false;
        }
    };

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    function appendMessage(sender, text, sources = []) {
        if (welcomeCard) welcomeCard.style.display = 'none';

        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${sender}`;
        
        let html = `<p style="white-space: pre-wrap">${text}</p>`;
        
        if (sources && sources.length > 0) {
            html += `<div class="sources-list">
                <h4>Sources:</h4>
                <ul>
                    ${sources.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>`;
        }

        msgDiv.innerHTML = html;
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // PRO Features Logic
    const proToggle = document.getElementById('pro-toggle');
    const proFeatures = document.getElementById('pro-features');
    const featureBtns = document.querySelectorAll('.feature-btn');

    proToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            proFeatures.style.display = 'flex';
            // Enable all features by default when PRO is turned on
            featureBtns.forEach(btn => btn.classList.add('active'));
        } else {
            proFeatures.style.display = 'none';
        }
    });

    featureBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
        });
    });
});
