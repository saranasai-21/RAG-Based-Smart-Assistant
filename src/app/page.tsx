"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
      } else {
        alert(`Error: ${data.detail}`);
      }
    } catch (err) {
      alert("Failed to upload document.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.content, history: messages }),
      });
      const data = await res.json();
      
      if (res.ok) {
        setMessages((prev) => [
          ...prev, 
          { 
            role: "assistant", 
            content: data.answer, 
            sources: data.sources,
            intent: data.intent 
          }
        ]);
      } else {
        setMessages((prev) => [
          ...prev, 
          { role: "assistant", content: `Error: ${data.detail}` }
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev, 
        { role: "assistant", content: "Failed to connect to the server." }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Sidebar */}
      <div className="w-80 bg-gray-950 p-6 flex flex-col border-r border-gray-800">
        <h1 className="text-2xl font-bold mb-2 flex items-center gap-2">
          🧠 DocMind AI
        </h1>
        <p className="text-sm text-gray-400 mb-8">Advanced RAG System</p>
        
        <div className="flex-1">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
            Knowledge Base
          </h2>
          
          <div className="border-2 border-dashed border-gray-700 rounded-lg p-4 text-center hover:border-blue-500 transition-colors">
            <input 
              type="file" 
              id="file-upload" 
              className="hidden" 
              onChange={handleFileUpload} 
              disabled={isUploading}
            />
            <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
              <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="text-sm text-gray-300">
                {isUploading ? "Indexing..." : "Upload Document"}
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-500 flex-col">
              <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4">
                🧠
              </div>
              <h2 className="text-xl font-medium text-gray-300">Welcome to DocMind AI</h2>
              <p className="mt-2 text-center max-w-md">Upload a document on the left and ask me anything about it. I use advanced RAG to find exactly what you need.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] rounded-2xl p-4 ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-200"}`}>
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-gray-700">
                      <p className="text-xs text-gray-400 font-semibold mb-2">SOURCES</p>
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.map((s: any, i: number) => (
                          <span key={i} className="text-xs bg-gray-900 px-2 py-1 rounded-md text-gray-300 border border-gray-700 flex items-center gap-1">
                            📄 {s.file} (Pg {s.page}) 
                            <span className="text-green-400">{(s.relevance * 100).toFixed(0)}%</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {msg.intent && (
                    <div className="mt-2 flex gap-2">
                      <span className="text-[10px] uppercase tracking-wider bg-indigo-900/50 text-indigo-300 px-2 py-0.5 rounded border border-indigo-800">
                        {msg.intent} MODE
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-gray-800 rounded-2xl p-4 flex gap-2 items-center">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: "0.2s"}}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: "0.4s"}}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gray-900 border-t border-gray-800">
          <form onSubmit={handleChat} className="max-w-4xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-gray-800 text-gray-100 border border-gray-700 rounded-xl pl-4 pr-12 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              disabled={isTyping}
            />
            <button
              type="submit"
              disabled={isTyping || !input.trim()}
              className="absolute right-2 top-2 bottom-2 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center aspect-square"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
