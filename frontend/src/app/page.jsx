// [수정 사항]
// Next.js App Router 환경에서 useState, useEffect와 같은 클라이언트 측 Hook을 사용하기 위해
// 파일 최상단에 "use client" 지시어를 추가해야 합니다.
"use client";

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

// 아이콘 SVG 컴포넌트
const PaperAirplaneIcon = (props) => (
  <svg
    {...props}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
    />
  </svg>
);

// 컴포넌트 이름을 ChatPage로 변경하여 페이지 역할을 명확히 할 수 있습니다. (선택 사항)
// 이 파일을 app/chat/page.js 와 같이 저장하면 /chat 경로에서 이 페이지가 렌더링됩니다.
const ChatPage = () => {
  // 사용 가능한 모델 목록
  const models = [
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-o4",
    "Qwen2.5",
  ];

  // 상태 관리
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "안녕하세요! 모델을 선택하고 대화를 시작하세요.",
      sender: "bot",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [previousResponseId, setPreviousResponseId] = useState(null);
  const [selectedModel, setSelectedModel] = useState(models[0]);

  // Ref 참조
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // 메시지 목록이 업데이트될 때마다 스크롤을 맨 아래로 이동시키는 효과
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // 로딩 상태가 끝날 때마다 입력창에 포커스를 주기 위한 useEffect
  useEffect(() => {
    // 로딩이 false일 때 (즉, API 응답이 완료되었을 때) 포커스를 줍니다.
    if (!loading) {
      inputRef.current?.focus();
    }
  }, [loading]);

  // 사용자의 입력을 처리하는 함수
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // 챗봇의 응답을 가져오는 비동기 함수 (axios 사용)
  const fetchBotResponse = async (userInput) => {
    console.log(
      `챗봇 응답 요청 시작. 모델: ${selectedModel}, 이전 응답 ID: ${previousResponseId}`
    );
    setError(null);

    try {
      // --- API 호출 시뮬레이션 시작 (이 부분은 실제 연동 시 실제 코드로 교체) ---
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const data = {
        response_message: `[${selectedModel} 모델 응답] '${userInput}'라고 하셨네요. 이 메시지는 시뮬레이션된 응답입니다.`,
        previous_response_id: `id_${Math.random()
          .toString(36)
          .substring(2, 9)}`,
      };
      // --- API 호출 시뮬레이션 종료 ---

      console.log("API 응답 수신:", data);

      const botMessage = {
        id: Date.now(),
        text: data.response_message,
        sender: "bot",
      };

      setMessages((prevMessages) => [...prevMessages, botMessage]);
      setPreviousResponseId(data.previous_response_id);
    } catch (e) {
      const errorMessage = {
        id: Date.now(),
        text: "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
        sender: "bot",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      setError(e);
      console.error("챗봇 응답 요청 실패:", e);
    } finally {
      setLoading(false);
      console.log("요청 처리 완료. loading 상태:", false);
    }
  };

  // 메시지 전송 처리 함수
  const handleSendMessage = (e) => {
    e.preventDefault();
    const trimmedInput = inputValue.trim();

    if (!trimmedInput || loading) {
      return;
    }

    const userMessage = {
      id: Date.now(),
      text: trimmedInput,
      sender: "user",
    };

    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInputValue("");
    setLoading(true);

    fetchBotResponse(trimmedInput);
  };

  return (
    // Next.js에서는 <html>, <body> 태그가 자동으로 생성되므로 최상위 div부터 시작합니다.
    // h-screen을 사용하기 위해 상위 레이아웃(app/layout.js)의 html, body 태그에 h-full 클래스를 추가하는 것이 좋습니다.
    <div className="flex h-screen bg-gray-100">
      <aside className="w-60 flex-shrink-0 bg-gray-800 text-white flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold">모델 선택</h2>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          <ul>
            {models.map((model) => (
              <li
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`block px-4 py-2 rounded-md cursor-pointer text-sm transition-colors ${
                  selectedModel === model
                    ? "bg-blue-600 font-semibold"
                    : "hover:bg-gray-700"
                }`}
              >
                {model}
              </li>
            ))}
          </ul>
        </nav>
      </aside>
      {/* 오른쪽 메인 채팅 영역 */}
      <div className="flex-1 flex flex-col">
        <header className="bg-white border-b border-gray-200 p-4 shadow-sm">
          <h1 className="text-xl font-semibold text-gray-800 text-center">
            AI 챗봇
          </h1>
        </header>

        <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`items-baseline flex gap-2 ${
                message.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.sender === "bot" && (
                <div className="w-8 h-8 rounded-full bg-gray-500 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
                  AI
                </div>
              )}
              <div
                className={`rounded-2xl p-3 max-w-sm md:max-w-md shadow-md ${
                  message.sender === "user"
                    ? "bg-blue-500 text-white rounded-br-lg"
                    : "bg-white text-gray-800 rounded-bl-lg"
                }`}
              >
                <p className="text-sm">{message.text}</p>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-2 justify-start">
              <div className=" w-8 h-8 rounded-full bg-gray-500 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
                AI
              </div>
              <div className="rounded-2xl p-3 max-w-sm md:max-w-md shadow-md bg-white text-gray-800 rounded-bl-lg">
                <div className="flex items-center space-x-1">
                  <span className="text-sm text-gray-500">입력 중</span>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="text-red-500 text-center text-sm">
              오류: {error.message}
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>

        <footer className="bg-white border-t border-gray-200 p-4">
          <form
            onSubmit={handleSendMessage}
            className="flex items-center space-x-2"
          >
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={handleInputChange}
              placeholder="메시지를 입력하세요..."
              className="flex-1 w-full px-4 py-2 text-sm bg-gray-100 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !inputValue.trim()}
              className="p-2 rounded-full text-white bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              <PaperAirplaneIcon className="w-6 h-6" />
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
};

export default ChatPage;
