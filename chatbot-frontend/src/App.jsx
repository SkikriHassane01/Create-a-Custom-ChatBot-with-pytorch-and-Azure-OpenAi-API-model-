import React, { useState } from 'react';
import ChatIcon from "./components/ChatIcon/ChatIcon";
import Chatbox from './components/ChatBox/Chatbox';
import './App.css'; 
import '../public/style.css';



function App() {
  const [isChatboxVisible, setIsChatboxVisible] = useState(false);

  const toggleChatbox = () => {
    setIsChatboxVisible(prevState => !prevState);
  };

  return (
    <div className="App">
      <ChatIcon onClick={toggleChatbox} />
      <Chatbox onClose={toggleChatbox} isVisible={isChatboxVisible} />
    </div>
  );
}

export default App;