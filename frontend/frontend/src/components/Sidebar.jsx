import { useEffect, useRef, useState } from "react";
import { FaPlus, FaTrash, FaEdit, FaBars, FaAngleRight } from "react-icons/fa";

export default function Sidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onRenameChat,
  onDeleteChat,
}) {
  const [search, setSearch] = useState("");
  const [renamingId, setRenamingId] = useState(null);
  const [renameText, setRenameText] = useState("");
  const [collapsed, setCollapsed] = useState(false);

  const chatRefs = useRef({});

  const filteredChats = chats.filter((chat) =>
    chat.title.toLowerCase().includes(search.toLowerCase())
  );

  // Scroll to active chat
  useEffect(() => {
    if (activeChatId && chatRefs.current[activeChatId]) {
      chatRefs.current[activeChatId].scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [activeChatId]);

  return (
    <div
      className={`${
        collapsed ? "w-16" : "w-72"
      } h-screen pt-16 bg-[#1b0d3a] border-r border-[#2d1d4a] text-white flex flex-col transition-all duration-300`}
    >
      {/* Top: New Chat + Search */}
      <div className="px-2 pb-4">
        <button
          onClick={onNewChat}
          className={`${
            collapsed ? "justify-center" : ""
          } w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded mb-4 flex items-center gap-2`}
          title={collapsed ? "New Chat" : ""}
        >
          <FaPlus />
          {!collapsed && "New Chat"}
        </button>

        {!collapsed && (
          <input
            type="text"
            placeholder="🔍 Search chats"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-3 py-2 bg-[#2d1d4a] border border-[#3f2f5a] rounded text-sm text-white placeholder-gray-400 focus:outline-none"
          />
        )}
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {filteredChats.map((chat) => (
          <div
            key={chat.id}
            ref={(el) => (chatRefs.current[chat.id] = el)}
            className={`group flex items-center justify-between px-2 py-2 rounded transition-all ${
              chat.id === activeChatId
                ? "bg-purple-700 ring-2 ring-purple-400"
                : "hover:bg-[#2d1d4a]"
            }`}
          >
            {renamingId === chat.id ? (
              <input
                type="text"
                className="w-full bg-[#2d1d4a] border border-purple-400 px-2 py-1 rounded text-sm text-white"
                value={renameText}
                onChange={(e) => setRenameText(e.target.value)}
                onBlur={() => {
                  onRenameChat(chat.id, renameText);
                  setRenamingId(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    onRenameChat(chat.id, renameText);
                    setRenamingId(null);
                  }
                }}
                autoFocus
              />
            ) : (
              <>
                <span
                  onClick={() => onSelectChat(chat.id)}
                  title={chat.title}
                  className={`flex-grow truncate text-sm cursor-pointer ${
                    collapsed ? "text-center" : ""
                  }`}
                >
                  {collapsed ? "🧠" : chat.title}
                </span>

                {!collapsed && (
                  <div className="flex items-center">
                    <button
                      onClick={() => {
                        setRenameText(chat.title);
                        setRenamingId(chat.id);
                      }}
                      title="Rename"
                      className="ml-2 text-xs text-gray-400 hover:text-white"
                    >
                      <FaEdit />
                    </button>

                    <button
                      onClick={() => {
                        const confirmDelete = confirm("Delete this chat?");
                        if (confirmDelete) onDeleteChat(chat.id);
                      }}
                      title="Delete"
                      className="ml-2 text-xs text-gray-400 hover:text-red-500"
                    >
                      <FaTrash />
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>

      {/* Collapse Toggle */}
      <div className="p-2 border-t border-[#2d1d4a] text-center">
        <button
          onClick={() => setCollapsed((prev) => !prev)}
          className="text-white hover:text-purple-400 w-full py-2"
          title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {collapsed ? <FaAngleRight size={18} /> : <FaBars size={18} />}
        </button>
      </div>
    </div>
  );
}
