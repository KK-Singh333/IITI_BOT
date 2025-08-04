import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext"; // ✅
import logo from "/src/assets/images/logo.svg";

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const { pathname } = useLocation();
  const { isLoggedIn, userId, setIsLoggedIn, setUserId } = useAuth(); // ✅

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUserId(null);
    localStorage.removeItem("userEmail");
  };

  return (
    <>
      {/* Desktop & Main Nav */}
      <div className="absolute top-0 left-0 w-full z-50 flex items-center justify-between bg-[#1b0d3a] p-4">
        <div className="flex items-center gap-2">
          <img src={logo} alt="Logo" className="h-8 w-8 rounded-lg md:ml-10" />
          <h1 className="text-sm font-bold tracking-widest text-white">IITI BOT</h1>
        </div>

        <div className="md:hidden flex items-center">
          <button onClick={toggleMenu} className="text-2xl text-white hover:text-purple-300 mr-4">
            &#9776;
          </button>
        </div>

        <div className="hidden md:flex items-center gap-6 mr-10">
          <Link to="/" className={`hover:text-purple-300 ${pathname === "/" ? "text-purple-300" : "text-white"}`}>Home</Link>
          <Link to="/about" className={`hover:text-purple-300 ${pathname === "/about" ? "text-purple-300" : "text-white"}`}>About</Link>

          {isLoggedIn ? (
            <>
              <span className="text-purple-300 text-sm">👋 Hello, {userId?.split("@")[0]}</span>
              <button
                onClick={handleLogout}
                className="rounded border border-white px-3 py-1 text-sm text-white hover:bg-white hover:text-black"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="rounded border border-white px-3 py-1 text-sm text-white hover:bg-white hover:text-black">
                Log In
              </Link>
              <Link to="/signup" className="rounded bg-purple-500 px-4 py-1 text-sm text-white hover:bg-purple-600">
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>

      {/* Mobile Nav */}
      {isMenuOpen && (
        <div className="absolute top-16 right-4 flex flex-col items-end gap-4 bg-[#1b0d3a] px-4 py-3 rounded-md shadow-lg md:hidden z-50">
          <Link to="/" className="text-white hover:text-purple-300">
            Home
          </Link>
          <Link to="/about" className="text-white hover:text-purple-300">
            About
          </Link>

          {isLoggedIn ? (
            <>
              <span className="text-purple-300 text-sm">👋 {userId?.split("@")[0]}</span>
              <button
                onClick={handleLogout}
                className="text-white rounded border border-white px-3 py-1 text-sm hover:bg-white hover:text-black"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="text-white rounded border border-white px-3 py-1 text-sm hover:bg-white hover:text-black">
                Log In
              </Link>
              <Link to="/signup" className="rounded bg-purple-500 px-4 py-1 text-sm text-white hover:bg-purple-600">
                Sign Up
              </Link>
            </>
          )}
        </div>
      )}
    </>
  );
}
