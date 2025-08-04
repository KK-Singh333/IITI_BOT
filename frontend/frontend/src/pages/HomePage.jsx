import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence, easeOut } from "framer-motion";
import logo from "/src/assets/images/logo.svg";

export default function Homepage() {
  // const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [clicked, setClicked] = useState(false);
  const [showLogo, setShowLogo] = useState(false);
  const [animateLogo, setAnimateLogo] = useState(false);

  const navigate = useNavigate();

  const playSound = (src, volume = 0.5) => {
    const audio = new Audio(src);
    audio.volume = volume;
    audio.play();
  };

  // const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const handleGetStarted = () => {
    playSound("/sounds/click.mp3", 0.4);
    setClicked(true);

    setTimeout(() => {
      setShowLogo(true);
      playSound("/sounds/bounce.mp3", 0.6);

      setTimeout(() => {
        setAnimateLogo(true);
        playSound("/sounds/whoosh.mp3", 0.5);

        setTimeout(() => {
          // ✅ ONLY THIS: Let Layout handle chat creation
          navigate("/chatbot", { state: { createNewChat: true } });
        }, 2000);
      }, 1000);
    }, 800);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#000000] to-[#371F5A] text-white flex flex-col items-center justify-center relative overflow-hidden">

      {/* Hero Section */}
      <AnimatePresence>
        {!clicked && (
          <motion.div
            key="hero-content"
            initial={{ opacity: 1, y: 0 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 500, rotate: 20 }}
            transition={{ duration: 0.8, ease: "easeIn" }}
            className="flex flex-col items-center text-center"
          >
            <motion.img
              src={logo}
              alt="Logo"
              className="w-[13%] max-w-[1000px] min-w-[100px] mb-5"
            />
            <h2 className="mb-4 text-2xl font-semibold tracking-wider text-purple-500 md:text-4xl lg:text-5xl">
              CHAT WITH IITI BOT TO KNOW MORE
            </h2>
            <h1 className="mb-8 text-2xl font-bold tracking-wide text-white md:text-3xl lg:text-5xl">
              IIT INDORE
            </h1>
            <button
              onClick={handleGetStarted}
              className="rounded-lg border border-purple-500 px-8 py-3 text-white hover:border-purple-600 hover:bg-purple-600"
            >
              GET STARTED
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Logo Animation */}
      {showLogo && (
        <motion.div
          initial={{ y: -200, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 15 }}
          className="absolute flex items-center justify-center"
        >
          <motion.div
            initial={{ scale: 1, rotate: 0 }}
            animate={{
              scale: animateLogo ? 45 : 1,
              rotate: animateLogo ? 360 : 0,
            }}
            transition={{
              duration: 2,
              ease: easeOut,
            }}
            className="rounded-full overflow-hidden w-32 h-32 flex items-center justify-center relative"
          >
            <img
              src={logo}
              alt="Logo"
              className="absolute w-full h-full object-contain"
            />
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
