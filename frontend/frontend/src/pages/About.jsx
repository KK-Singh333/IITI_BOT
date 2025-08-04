import { useState } from "react";
import { Link } from "react-router-dom";
import logo from "/src/assets/images/logo.svg";

// Team SVGs
import Khush from "/src/assets/images/khush.svg";
import Rudra from "/src/assets/images/chitkara.svg";
import Nikesh from "/src/assets/images/srinikesh.svg";
import Aditya from "/src/assets/images/Aditya_rai.svg";
import Poorvansh from "/src/assets/images/poorvansh.svg";
import Snehith from "/src/assets/images/snehith.svg";

export default function About() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const teamMembers = [
    { name: "KHUSH KUMAR SINGH", img: Khush, ig: "khush", gh: "khush" },
    { name: "RUDRA CHITKARA", img: Rudra, ig: "rudra", gh: "rudra" },
    { name: "AYINALA SRI NIKESH", img: Nikesh, ig: "ayinala", gh: "ayinala" },
    { name: "ADITYA RAI", img: Aditya, ig: "aditya", gh: "aditya" },
    {
      name: "POORVANSH DASHORE",
      img: Poorvansh,
      ig: "poorvansh",
      gh: "poorvansh",
    },
    { name: "BUDDE SNEHITH", img: Snehith, ig: "snehith", gh: "snehith" },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-r from-[#0e001d] via-[#1a0733] to-[#3d2171] text-white pb-20">

      {/* About Section */}
      <div className="pt-28 px-8 md:px-24 max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-6">About IITI Bot</h1>
        <p className="mb-4">
          <strong>IITI Bot</strong> is your all-in-one assistant for exploring
          everything about
          <strong> IIT Indore</strong> — from academics, research, and clubs to
          sports, fests, tech events, hostel life, and more.
        </p>
        <p className="mb-8">
          Designed for students, faculty, aspirants, and anyone curious about
          IIT Indore, this bot provides accurate, real-time answers to questions
          across domains using natural language conversations.
        </p>

        <h2 className="text-xl font-semibold text-purple-300 mt-10 mb-2">
          What Can IITI Bot Do?
        </h2>
        <ul className="list-disc list-inside space-y-2 text-gray-200">
          <li>Answer queries about departments, courses, and academics</li>
          <li>Provide details on campus life, events, and hostel facilities</li>
          <li>Fetch updates on fests, sports activities, and student clubs</li>
          <li>Assist new joiners and aspirants with relevant information</li>
        </ul>

        <h2 className="text-xl font-semibold text-purple-300 mt-10 mb-2">
          How It Works
        </h2>
        <p className="mb-4">
          IITI Bot is powered by a{" "}
          <strong>Retrieval-Augmented Generation (RAG)</strong> architecture
          using the <strong>Pathway</strong> stream processing framework for
          real-time web search and knowledge retrieval. It integrates:
        </p>
        <ul className="list-disc list-inside space-y-2 text-gray-200">
          <li>
            <strong>Flask</strong> – Lightweight Python backend framework
          </li>
          <li>
            <strong>HTML, CSS, JS</strong> – For responsive and dynamic UI
          </li>
          <li>
            <strong>LLMs</strong> – To understand and generate accurate answers
          </li>
          <li>
            <strong>Live web tools</strong> – To keep responses up-to-date
          </li>
        </ul>

        <h2 className="text-xl font-semibold text-purple-300 mt-10 mb-2">
          Contact Us
        </h2>
        <p>
          Got suggestions or questions? Drop us an email at:{" "}
          <a
            href="mailto:iiti_bot@example.com"
            className="text-blue-300 underline"
          >
            iiti_bot@example.com
          </a>
        </p>
      </div>

      {/* Team Section */}
      <div className="mt-16 px-6 sm:px-10 lg:px-24">
        <h2 className="text-center text-2xl font-bold text-white mb-8">
          Meet Our Team
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-10 justify-items-center">
          {teamMembers.map((member) => (
            <div key={member.name} className="text-center">
              <img
                src={member.img}
                alt={member.name}
                className="w-36 h-36 rounded-full object-cover border-4 border-white mb-4 mx-auto"
              />
              <div className="font-semibold text-lg">{member.name}</div>
              <div className="flex justify-center gap-4 mt-2 text-2xl">
                <a
                  href={`https://instagram.com/${member.ig}`}
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-pink-500"
                >
                  <i className="fab fa-instagram"></i>
                </a>
                <a
                  href={`https://github.com/${member.gh}`}
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-gray-400"
                >
                  <i className="fab fa-github"></i>
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
