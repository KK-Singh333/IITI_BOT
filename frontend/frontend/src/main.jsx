import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App from "./App";
import HomePage from "./pages/HomePage";
import ChatbotPage from "./pages/ChatbotPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import About from "./pages/About";
import VerifyOtp from "./pages/VerifyOtp"; // ✅ Import OTP Page

import Layout from "./components/Layout";
import { AuthProvider } from "./context/AuthContext";

import "./index.css";

// Define the router
const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "login", element: <LoginPage /> },
      { path: "signup", element: <SignupPage /> },
      { path: "about", element: <About /> },
      { path: "VerifyOtp", element: <VerifyOtp /> }, // ✅ OTP Route
      {
        path: "chatbot",
        element: <Layout />,
        children: [
          { index: true, element: <ChatbotPage /> },
        ],
      },
    ],
  },
]);

// Render the app with AuthContext and Router
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  </React.StrictMode>
);
