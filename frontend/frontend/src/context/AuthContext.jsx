import React, { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(null);
  const [userEmail, setUserEmail] = useState(null);
  const [userName, setUserName] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Load from localStorage on first mount
  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    const storedEmail = localStorage.getItem("userEmail");

    if (storedToken) {
      setToken(storedToken);
      setIsLoggedIn(true);
    }

    if (storedEmail) {
      setUserEmail(storedEmail);
      setUserName(storedEmail.split("@")[0]);
    }
  }, []);

  const updateToken = (newToken) => {
    setToken(newToken);
    localStorage.setItem("token", newToken);
    setIsLoggedIn(true);
  };

  const setUserId = (email) => {
    setUserEmail(email);
    localStorage.setItem("userEmail", email);
    setUserName(email.split("@")[0]);
  };

  const logout = () => {
    setToken(null);
    setUserEmail(null);
    setUserName("");
    setIsLoggedIn(false);
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
  };

  return (
    <AuthContext.Provider
      value={{
        token,
        setToken: updateToken,
        userEmail,
        userName,
        isLoggedIn,
        setIsLoggedIn,
        setUserId,
        setUserName,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
