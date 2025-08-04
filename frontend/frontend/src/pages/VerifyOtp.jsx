import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

export default function VerifyOTP() {
  const [otp, setOtp] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  // If email is passed via location.state, use that
  useState(() => {
    if (location.state?.email) setEmail(location.state.email);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    try {
      const response = await fetch("http://localhost:8000/verify-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, otp }),
      });

      const result = await response.json();
      if (response.ok && result.success) {
        setSuccess("OTP verified successfully! Redirecting to login...");
        setTimeout(() => navigate("/login"), 2000);
      } else {
        setError(result.message || "OTP verification failed");
      }
    } catch (err) {
      setError("Failed to connect to the server");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-[#0e001d] via-[#1a0733] to-[#3d2171] text-white">
      <div className="flex justify-center items-center h-full py-20 px-4">
        <div className="max-w-md w-full bg-[#3a0066] p-8 rounded-xl shadow-2xl border border-white/20">
          <h2 className="text-2xl font-bold mb-6 text-center">Verify Your Email</h2>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {location.state?.email ? (
              <input
                type="email"
                value={email}
                readOnly
                className="p-3 rounded bg-gray-200 text-black placeholder-gray-600 cursor-not-allowed"
              />
            ) : (
              <input
                type="email"
                placeholder="Enter your email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="p-3 rounded bg-white text-black placeholder-gray-600"
              />
            )}

            <input
              type="text"
              placeholder="Enter OTP"
              maxLength={6}
              pattern="\d{6}"
              required
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="p-3 rounded bg-white text-black placeholder-gray-600"
            />

            <button
              type="submit"
              className="mt-4 bg-[#cc33ff] hover:bg-[#aa00cc] text-white font-semibold py-3 rounded"
            >
              Verify OTP
            </button>

            {error && <p className="text-red-400 text-sm text-center">{error}</p>}
            {success && <p className="text-green-400 text-sm text-center">{success}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}
