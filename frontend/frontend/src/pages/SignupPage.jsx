import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "/src/assets/images/logo.svg";

export default function Signup() {
  const [formData, setFormData] = useState({});
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    try {
      const response = await fetch("http://localhost:8000/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (response.ok) {
        localStorage.setItem("pendingSignupEmail", formData.email);
        navigate("/VerifyOtp");
      } else {
        setError(result.detail || "Signup failed");
      }
    } catch (err) {
      setError("Error connecting to server.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-[#0e001d] via-[#1a0733] to-[#3d2171] text-white">
      <div className="flex flex-wrap justify-center items-start gap-10 px-10 py-20">
        <div className="w-full max-w-xl bg-[#3a0066] p-10 rounded-xl border border-white/20 shadow-2xl">
          <h2 className="text-2xl font-bold mb-6">Create Your Account</h2>

          <form autoComplete="on" className="flex flex-col gap-4" onSubmit={handleSubmit}>
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="text"
                name="Name"
                placeholder="Full Name"
                required
                autoComplete="name"
                onChange={handleChange}
                className="flex-1 p-3 rounded bg-white text-black placeholder-gray-600"
              />
              <input
                type="email"
                name="email"
                placeholder="Email"
                required
                autoComplete="email"
                onChange={handleChange}
                className="flex-1 p-3 rounded bg-white text-black placeholder-gray-600"
              />
            </div>

            <input
              type="tel"
              name="phone"
              placeholder="Mobile Number"
              pattern="[0-9]{10}"
              required
              onChange={handleChange}
              className="p-3 rounded bg-white text-black placeholder-gray-600"
            />

            <input
              type="text"
              name="department"
              placeholder="Department"
              required
              onChange={handleChange}
              className="p-3 rounded bg-white text-black placeholder-gray-600"
            />

            <select
              name="member_type"
              required
              onChange={handleChange}
              className="p-3 rounded bg-white text-black"
            >
              <option value="">Select Member Type</option>
              <option value="Student">Student</option>
              <option value="Faculty">Faculty</option>
              <option value="Staff">Staff</option>
              <option value="Visitor">Visitor</option>
            </select>

            <input
              type="password"
              name="password"
              placeholder="Password"
              required
              onChange={handleChange}
              className="p-3 rounded bg-white text-black placeholder-gray-600"
            />
            <input
              type="password"
              name="confirm_password"
              placeholder="Confirm Password"
              required
              onChange={handleChange}
              className="p-3 rounded bg-white text-black placeholder-gray-600"
            />

            <button
              type="submit"
              className="mt-4 bg-[#cc33ff] hover:bg-[#aa00cc] text-white font-semibold py-3 rounded"
            >
              Register
            </button>

            {error && <p className="text-red-400 text-sm text-center">{error}</p>}
            {success && <p className="text-green-400 text-sm text-center">{success}</p>}

            <p className="text-center text-sm mt-4">
              Already have an account?{" "}
              <Link to="/login" className="text-purple-300 underline">
                Login
              </Link>
            </p>
          </form>
        </div>

        {/* Description Section */}
        <div className="w-full max-w-xl bg-[#400080] p-10 rounded-xl border border-white/20 shadow-2xl text-gray-200">
          <h2 className="text-xl font-bold mb-4">Welcome to IITI Bot</h2>
          <p>
            IITI Bot is an intelligent platform crafted for IIT Indore members and learners across India.
            Whether you're a student, researcher, or staff — this AI-powered bot helps you interact with services and
            resources more effectively.
          </p>
        </div>
      </div>
    </div>
  );
}