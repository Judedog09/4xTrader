import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Zap, Mail, Lock, LogOut } from 'lucide-react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// ══════════════════════════════════════════════════════════════════════════════
// AUTH PAGES
// ══════════════════════════════════════════════════════════════════════════════

const AuthPage = ({ onLogin }) => {
  const [isSignup, setIsSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [alpacaKey, setAlpacaKey] = useState('');
  const [alpacaSecret, setAlpacaSecret] = useState('');
  const [alpacaUrl, setAlpacaUrl] = useState('https://paper-api.alpaca.markets');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isSignup) {
        await axios.post(`${API}/auth/signup`, {
          email, password, alpaca_key: alpacaKey, alpaca_secret: alpacaSecret, alpaca_base_url: alpacaUrl
        });
        // Auto-login after signup
        const loginRes = await axios.post(`${API}/auth/login`, { email, password });
        onLogin(loginRes.data.token);
      } else {
        const res = await axios.post(`${API}/auth/login`, { email, password });
        onLogin(res.data.token);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center', background:'linear-gradient(135deg, #0a0f1a 0%, #1a1f2e 100%)', padding:20 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .auth-input {
          width: 100%;
          padding: 14px 16px;
          background: #1a2332;
          border: 1px solid #2d3748;
          border-radius: 10px;
          color: #e2e8f0;
          font-size: 14px;
          outline: none;
          transition: all 0.2s;
        }
        .auth-input:focus { border-color: #00ffbb; box-shadow: 0 0 0 3px rgba(0,255,187,0.1); }
        .auth-btn {
          width: 100%;
          padding: 14px;
          background: #00ffbb;
          border: none;
          border-radius: 10px;
          color: #000;
          font-weight: 800;
          font-size: 15px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .auth-btn:hover { background: #00e6a8; transform: translateY(-1px); }
        .auth-btn:disabled { background: #2d3748; color: #666; cursor: not-allowed; }
      `}</style>

      <div style={{ width:'100%', maxWidth:440, background:'#0d1117', borderRadius:20, padding:'48px 40px', border:'1px solid #21262d', boxShadow:'0 20px 60px rgba(0,0,0,0.4)' }}>
        {/* Logo */}
        <div style={{ textAlign:'center', marginBottom:36 }}>
          <div style={{ display:'inline-flex', alignItems:'center', gap:10, marginBottom:12 }}>
            <Zap size={32} color="#00ffbb" fill="#00ffbb"/>
            <span style={{ fontSize:28, fontWeight:900, color:'#00ffbb', letterSpacing:'-0.5px' }}>PULSE 4X</span>
          </div>
          <div style={{ fontSize:13, color:'#3d4f6e', fontWeight:600, letterSpacing:'2px', textTransform:'uppercase' }}>
            Titan Trading System
          </div>
        </div>

        {/* Tab selector */}
        <div style={{ display:'flex', gap:8, marginBottom:32, background:'#161b22', borderRadius:10, padding:4 }}>
          <button onClick={() => setIsSignup(false)}
            style={{ flex:1, padding:'10px', borderRadius:7, border:'none', background:!isSignup?'#00ffbb':'transparent', color:!isSignup?'#000':'#666', fontWeight:700, fontSize:13, cursor:'pointer', transition:'all 0.2s' }}>
            Login
          </button>
          <button onClick={() => setIsSignup(true)}
            style={{ flex:1, padding:'10px', borderRadius:7, border:'none', background:isSignup?'#00ffbb':'transparent', color:isSignup?'#000':'#666', fontWeight:700, fontSize:13, cursor:'pointer', transition:'all 0.2s' }}>
            Sign Up
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom:16 }}>
            <label style={{ display:'block', fontSize:12, fontWeight:700, color:'#3d4f6e', marginBottom:8, textTransform:'uppercase', letterSpacing:'1px' }}>Email</label>
            <div style={{ position:'relative' }}>
              <Mail size={18} style={{ position:'absolute', left:16, top:'50%', transform:'translateY(-50%)', color:'#3d4f6e' }}/>
              <input type="email" required value={email} onChange={e=>setEmail(e.target.value)} className="auth-input" style={{ paddingLeft:46 }} placeholder="you@example.com"/>
            </div>
          </div>

          <div style={{ marginBottom:16 }}>
            <label style={{ display:'block', fontSize:12, fontWeight:700, color:'#3d4f6e', marginBottom:8, textTransform:'uppercase', letterSpacing:'1px' }}>Password</label>
            <div style={{ position:'relative' }}>
              <Lock size={18} style={{ position:'absolute', left:16, top:'50%', transform:'translateY(-50%)', color:'#3d4f6e' }}/>
              <input type="password" required value={password} onChange={e=>setPassword(e.target.value)} className="auth-input" style={{ paddingLeft:46 }} placeholder="••••••••"/>
            </div>
          </div>

          {isSignup && (
            <>
              <div style={{ marginBottom:16 }}>
                <label style={{ display:'block', fontSize:12, fontWeight:700, color:'#3d4f6e', marginBottom:8, textTransform:'uppercase', letterSpacing:'1px' }}>Alpaca API Key</label>
                <input type="text" required value={alpacaKey} onChange={e=>setAlpacaKey(e.target.value)} className="auth-input" placeholder="PK..."/>
              </div>

              <div style={{ marginBottom:16 }}>
                <label style={{ display:'block', fontSize:12, fontWeight:700, color:'#3d4f6e', marginBottom:8, textTransform:'uppercase', letterSpacing:'1px' }}>Alpaca Secret Key</label>
                <input type="password" required value={alpacaSecret} onChange={e=>setAlpacaSecret(e.target.value)} className="auth-input" placeholder="••••••••"/>
              </div>

              <div style={{ marginBottom:16 }}>
                <label style={{ display:'block', fontSize:12, fontWeight:700, color:'#3d4f6e', marginBottom:8, textTransform:'uppercase', letterSpacing:'1px' }}>Alpaca URL</label>
                <select value={alpacaUrl} onChange={e=>setAlpacaUrl(e.target.value)} className="auth-input">
                  <option value="https://paper-api.alpaca.markets">Paper Trading</option>
                  <option value="https://api.alpaca.markets">Live Trading</option>
                </select>
              </div>
            </>
          )}

          {error && (
            <div style={{ padding:14, background:'rgba(255,68,102,0.1)', border:'1px solid rgba(255,68,102,0.3)', borderRadius:8, color:'#ff4466', fontSize:13, marginBottom:20, textAlign:'center' }}>
              {error}
            </div>
          )}

          <button type="submit" disabled={loading} className="auth-btn">
            {loading ? 'Loading...' : isSignup ? 'Create Account' : 'Login'}
          </button>
        </form>

        {isSignup && (
          <div style={{ marginTop:24, padding:14, background:'rgba(0,209,255,0.06)', border:'1px solid rgba(0,209,255,0.15)', borderRadius:8, fontSize:12, color:'#00d1ff', lineHeight:1.6 }}>
            <strong>Your credentials are encrypted and stored securely.</strong> The bot will trade on your Alpaca account automatically based on your configured strategy.
          </div>
        )}
      </div>
    </div>
  );
};

// Import the main TradingApp from app.js (you'll need to export it from that file)
// For now, I'll create a placeholder - in production you'd import your actual app
const TradingApp = ({ token, onLogout }) => {
  return (
    <div>
      <div style={{ padding:20, background:'#0d1117', borderBottom:'1px solid #21262d', display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <Zap size={24} color="#00ffbb" fill="#00ffbb"/>
          <span style={{ fontSize:18, fontWeight:900, color:'#00ffbb' }}>PULSE 4X</span>
        </div>
        <button onClick={onLogout} style={{ padding:'8px 16px', background:'#ff446620', border:'1px solid #ff446640', borderRadius:8, color:'#ff4466', fontWeight:700, cursor:'pointer', display:'flex', alignItems:'center', gap:8 }}>
          <LogOut size={16}/> Logout
        </button>
      </div>
      <div style={{ padding:40, color:'#fff', textAlign:'center' }}>
        <h1>Trading Dashboard</h1>
        <p>Your main trading app would be rendered here.</p>
        <p style={{ marginTop:20, color:'#666' }}>Import your existing TradingApp component and pass the token via axios interceptors.</p>
      </div>
    </div>
  );
};

// ══════════════════════════════════════════════════════════════════════════════
// ROOT APP WITH AUTH
// ══════════════════════════════════════════════════════════════════════════════

export default function AppWithAuth() {
  const [token, setToken] = useState(localStorage.getItem('titan_token'));

  const handleLogin = (newToken) => {
    localStorage.setItem('titan_token', newToken);
    setToken(newToken);
    // Set axios default header
    axios.defaults.headers.common['Authorization'] = `Bearer ${newToken}`;
  };

  const handleLogout = async () => {
    try {
      await axios.post(`${API}/auth/logout`);
    } catch (e) {}
    localStorage.removeItem('titan_token');
    setToken(null);
    delete axios.defaults.headers.common['Authorization'];
  };

  // Set axios header on mount if token exists
  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
  }, [token]);

  if (!token) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return <TradingApp token={token} onLogout={handleLogout} />;
}