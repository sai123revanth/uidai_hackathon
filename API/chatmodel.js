module.exports = async (req, res) => {
    // 1. CORS Headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS, POST');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
    // 2. Preflight
    if (req.method === 'OPTIONS') {
      res.status(200).end();
      return;
    }
  
    // 3. Method Check
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Method not allowed' });
    }
  
    try {
      // 4. Body Parsing
      let message;
      let body = req.body;
  
      if (typeof body === 'string') {
        try { body = JSON.parse(body); } catch (e) {}
      }
      
      message = body?.message || body?.prompt;
  
      if (!message) {
        return res.status(400).json({ error: 'Message is required' });
      }
  
      // 5. API Key Check
      const apiKey = process.env.GROQ_API_KEY;
      if (!apiKey) {
        return res.status(500).json({ error: 'Configuration Error: GROQ_API_KEY is missing.' });
      }
  
      // 6. Groq API Call
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [
            { role: "system", content: "You are a helpful AI assistant." },
            { role: "user", content: message }
          ],
          model: "llama3-8b-8192",
          temperature: 0.7,
          max_tokens: 1024,
        }),
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        return res.status(response.status).json({ 
          error: 'Groq API Error', 
          details: data.error?.message || 'Unknown error' 
        });
      }
  
      return res.status(200).json({ reply: data.choices[0].message.content });
  
    } catch (error) {
      console.error('Server Error:', error);
      return res.status(500).json({ error: 'Internal Server Error', details: error.message });
    }
  };