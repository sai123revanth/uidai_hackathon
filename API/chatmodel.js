export default async function handler(req, res) {
  // 1. Setup CORS headers
  // allow * is only valid if allow-credentials is NOT true. 
  // We removed credentials true to make the wildcard origin work safely.
  res.setHeader('Access-Control-Allow-Origin', '*'); 
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS, POST');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // 2. Handle the OPTIONS method (Preflight request)
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  let message;

  // 3. Robust Body Parsing
  // Vercel sometimes parses it, sometimes passes a string. We handle both.
  try {
    let body = req.body;
    if (typeof body === 'string') {
      body = JSON.parse(body);
    }
    message = body?.message;
  } catch (e) {
    return res.status(400).json({ error: 'Invalid JSON body', details: e.message });
  }

  if (!message) {
    return res.status(400).json({ error: 'Message is required in the request body' });
  }

  // Get API key from Vercel Environment Variables
  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    console.error('Missing GROQ_API_KEY environment variable');
    return res.status(500).json({ error: 'Server configuration error: API Key missing' });
  }

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
            { 
                role: "system", 
                content: "You are a helpful, friendly, and fast AI assistant." 
            },
            { 
                role: "user", 
                content: message 
            }
        ],
        model: "llama3-8b-8192",
        temperature: 0.7,
        max_tokens: 1024,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Groq API Error Details:', JSON.stringify(data));
      return res.status(response.status).json({ 
        error: 'Error from Groq API', 
        details: data.error?.message || 'Unknown error' 
      });
    }

    // Extract the actual reply text
    const reply = data.choices[0].message.content;

    return res.status(200).json({ reply });

  } catch (error) {
    console.error('Server Execution Error:', error);
    return res.status(500).json({ error: 'Failed to process request', details: error.message });
  }
}