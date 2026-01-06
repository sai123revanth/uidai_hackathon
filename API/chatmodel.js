export default async function handler(req, res) {
  // 1. Setup CORS headers
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

  try {
    // 3. Robust Body Parsing
    let message;
    let body = req.body;

    // Log the raw body type for debugging in Vercel logs
    console.log('Request Body Type:', typeof body);

    if (typeof body === 'string') {
      try {
        body = JSON.parse(body);
      } catch (parseError) {
        console.error('JSON Parse Error:', parseError);
        return res.status(400).json({ error: 'Invalid JSON body' });
      }
    }
    
    message = body?.message;

    if (!message) {
      return res.status(400).json({ error: 'Message is required in the request body' });
    }

    // 4. Get API key from Vercel Environment Variables
    const apiKey = process.env.GROQ_API_KEY;

    // Debug log to check if key exists (DO NOT LOG THE KEY ITSELF)
    console.log('GROQ_API_KEY Status:', apiKey ? 'Found' : 'Missing');

    if (!apiKey) {
      return res.status(500).json({ 
        error: 'Server Error: GROQ_API_KEY is missing in Vercel Environment Variables.' 
      });
    }

    // 5. Call Groq API
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
        model: "llama3-8b-8192", // Ensure this model name is current
        temperature: 0.7,
        max_tokens: 1024,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      // Log the full error from Groq
      console.error('Groq API Error Response:', JSON.stringify(data, null, 2));
      
      return res.status(response.status).json({ 
        error: 'Groq API Error', 
        details: data.error?.message || 'Unknown error from AI provider' 
      });
    }

    // Extract the actual reply text
    const reply = data.choices[0].message.content;

    return res.status(200).json({ reply });

  } catch (error) {
    console.error('Server Execution Error:', error);
    return res.status(500).json({ 
      error: 'Internal Server Error', 
      details: error.message 
    });
  }
}