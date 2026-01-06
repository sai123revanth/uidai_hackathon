export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message } = req.body;

  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  // Get API key from Vercel Environment Variables
  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
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
            // System prompt to define behavior
            { 
                role: "system", 
                content: "You are a helpful, friendly, and fast AI assistant." 
            },
            // The user's message
            { 
                role: "user", 
                content: message 
            }
        ],
        model: "llama3-8b-8192", // Using Llama 3 for speed and quality
        temperature: 0.7,
        max_tokens: 1024,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error?.message || 'Failed to fetch from Groq');
    }

    // Extract the actual reply text
    const reply = data.choices[0].message.content;

    return res.status(200).json({ reply });

  } catch (error) {
    console.error('Groq API Error:', error);
    return res.status(500).json({ error: 'Failed to process request' });
  }
}