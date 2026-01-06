export default async function handler(req, res) {
    // 1. Only allow POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed. Please use POST.' });
    }

    const { prompt } = req.body;
    const apiKey = process.env.LUMA_API_KEY; // You can keep the name, but ensure the value in Vercel is your Groq API Key

    // 2. Check if API Key is missing
    if (!apiKey) {
        return res.status(500).json({ 
            error: 'Missing API Key. Ensure your Groq API Key is added to Vercel Environment Variables as "LUMA_API_KEY" and redeploy.' 
        });
    }

    try {
        // Calling the Groq API (OpenAI-compatible)
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "llama-3.3-70b-versatile",
                messages: [
                    { 
                        role: "system", 
                        content: "You are a helpful, high-performance assistant powered by Llama 3.3." 
                    },
                    { 
                        role: "user", 
                        content: prompt 
                    }
                ],
                temperature: 0.7,
                max_tokens: 1024,
                stream: false
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.error("Groq API Error Details:", data);
            return res.status(response.status).json({ 
                error: `API Error (${response.status}): ${data.error?.message || JSON.stringify(data)}` 
            });
        }

        const reply = data.choices?.[0]?.message?.content || "The AI returned an empty response.";
        return res.status(200).json({ reply });

    } catch (error) {
        console.error("Serverless Function Error:", error);
        return res.status(500).json({ error: `Server Error: ${error.message}` });
    }
}