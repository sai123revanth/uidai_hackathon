export default async function handler(req, res) {
    // 1. Only allow POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed. Please use POST.' });
    }

    const { prompt } = req.body;
    const apiKey = process.env.LUMA_API_KEY;

    // 2. Check if API Key is missing
    if (!apiKey) {
        return res.status(500).json({ 
            error: 'Missing API Key. Ensure "LUMA_API_KEY" is added to Vercel Environment Variables and you have redeployed.' 
        });
    }

    try {
        /**
         * NOTE: Luma AI is primarily for Video/Images. 
         * If you are using a third-party provider (like OpenRouter) or a specific Luma text model,
         * ensure the URL and Model name below are correct.
         */
        const response = await fetch('https://api.lumalabs.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                model: "luma-chat-v1", // WARNING: Ensure this is a valid model name for your specific Luma API access
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "user", content: prompt }
                ],
                stream: false
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.error("Luma API Error Details:", data);
            return res.status(response.status).json({ 
                error: `Luma API Error (${response.status}): ${data.error?.message || JSON.stringify(data)}` 
            });
        }

        const reply = data.choices?.[0]?.message?.content || "The AI returned an empty response.";
        return res.status(200).json({ reply });

    } catch (error) {
        console.error("Serverless Function Error:", error);
        return res.status(500).json({ error: `Server Error: ${error.message}` });
    }
}