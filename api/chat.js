export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    const { prompt } = req.body;
    const apiKey = process.env.LUMA_API_KEY;

    if (!apiKey) {
        return res.status(500).json({ error: 'API Key not configured in Vercel environment' });
    }

    try {
        // Example Luma AI completion endpoint (Adjust based on your specific Luma provider/API version)
        // Using common AI/ML proxy structure often used for Luma access
        const response = await fetch('https://api.lumalabs.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "luma-chat-v1", // Replace with the actual model name you intend to use
                messages: [{ role: "user", content: prompt }]
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error?.message || 'Luma API Error');
        }

        const reply = data.choices?.[0]?.message?.content || "No response from AI.";
        res.status(200).json({ reply });

    } catch (error) {
        console.error("Luma API Error:", error);
        res.status(500).json({ error: error.message });
    }
}