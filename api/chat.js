export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    const { prompt } = req.body;
    const apiKey = process.env.LUMA_API_KEY;

    if (!apiKey) {
        return res.status(500).json({ 
            error: 'API Key not configured. Please add your Groq API Key to Vercel environment variables.' 
        });
    }

    try {
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
                        content: `You are the UIDAI (Unique Identification Authority of India) Assistant. 
                        
                        CRITICAL INSTRUCTION: You must support all Indian languages (Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Odia, Malayalam, Punjabi, etc.). 
                        Detect the language used by the user in their query and respond ENTIRELY in that same language. 
                        If the user asks in Hindi, answer in Hindi. If they ask in Tamil, answer in Tamil.
                        
                        Knowledge Domain:
                        - Aadhaar enrollment, updates (Name, Address, DOB, Gender, Mobile, Email).
                        - Document requirements (POI, POA).
                        - Aadhaar PVC Card, E-Aadhaar downloads, and m-Aadhaar.
                        - Authentication history and biometric locking.
                        - Finding Aadhaar centers.
                        
                        Always be professional, helpful, and concise. Ensure your translations are accurate and culturally appropriate for the specific Indian language used.` 
                    },
                    { 
                        role: "user", 
                        content: prompt 
                    }
                ],
                temperature: 0.5, // Lower temperature for more factual multilingual accuracy
                max_tokens: 1500
            })
        });

        const data = await response.json();

        if (!response.ok) {
            return res.status(response.status).json({ 
                error: data.error?.message || 'API Communication Error' 
            });
        }

        const reply = data.choices?.[0]?.message?.content || "I am unable to provide a response at this moment.";
        return res.status(200).json({ reply });

    } catch (error) {
        return res.status(500).json({ error: "Internal Server Error" });
    }
}