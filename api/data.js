export default async function handler(req, res) {
    // 1. Securely access the key from Vercel Environment Variables
    const API_KEY = process.env.UIDAI_API_KEY;

    if (!API_KEY) {
        return res.status(500).json({ error: 'Server configuration error: UIDAI_API_KEY is missing.' });
    }

    // 2. Get parameters passed from your HTML frontend
    // Default to 100 rows and Andhra Pradesh if not specified
    const limit = req.query.limit || 100;
    const state = req.query.state || 'Andhra Pradesh';

    // 3. Define the Hackathon API Endpoint
    // IMPORTANT: Replace this URL with the actual endpoint URL provided in your Hackathon documentation.
    // It usually looks like: https://api.uidai.gov.in/hackathon/data or similar.
    const EXTERNAL_API_URL = 'https://api.uidai.gov.in/hackathon/v1/insights'; 

    try {
        // 4. Make the request to the real UIDAI API
        const response = await fetch(`${EXTERNAL_API_URL}?limit=${limit}&state=${encodeURIComponent(state)}&format=json`, {
            method: 'GET',
            headers: {
                // Common patterns for API keys. Uncomment the one used by your specific Hackathon documentation:
                
                // Pattern A: Bearer Token (Most common)
                // 'Authorization': `Bearer ${API_KEY}`,
                
                // Pattern B: Custom Header
                'x-api-key': API_KEY,
                
                // Pattern C: Sometimes keys are sent as 'apikey'
                // 'apikey': API_KEY,

                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`UIDAI API responded with status: ${response.status}`);
        }

        const data = await response.json();

        // 5. Return the data to your frontend
        // We set Cache-Control to prevent hitting the hackathon API limit too frequently
        res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate');
        return res.status(200).json(data);

    } catch (error) {
        console.error('Proxy Error:', error);
        return res.status(500).json({ 
            error: 'Failed to fetch data from UIDAI', 
            details: error.message 
        });
    }
}