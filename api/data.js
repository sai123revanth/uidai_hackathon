export default async function handler(req, res) {
    // 1. Securely access the key from Vercel Environment Variables
    const API_KEY = process.env.UIDAI_API_KEY;

    if (!API_KEY) {
        return res.status(500).json({ error: 'Server configuration error: UIDAI_API_KEY is missing.' });
    }

    // 2. Parameters from frontend
    const limit = req.query.limit || 100;
    const state = req.query.state || 'Andhra Pradesh';

    // 3. The exact endpoint from the Data.gov.in / UIDAI portal
    // Based on your screenshot, this is the specific resource ID for Aadhaar insights
    const BASE_URL = 'https://api.data.gov.in/resource/319e0787-8d05-43a0-8356-591a580a569a'; 

    try {
        /**
         * 4. Construct the URL with correct query params:
         * api-key: The authentication key
         * format: json (as selected in your screenshot)
         * limit: 100
         * filters[state]: Filter for Andhra Pradesh
         */
        const targetUrl = new URL(BASE_URL);
        targetUrl.searchParams.append('api-key', API_KEY);
        targetUrl.searchParams.append('format', 'json');
        targetUrl.searchParams.append('limit', limit);
        targetUrl.searchParams.append('filters[state]', state);

        const response = await fetch(targetUrl.toString(), {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`External API error (${response.status}): ${errorText}`);
        }

        const data = await response.json();

        // 5. Return the data to your frontend
        // Note: data.gov.in usually returns an object with a 'records' array
        res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate');
        return res.status(200).json(data);

    } catch (error) {
        console.error('Proxy Error:', error);
        return res.status(500).json({ 
            error: 'Failed to fetch data from the UIDAI portal', 
            details: error.message 
        });
    }
}