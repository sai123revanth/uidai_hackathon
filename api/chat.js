export const config = {
  runtime: 'edge',
};

export default async function handler(req) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405 });
  }

  try {
    const { message } = await req.json();

    // System prompt defining the UDAAN persona
    const systemPrompt = `
      You are the AI Assistant for Project UDAAN (Unified Data Analysis for Aadhaar Network).
      
      Your Role:
      - Provide professional, insightful, and data-driven answers regarding urban analytics, Aadhaar integration, and demographics in India.
      - Explain features of the portal: Slum Gentrification Monitor, Suburban Sprawl Detection, Rental Heatmaps, and Smart Resource Allocation.
      - Maintain a tone that is technical yet accessible to policymakers and urban planners.
      - If asked about sensitive personal data, clarify that UDAAN uses anonymized, aggregated data patterns, not individual private records.
      - IMPORTANT: You must support ALL major Indian languages (Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Malayalam, Odia, Punjabi, etc.).
      - If the user asks in an Indian language, reply in that same language while maintaining the professional persona.
      
      Context:
      - UDAAN integrates Aadhaar insights for comprehensive demographic analysis.
      - It monitors real-time urban sprawl and migration trends.
      - It aids in AI-driven resource allocation for smart city planning.
    `;

    // Using Groq API with OpenAI compatibility
    // Ensure process.env.LUMA_API_KEY contains your Groq API Key
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.LUMA_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: message }
        ],
        temperature: 0.7,
        max_tokens: 1024,
      }),
    });

    const data = await response.json();

    // Error handling for the API response
    if (!response.ok) {
      console.error('Groq API Error:', data);
      return new Response(JSON.stringify({ error: 'Failed to process query' }), { status: 500 });
    }

    // Extract the content from Groq's OpenAI-compatible response structure
    const botResponse = data.choices[0].message.content;

    return new Response(JSON.stringify({ reply: botResponse }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Server Error:', error);
    return new Response(JSON.stringify({ error: 'Internal Server Error' }), { status: 500 });
  }
}