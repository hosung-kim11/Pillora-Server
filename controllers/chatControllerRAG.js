const { OpenAI } = require('openai');
const axios = require('axios');
const { LLMChain, OpenAIChat } = require('langchain');
const { PromptTemplate } = require('langchain/prompts');
const NodeCache = require('node-cache');
require('dotenv').config();

const cache = new NodeCache({ stdTTL: 3600 });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const llm = new OpenAIChat({
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0.7,
});

//define the prompt template
const promptTemplate = new PromptTemplate({
  inputVariables: ['message'],
  template: `As an AI assistant, analyze the user's message: "{message}". Extract the following information in JSON format:
{
  "intent": "Identified intent (e.g., Side Effects, Dosage, Interactions)",
  "entities": {
    "drugName": "Extracted drug name if any",
    "symptoms": ["List of symptoms mentioned"],
    "dosage": "Extracted dosage information",
    "time": "Extracted time information"
  }
}
Only provide the JSON response without additional text.`
});


//OpenFDA API integration
async function fetchDrugInfo(drugName) {
  const cacheKey = `drug_${drugName}`;
  const cachedData = cache.get(cacheKey);
  
  if (cachedData) return cachedData;

  try {
    const response = await axios.get(`https://api.fda.gov/drug/label.json`, {
      params: {
        search: `openfda.generic_name:"${drugName}" OR openfda.brand_name:"${drugName}"`,
        limit: 1
      }
    });
    const drugInfo = response.data.results[0];
    cache.set(cacheKey, drugInfo);
    return drugInfo;
  } catch (error) {
    console.error('OpenFDA API Error:', error);
    return null;
  }
}

//drug interaction checker
async function checkDrugInteractions(drug1, drug2) {
  const cacheKey = `interaction_${drug1}_${drug2}`;
  const cachedData = cache.get(cacheKey);
  
  if (cachedData) return cachedData;

  try {
    const response = await axios.get(`https://api.fda.gov/drug/event.json`, {
      params: {
        search: `patient.drug.openfda.generic_name:"${drug1}" AND patient.drug.openfda.generic_name:"${drug2}"`,
        limit: 1
      }
    });
    cache.set(cacheKey, response.data);
    return response.data;
  } catch (error) {
    console.error('Drug Interaction OpenFDA API Error:', error);
    return null;
  }
}

//format the recall information
const formatRecallInfo = (recalls) => {
  if (!recalls || !recalls.length) return '';
  return recalls.map(recall => 
    `recall: ${recall.reason_for_recall} (status: ${recall.status})`
  ).join('\n');
};

const chatHandler = async (req, res) => {
  try {
    const { messages } = req.body;

    //NLP processing
    const chain = new LLMChain({ llm, prompt: promptTemplate });
    const nlpResult = await chain.call({ message: userMessage });
    let intent, entities;

    try {
      ({ intent, entities } = JSON.parse(nlpResult.text));
    } catch (error) {
      console.error('NLP parsing error:', error);
      intent = 'Unknown';
      entities = {};
    }

    //fetch the drug information and recalls
    let drugInfo = null;
    let recallInfo = null;
    let interactionInfo = null;

    if (entities.drugName) {
      drugInfo = await fetchDrugInfo(entities.drugName);
      recallInfo = await fetchRecallInfo(entities.drugName);

      if (entities.drugName.includes(',')) {
        const [drug1, drug2] = entities.drugName.split(',').map(d => d.trim());
        interactionInfo = await checkDrugInteractions(drug1, drug2);
      }
    }
    const systemMessage = {
      role: 'system',
      content: `Your name is Pillora. Your role is to serve as a medical assistant and respond to users' questions in a formal yet friendly tone. Follow these important rules strictly:

1. **No Hallucinations**: Always provide accurate and verified information. If unsure of the answer, inform the user that you are unsure. Never make up information.
2. **Emergency Warning**: Always remind the user to contact a medical professional in emergencies.
3. **Friendly and Formal**: Maintain a respectful tone, showing care and attention in all interactions.

Use the following extracted information to assist the user:
- Intent: ${intent}
- Entities: ${JSON.stringify(entities)}
${drugInfo ? '- Drug Information: ' + JSON.stringify(drugInfo) : ''}
${recallInfo ? '- Recall Information: ' + formatRecallInfo(recallInfo) : ''}
${interactionInfo ? '- Drug Interaction Information: ' + JSON.stringify(interactionInfo) : ''}`
    };

    const conversation = [systemMessage, ...messages];

    const response = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: conversation,
      temperature: 0.7,
      max_tokens: 500,
    });

    res.json({
      reply: response.choices[0].message,
      recalls: recallInfo,
      drugInfo: drugInfo,
      interactions: interactionInfo
    });

  } catch (error) {
    console.error('Error in chatHandler:', error);
    res.status(500).json({
      error: error.message || 'An error occurred while processing your request.',
      status: 'error'
    });
  }
};

async function fetchRecallInfo(drugName) {
  const cacheKey = `recall_${drugName}`;
  const cachedData = cache.get(cacheKey);
  
  if (cachedData) return cachedData;

  try {
    const response = await axios.get(`https://api.fda.gov/drug/enforcement.json`, {
      params: {
        search: `product_description:"${drugName}"`,
        limit: 5
      }
    });
    const recalls = response.data.results;
    cache.set(cacheKey, recalls);
    return recalls;
  } catch (error) {
    console.error('Recall API Error:', error);
    return null;
  }
}

module.exports = {
  chatHandler
};