import streamlit as st
import os
import uuid
import requests
from typing import Optional, List
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

# Load environment variables from .env file (local development)
load_dotenv()


def get_secret(key: str) -> Optional[str]:
    """Get secret from Streamlit secrets or environment variables
    
    Priority:
    1. Streamlit secrets (for production/Streamlit Cloud)
    2. Environment variables (for local development with .env)
    """
    # Try Streamlit secrets first (production)
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    
    # Fallback to environment variables (local development)
    return os.getenv(key)


class PromptWithExamples(BaseModel):
    """Model for system prompt with example questions"""
    system_prompt: str
    example_questions: List[str]


class PromptGenerator:
    """Handles Azure OpenAI integration for prompt generation"""
    
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name
        
        # Select credentials based on model (using get_secret for compatibility)
        if model_name == 'gpt-4o-mini':
            api_key = get_secret("GPT4_MINI_API_KEY")
            endpoint = get_secret("GPT4_MINI_ENDPOINT")
            self.deployment = get_secret("GPT4_MINI_DEPLOYMENT")
        else:  # gpt-o3-mini
            api_key = get_secret("GPT3_MINI_API_KEY")
            endpoint = get_secret("GPT3_MINI_ENDPOINT")
            self.deployment = get_secret("GPT3_MINI_DEPLOYMENT")
        
        # Validate credentials
        if not all([api_key, endpoint, self.deployment]):
            missing = []
            if not api_key: missing.append("API Key")
            if not endpoint: missing.append("Endpoint")
            if not self.deployment: missing.append("Deployment")
            raise ValueError(f"Missing {model_name} credentials: {', '.join(missing)}")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-12-01-preview"
        )
    
    def generate_system_prompt(self, answers: dict) -> tuple[str, List[str]]:
        """Generate a system prompt and example questions based on user answers
        
        Returns:
            tuple: (system_prompt, example_questions)
        """
        
        # Construct the generation prompt
        prompt = f"""You are an expert in creating high-quality system prompts for AI assistants.
Based on the following information provided by the user, create a comprehensive, well-structured system prompt that will guide an AI assistant's behavior.

You also need to generate 4-5 realistic example questions that users might ask this AI assistant based on its role and domain.

**User's Answers:**

1. **Activité et rôle de l'assistant IA:**
{answers.get('activite', 'Non spécifié')}

2. **Règles absolues à respecter:**
{answers.get('regles', 'Non spécifié')}

3. **Personnalité de l'assistant:**
{answers.get('personnalite', 'Non spécifié')}

4. **Scénarios spécifiques:**
{answers.get('scenarios', 'Non spécifié')}

**Best Practices for System Prompts:**
- Start with a clear role definition
- Specify the assistant's expertise and knowledge domain
- Define behavioral guidelines and constraints
- Include tone and communication style instructions
- Address edge cases and special scenarios
- Be specific and actionable
- Use clear, structured language
- Include examples when relevant

**Examples of Well-Structured System Prompts for Inspiration:**

Example 1 - E-commerce Assistant (Glisshop):
```
You are an advanced AI sales assistant representing Glisshop, an e-commerce store specialized in winter sports and outdoor gear. Your mission is to guide customers throughout their shopping journey: recommending products, answering pre-sales questions, and providing relevant post-sales support.

You must ONLY use information and URLs that are EXPLICITLY provided in the search results given to you. NEVER use or reference any URLs, products, or information that does not appear in these search results. If certain information is not available in the results, politely inform the customer that you don't have that specific information.

Your objectives are to:
- Enhance the customer experience
- Increase conversion and average order value
- Ensure accurate, fast, and friendly assistance

Guidelines for Response:
- If the user sends a message with only one or two words (e.g., "Retour", "Randonnée bivouac"), you HAVE TO ask them to provide more details or clarify their question before you continue.
- Only rely on the information provided in the search results. Never guess, invent, or modify product information or URL links.
- Share links ONLY if they appear in the search results provided to you. Never create, modify, or assume URLs exist.
- Before recommending any product, YOU MUST ALWAYS ask at least two clarifying questions to better understand the customer's needs (e.g., use, budget, size).
- Only suggest complementary products (e.g., helmet with skis) if they actually appear in the provided search results.
- Include product links with descriptive anchor text (NEVER naked or invented URLs).
- If the search results don't contain certain requested information, clearly state that you don't have this information and suggest contacting customer service or offer to talk to a human.
- Adopt a friendly tone but not too excited, and always use the "vous" form.
- When providing recommendations, limit the examples to 3.

Conditional Instructions:
- If a question concerns ski boots AND the link https://www.glisshop.com/conseils/taille-chaussure-ski.html appears in your search results, then include this link in your answer.
- When suggesting a product, only give the specific product link if it appears in the search results. Never construct or guess product URLs.
- Only mention products that are in stock (not marked as "épuisé") IF this information is available in the search results.

Protocol for Answering:
- For product search: Recommend products that are present in the results.
- For product inquiry: Provide information using only verified data from the results.
- For pre/post-sales questions: Answer using only information from the results.
- If you don't have the answer just say "Je ne sais pas répondre à cette question, pouvez-vous reformuler ? A moins que vous ne vouliez parler à un humain ?"

Fallback Response:
If none of the search results contain relevant information for the query, respond with: "Je n'ai pas trouvé d'informations spécifiques sur ce sujet dans ma base de connaissances actuelle. Pour obtenir des informations précises, je vous invite à contacter notre service client."

You always HAVE TO end every conversation with: "Votre avis nous est précieux ! N'hésitez pas à noter la conversation." (translate if conversation is not in French)

Add subtle emojis: for example green ✅ for strengths, red ❌ for weaknesses, keeping it professional.
```

Example 2 - Technical Support Assistant:
```
You are a technical support specialist for a SaaS company providing project management software. Your role is to help users troubleshoot issues, understand features, and optimize their use of the platform.

Core Responsibilities:
- Diagnose and resolve technical issues
- Provide clear step-by-step instructions
- Escalate complex problems when necessary
- Document common issues and solutions

Communication Guidelines:
- Be patient and empathetic, especially with frustrated users
- Use simple language, avoiding unnecessary jargon
- Break down complex solutions into manageable steps
- Always confirm understanding before closing the conversation
- Respond within 2-3 minutes maximum

Constraints:
- Never ask for passwords or sensitive credentials
- Do not make promises about feature releases or timelines
- Always verify user identity before accessing account details
- Escalate to senior support for billing or account termination requests

When uncertain:
If you don't have enough information to provide a solution, ask clarifying questions about:
- The exact steps that led to the issue
- Any error messages received
- The user's browser/device/OS version
- Whether the issue is reproducible
```

Example 3 - Customer Service Assistant (Banking):
```
You are a customer service representative for a digital bank. Your mission is to assist customers with account inquiries, transaction questions, and general banking support while maintaining the highest standards of security and professionalism.

Key Principles:
- Security first: Never request or share sensitive information (passwords, PINs, full card numbers)
- Accuracy: Only provide information you are certain about
- Compliance: Adhere to banking regulations and privacy laws at all times

Your Tone Should Be:
- Professional yet warm
- Reassuring, especially regarding security concerns
- Clear and concise, avoiding banking jargon
- Patient with less tech-savvy customers

What You Can Help With:
- Account balance and transaction history inquiries
- Explanation of fees and charges
- Card activation and replacement
- General product information
- Directing customers to appropriate resources

What Requires Escalation:
- Fraud reports or suspicious activity
- Loan applications or modifications
- Account closures
- Disputes exceeding $500
- Legal or regulatory inquiries

Standard Responses:
- For security verification: "For your security, I'll need to verify your identity. Can you please provide [specific non-sensitive information]?"
- For out-of-scope requests: "I understand this is important. For [specific issue], I'll need to connect you with our specialized team who can assist you better."
- For unclear requests: "To help you more effectively, could you provide more details about [specific aspect]?"
```

Now, based on the user's answers and inspired by the structure and clarity of these examples, you need to generate TWO things:

1. **system_prompt**: A professional, comprehensive system prompt that incorporates all the information provided above. 
   The prompt should be:
   - Ready to use directly as a system message
   - Well-structured with clear sections
   - Specific and actionable
   - Professional yet natural

2. **example_questions**: A list of 4-5 realistic questions that users might ask this AI assistant, based on:
   - The assistant's domain and role
   - Common scenarios in this context
   - Different types of inquiries (simple, complex, edge cases)
   - Varied question formats (short/long, general/specific)

Examples of good example_questions for different domains:
- E-commerce: "Quelles chaussures de running recommandez-vous pour un débutant ?", "Comment retourner un article ?", "Avez-vous des promotions en cours ?", "Quelle est la durée de livraison ?"
- Tech Support: "Mon application ne se lance pas, que faire ?", "Comment réinitialiser mon mot de passe ?", "L'export Excel ne fonctionne plus"
- Banking: "Quel est mon solde actuel ?", "Comment activer ma carte bancaire ?", "Puis-je augmenter mon plafond de paiement ?"

Return your response in the following JSON structure."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at creating effective system prompts for AI assistants."},
                    {"role": "user", "content": prompt}
                ],
                response_format=PromptWithExamples,
                temperature=0.7,
                max_tokens=3000
            )
            
            result = completion.choices[0].message.parsed
            return result.system_prompt, result.example_questions
        
        except Exception as e:
            raise Exception(f"Erreur lors de la génération du prompt: {str(e)}")


class ChatbotTester:
    """Handles Tolk.ai API integration for chatbot testing"""
    
    def __init__(self, project_id: str, system_prompt: str):
        self.project_id = project_id
        self.system_prompt = system_prompt
        self.api_url = f"https://genii-messages-01.tolk.ai/v1/projects/{project_id}/answer"
    
    def generate_uuid(self) -> str:
        """Generate a UUID for conversation and user IDs"""
        return str(uuid.uuid4())
    
    def send_message(self, message: str, language: str = "fr", model: str = "gpt-4o-mini") -> dict:
        """Send a message to the chatbot and get response"""
        
        request_body = {
            "conversation": {
                "id": self.generate_uuid()
            },
            "message": {
                "text": message.strip()
            },
            "trigger": {
                "type": "input",
                "resource": None
            },
            "user": {
                "id": self.generate_uuid(),
                "language": language
            },
            "history": [],
            "promptConfig": {
                "value": self.system_prompt,
                "temperature": 0,
                "model": model
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if not response.ok:
                return {
                    "status": "error",
                    "text": f"HTTP error! status: {response.status_code}, body: {response.text}"
                }
            
            data = response.json()
            
            # Try to extract text from response
            if data.get('answer', {}).get('text'):
                return {"status": "success", "text": data['answer']['text']}
            elif data.get('content'):
                content = data['content']
                if isinstance(content, list):
                    text_content = next((item.get('text') for item in content if item.get('text')), None)
                    if text_content:
                        return {"status": "success", "text": text_content}
                elif isinstance(content, str):
                    return {"status": "success", "text": content}
            
            return {"status": "error", "text": "Unable to extract text from API response"}
        
        except requests.Timeout:
            return {"status": "error", "text": "Request timed out. Please try again."}
        except Exception as e:
            return {"status": "error", "text": f"Error: {str(e)}"}


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'generated_prompt' not in st.session_state:
        st.session_state.generated_prompt = None
    if 'example_questions' not in st.session_state:
        st.session_state.example_questions = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'project_id' not in st.session_state:
        st.session_state.project_id = ""
    if 'prompt_edited' not in st.session_state:
        st.session_state.prompt_edited = None
    if 'clicked_question' not in st.session_state:
        st.session_state.clicked_question = None


def main():
    st.set_page_config(
        page_title="Générateur de System Prompt",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 Générateur de System Prompt")
    
    # Phase 1: Prompt Generation in Sidebar
    with st.sidebar:
        st.header("📝 Phase 1: Génération")
        
        activite = st.text_area(
            "1. Activité et rôle de l'IA",
            placeholder="Ex: E-commerce de produits tech. L'assistant aide à trouver des produits et traiter les réclamations.",
            height=100,
            help="Domaine d'activité, type d'assistant, objectifs"
        )
        
        regles = st.text_area(
            "2. Règles absolues",
            placeholder="Ex: Ne jamais promettre de réductions non autorisées...",
            height=80,
            help="Contraintes et limitations"
        )
        
        personnalite = st.text_area(
            "3. Personnalité",
            placeholder="Ex: Empathique, patient, solution-oriented...",
            height=80,
            help="Traits de personnalité et ton"
        )
        
        scenarios = st.text_area(
            "4. Scénarios spécifiques",
            placeholder="Ex: Escalader les réclamations urgentes...",
            height=80,
            help="Cas particuliers (optionnel)"
        )
        
        # Model selection
        model_choice = st.selectbox(
            "Modèle Azure OpenAI",
            options=['gpt-4o-mini', 'gpt-o3-mini'],
            help="Modèle pour générer le prompt"
        )
        
        if st.button("✨ Générer", type="primary", use_container_width=True):
            if not activite.strip():
                st.error("⚠️ Décrivez au minimum l'activité")
            else:
                with st.spinner("Génération en cours..."):
                    try:
                        generator = PromptGenerator(model_name=model_choice)
                        answers = {
                            'activite': activite,
                            'regles': regles,
                            'personnalite': personnalite,
                            'scenarios': scenarios
                        }
                        
                        prompt, example_questions = generator.generate_system_prompt(answers)
                        st.session_state.generated_prompt = prompt
                        st.session_state.prompt_edited = prompt
                        st.session_state.example_questions = example_questions
                        st.success("✅ Généré !")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
    
    # Main content: Split screen layout
    if st.session_state.generated_prompt:
        # Create two columns: left for prompt/config, right for chat
        left_col, right_col = st.columns([1, 1])
        
        # LEFT COLUMN: System Prompt + Configuration
        with left_col:
            st.subheader("📝 System Prompt")
            
            edited_prompt = st.text_area(
                "Modifiez le prompt si nécessaire:",
                value=st.session_state.prompt_edited,
                height=300,
                key="prompt_editor",
                label_visibility="collapsed"
            )
            
            # Update the edited prompt in session state
            if edited_prompt != st.session_state.prompt_edited:
                st.session_state.prompt_edited = edited_prompt
            
            st.markdown("---")
            st.subheader("⚙️ Configuration")
            
            # Project ID input
            project_id = st.text_input(
                "Project ID",
                value=st.session_state.project_id,
                placeholder="Entrez votre Project ID",
                help="ID du projet Tolk.ai"
            )
            
            if project_id != st.session_state.project_id:
                st.session_state.project_id = project_id
                # Reset chat history when project ID changes
                st.session_state.messages = []
            
            # Configuration options
            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox(
                    "Langue",
                    options=['fr', 'en', 'es', 'de', 'it', 'pt', 'nl'],
                    index=0
                )
            with col2:
                model = st.selectbox(
                    "Modèle",
                    options=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                    index=0
                )
        
        # RIGHT COLUMN: Chat Interface
        with right_col:
            st.subheader("💬 Test du Prompt")
            
            if not project_id.strip():
                st.warning("⚠️ Entrez un Project ID dans la configuration")
            else:
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Display example questions as clickable buttons if no messages yet
                if not st.session_state.messages and st.session_state.example_questions:
                    st.markdown("**💡 Questions suggérées :**")
                    st.caption("Cliquez pour démarrer")
                    
                    # Display questions as buttons
                    for idx, question in enumerate(st.session_state.example_questions):
                        if st.button(
                            f"💬 {question}",
                            key=f"example_q_{idx}",
                            use_container_width=True,
                            type="secondary"
                        ):
                            st.session_state.clicked_question = question
                            st.rerun()
                    
                    st.markdown("")
                
                # Handle clicked question from button
                user_input = None
                if hasattr(st.session_state, 'clicked_question') and st.session_state.clicked_question:
                    user_input = st.session_state.clicked_question
                    st.session_state.clicked_question = None
                
                # Chat input
                if not user_input:
                    user_input = st.chat_input("Posez votre question...")
                
                if user_input:
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    # Get bot response
                    with st.chat_message("assistant"):
                        with st.spinner("Réflexion..."):
                            try:
                                tester = ChatbotTester(
                                    project_id=project_id,
                                    system_prompt=st.session_state.prompt_edited
                                )
                                
                                response = tester.send_message(
                                    message=user_input,
                                    language=language,
                                    model=model
                                )
                                
                                if response["status"] == "success":
                                    st.write(response["text"])
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": response["text"]
                                    })
                                else:
                                    error_msg = f"❌ Erreur: {response['text']}"
                                    st.error(error_msg)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": error_msg
                                    })
                            
                            except Exception as e:
                                error_msg = f"❌ Erreur: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                
                # Clear chat button
                if st.session_state.messages:
                    if st.button("🗑️ Effacer", use_container_width=True):
                        st.session_state.messages = []
                        st.rerun()
    
    else:
        st.info("👈 Remplissez les questions dans la sidebar et cliquez sur 'Générer' pour commencer")


if __name__ == "__main__":
    main()

