import os
import asyncio
from dataclasses import dataclass
from typing import Dict, List
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import time

load_dotenv()

@dataclass
class AgentResponse:
    """Structure for storing safety agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

class AgentStatus:
    """Safety agent status management with sidebar display"""
    def __init__(self):
        self.sidebar_placeholder = None
        self.agents = {
            'medical_response': {'status': 'idle', 'progress': 0, 'message': ''},
            'personal_safety': {'status': 'idle', 'progress': 0, 'message': ''},
            'disaster_response': {'status': 'idle', 'progress': 0, 'message': ''},
            'women_safety': {'status': 'idle', 'progress': 0, 'message': ''}
        }
        
    def initialize_sidebar_placeholder(self):
        with st.sidebar:
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float, message: str = ""):
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        colors = {
            'idle': '#6c757d',
            'working': '#28a745',
            'completed': '#17a2b8',
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
            ">
                <div style="color: {color}; font-weight: bold;">
                    {agent_name.replace('_', ' ').title()}
                </div>
                <div style="
                    color: #FFFFFF;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

class SafetyResponseSystem:
    """Emergency response system with specialized safety agents"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,  # Increased for more creative responses
            model_name="mixtral-8x7b-32768",  # Using a more capable model
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize specialized safety agent prompts"""
        base_prompt = """You are an expert emergency response system. Your role is to provide detailed, practical guidance for emergency situations.

USER QUERY: {query}
CHAT HISTORY: {chat_history}

Analyze the situation carefully and provide comprehensive guidance following this structure:

1. SITUATION ASSESSMENT
- Analyze the specific emergency details
- Identify immediate risks and threats
- Determine priority actions

2. IMMEDIATE ACTIONS
- List specific step-by-step instructions
- Include exact timing for each step
- Specify who should take what action

3. SAFETY MEASURES
- Detail specific safety protocols
- List required safety equipment
- Provide environmental safety tips

4. EMERGENCY CONTACTS
- List relevant emergency numbers
- Specify when to contact each service
- Include backup contact options

5. FOLLOW-UP STEPS
- Provide post-emergency guidance
- List monitoring requirements
- Specify recovery actions

FORMAT YOUR RESPONSE:
• Use clear, specific language
• Include exact measurements and timing
• Provide specific examples
• Reference actual emergency protocols
• Include relevant medical/safety terms

Remember: Your guidance could save lives. Be thorough and precise."""

        self.prompts = {
            'medical_response': base_prompt + """

For medical emergencies:
- Use standard medical protocols
- Reference specific first aid procedures
- Include vital signs monitoring
- Specify medication considerations
- Detail infection control measures""",

            'personal_safety': base_prompt + """

For personal safety:
- Include de-escalation techniques
- Detail self-defense options
- Specify escape routes planning
- Include surveillance awareness
- Detail communication protocols""",

            'disaster_response': base_prompt + """

For disaster response:
- Include evacuation protocols
- Detail shelter requirements
- Specify resource management
- Include weather considerations
- Detail communication plans""",

            'women_safety': base_prompt + """

For women's safety:
- Include specific defense strategies
- Detail support network activation
- Specify legal protection steps
- Include documentation guidance
- Detail recovery resources"""
        }

    def _initialize_agents(self):
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        formatted = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_query(
        self,
        query: str,
        status_callback
    ) -> Dict[str, AgentResponse]:
        responses = {}
        chat_history = self._format_chat_history()
        
        try:
            # Determine relevant agents based on query content
            relevant_agents = self._select_relevant_agents(query)
            
            agent_tasks = []
            for agent_name in relevant_agents:
                status_callback(agent_name, 'working', 0.2, f"Analyzing situation")
                agent_tasks.append(self._get_agent_response(
                    agent_name, query, chat_history
                ))

            agent_responses = await asyncio.gather(*agent_tasks)
            
            for agent_name, response in zip(relevant_agents, agent_responses):
                responses[agent_name] = response
                status_callback(
                    agent_name,
                    'completed',
                    1.0,
                    f"Analysis complete"
                )

            final_response = await self._synthesize_safety_responses(
                query, chat_history, responses
            )
            responses['final_analysis'] = final_response

            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            for agent in self.agents.keys():
                status_callback(agent, 'error', 0, str(e))
            raise Exception(f"Safety analysis error: {str(e)}")

    def _select_relevant_agents(self, query: str) -> List[str]:
        """Select relevant agents based on query content"""
        query = query.lower()
        selected_agents = []
        
        # Medical keywords
        if any(word in query for word in ['medical', 'health', 'hurt', 'pain', 'injury', 'sick', 'blood', 'breathing', 'heart']):
            selected_agents.append('medical_response')
            
        # Personal safety keywords
        if any(word in query for word in ['safety', 'threat', 'danger', 'scared', 'attack', 'follow', 'suspicious']):
            selected_agents.append('personal_safety')
            
        # Disaster keywords
        if any(word in query for word in ['disaster', 'flood', 'fire', 'earthquake', 'storm', 'hurricane', 'evacuation']):
            selected_agents.append('disaster_response')
            
        # Women's safety keywords
        if any(word in query for word in ['woman', 'women', 'girl', 'harassment', 'stalking', 'assault']):
            selected_agents.append('women_safety')
            
        # If no specific category is detected, use all agents
        if not selected_agents:
            selected_agents = list(self.agents.keys())
            
        return selected_agents

    async def _get_agent_response(
        self,
        agent_name: str,
        query: str,
        chat_history: str
    ) -> AgentResponse:
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "query": query,
                "chat_history": chat_history
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "query_length": len(query),
                "response_confidence": self._calculate_confidence(response)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=self._calculate_confidence(response),
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Safety agent {agent_name} error: {str(e)}")

    def _calculate_confidence(self, response: str) -> float:
        """Calculate response confidence based on content analysis"""
        base_confidence = 0.7
        
        # Key phrases that indicate high-quality response
        quality_indicators = {
            "critical": 0.05,
            "immediate": 0.05,
            "emergency": 0.04,
            "warning": 0.04,
            "step": 0.03,
            "safety": 0.03,
            "contact": 0.03,
            "call": 0.03
        }
        
        # Structure indicators
        structure_indicators = {
            "1.": 0.05,
            "2.": 0.05,
            "3.": 0.05,
            "4.": 0.05
        }
        
        confidence = base_confidence
        
        # Check for quality phrases
        for phrase, weight in quality_indicators.items():
            if phrase.lower() in response.lower():
                confidence += weight
                
        # Check for proper structure
        for marker, weight in structure_indicators.items():
            if marker in response:
                confidence += weight
                
        # Penalize very short responses
        if len(response) < 100:
            confidence -= 0.2
        
        return min(0.95, max(0.3, confidence))

    async def _synthesize_safety_responses(
        self,
        query: str,
        chat_history: str,
        responses: Dict[str, AgentResponse]
    ) -> AgentResponse:
        try:
            greetings = ['hi', 'hello', 'hey', 'hii', 'greetings']
            if query.lower().strip() in greetings:
                return AgentResponse(
                    agent_name="final_analysis",
                    content="""Welcome to the Emergency Safety Response System. I'm here to provide immediate guidance for any emergency situation. How can I help you today?

Please describe your situation in detail, including:
- The type of emergency you're facing
- Your current location and environment
- Any immediate risks or threats
- Available resources or help nearby
- Any medical conditions or special circumstances

I can assist with:
- Medical emergencies and first aid guidance
- Personal safety in threatening situations
- Natural disaster response and preparation
- Women's safety and support
- General emergency guidance

The more details you provide, the more specific and helpful my guidance will be.""",
                    confidence=1.0,
                    metadata={"greeting": True},
                    processing_time=0.1
                )

            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'final_analysis'
            ])

            start_time = time.time()
            
            synthesis_template = """You are the Emergency Response Coordinator synthesizing expert analyses into actionable guidance.

EMERGENCY QUERY: {query}

EXPERT ANALYSES:
{formatted_responses}

Create a comprehensive response that:
1. Addresses the specific emergency situation
2. Provides clear, actionable steps
3. Includes relevant safety measures
4. Lists specific emergency contacts
5. Details follow-up actions

FORMAT:
• Start with most critical actions
• Use clear, specific language
• Include exact timings and measurements
• Reference proper procedures and protocols
• Maintain a calm, authoritative tone

Remember: This is a real emergency situation requiring detailed, accurate guidance."""

            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", synthesis_template)
            ])
            
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            synthesis_response = await synthesis_chain.ainvoke({
                "query": query,
                "formatted_responses": formatted_responses
            })
            
            processing_time = time.time() - start_time
            
            overall_confidence = sum(
                response.confidence for response in responses.values()
            ) / len(responses)
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "overall_confidence": overall_confidence
            }
            
            return AgentResponse(
                agent_name="final_analysis",
                content=synthesis_response,
                confidence=overall_confidence,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Safety synthesis error: {str(e)}")

def setup_streamlit_ui():
    st.set_page_config(
        page_title="Emergency Safety Response System",
        page_icon="🚨",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #28a745;
            background-color: rgba(40, 167, 69, 0.1);
            white-space: pre-wrap;
        }
        
        .agent-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #28a745;
            border-radius: 0.5rem;
            background-color: rgba(40, 167, 69, 0.05);
        }
        
        .metadata-section {
            font-size: 0.8rem;
            color: #28a745;
            margin-top: 0.5rem;
        }
        
        .emergency-button {
            background-color: #dc3545;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            cursor: pointer;
            margin: 1rem 0;
            font-weight: bold;
        }
        
        .emergency-button:hover {
            background-color: #c82333;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1a1a1a;
        }
        
        .stTextInput>div>div>input {
            color: white;
            background-color: #2d2d2d;
            border-color: #28a745;
        }
        
        .stButton>button {
            width: 100%;
            background-color: #28a745;
            color: white;
            border: none;
            padding: 0.75rem 1rem;
            border-radius: 0.3rem;
            transition: background-color 0.3s;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .stButton>button:hover {
            background-color: #218838;
        }
        
        .st-expander {
            background-color: #2d2d2d;
            border: 1px solid #28a745;
            border-radius: 0.5rem;
        }
        
        .st-expander:hover {
            border-color: #218838;
        }
        
        div[data-testid="stMarkdownContainer"] {
            color: #ffffff;
        }
        
        .critical-section {
            border-left: 4px solid #dc3545;
            padding-left: 1rem;
            margin: 1rem 0;
            background-color: rgba(220, 53, 69, 0.1);
        }
        
        .warning-section {
            border-left: 4px solid #ffc107;
            padding-left: 1rem;
            margin: 1rem 0;
            background-color: rgba(255, 193, 7, 0.1);
        }
        
        .info-section {
            border-left: 4px solid #17a2b8;
            padding-left: 1rem;
            margin: 1rem 0;
            background-color: rgba(23, 162, 184, 0.1);
        }
        
        .action-section {
            border-left: 4px solid #28a745;
            padding-left: 1rem;
            margin: 1rem 0;
            background-color: rgba(40, 167, 69, 0.1);
        }
        
        .step-by-step {
            margin: 1rem 0;
            padding: 1rem;
            background-color: rgba(40, 167, 69, 0.05);
            border-radius: 0.5rem;
        }
        
        .step-number {
            display: inline-block;
            width: 24px;
            height: 24px;
            background-color: #28a745;
            color: white;
            text-align: center;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .emergency-contact {
            background-color: rgba(220, 53, 69, 0.1);
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #dc3545;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-active {
            background-color: #28a745;
            box-shadow: 0 0 8px #28a745;
        }
        
        .status-inactive {
            background-color: #dc3545;
        }
        
        .chat-input {
            background-color: #2d2d2d !important;
            color: white !important;
            border-color: #28a745 !important;
            border-radius: 0.5rem !important;
        }
        
        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.5) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_streamlit_ui()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = SafetyResponseSystem()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    
    st.markdown("""
        <div style='text-align: center; color: #ffffff; margin-bottom: 2rem;'>
            <h1>🚨 Emergency Safety Response System</h1>
            <p style='font-size: 1.2rem;'>Get immediate, detailed guidance for emergency situations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with emergency resources
    with st.sidebar:
        st.markdown("""
            <div style='color: #28a745; margin-bottom: 1rem;'>
                <h3>🏥 Emergency Resources</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='emergency-button'>
                <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>🚑 Emergency: Call 100,108</div>
                <div style='font-size: 0.9rem; opacity: 0.8;'>Available 24/7 for immediate assistance</div>
            </div>
            
            <div style='margin-top: 2rem;'>
                <h4>Quick Access Emergency Guides:</h4>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏥 Medical Emergency Guide"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need comprehensive medical emergency instructions. Please provide step-by-step guidance for assessing and responding to a medical emergency."
            })
        
        if st.button("🛡️ Personal Safety Protocol"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need detailed personal safety instructions. What specific steps should I take to protect myself in a threatening situation?"
            })
            
        if st.button("🌪️ Disaster Response Plan"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are the detailed steps for responding to a natural disaster? Please provide comprehensive guidance for preparation, during, and after the disaster."
            })
            
        if st.button("👩 Women's Safety Guide"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need specific women's safety instructions. What detailed steps should I take for personal protection and getting immediate help if threatened?"
            })
        
        st.markdown("""
            <div style='color: #28a745; margin-top: 2rem;'>
                <h3>🤖 Safety Agents Status</h3>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                # Format the final analysis with proper sections
                final_content = message['content']['final_analysis'].content
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {format_response_sections(final_content)}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Expert Analysis", expanded=False):
                    for agent_name, response in message['content'].items():
                        if agent_name != 'final_analysis':
                            st.markdown(f"""
                                <div class="agent-card">
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="font-size: 1.5rem;">
                                            {get_agent_emoji(agent_name)}
                                        </span>
                                        <strong>{agent_name.replace('_', ' ').title()} Assessment</strong>
                                    </div>
                                    <div style="margin: 0.5rem 0;">
                                        {format_response_sections(response.content)}
                                    </div>
                                    <div class="metadata-section">
                                        <div>Response Confidence: {response.confidence:.2%}</div>
                                        <div>Analysis Time: {response.processing_time:.2f}s</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Describe your situation in detail for specific guidance...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                if responses:
                    response_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            {format_response_sections(responses['final_analysis'].content)}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": responses
                    })
                
            except Exception as e:
                response_placeholder.error(f"Error processing emergency response: {str(e)}")

def format_response_sections(content: str) -> str:
    """Format response content with proper styling for different sections"""
    # Add section-specific styling
    content = content.replace("CRITICAL PRIORITIES:", """<div class="critical-section">
        <strong>🚨 CRITICAL PRIORITIES:</strong>""")
    content = content.replace("IMMEDIATE ACTIONS:", """<div class="action-section">
        <strong>⚡ IMMEDIATE ACTIONS:</strong>""")
    content = content.replace("SAFETY MEASURES:", """<div class="warning-section">
        <strong>🛡️ SAFETY MEASURES:</strong>""")
    content = content.replace("EMERGENCY CONTACTS:", """<div class="emergency-contact">
        <strong>📞 EMERGENCY CONTACTS:</strong>""")
    content = content.replace("FOLLOW-UP STEPS:", """<div class="info-section">
        <strong>📋 FOLLOW-UP STEPS:</strong>""")
    
    # Add closing div tags
    sections = ["CRITICAL PRIORITIES:", "IMMEDIATE ACTIONS:", "SAFETY MEASURES:", 
                "EMERGENCY CONTACTS:", "FOLLOW-UP STEPS:"]
    for section in sections:
        if section in content:
            next_section_index = float('inf')
            for next_section in sections:
                if next_section != section:
                    index = content.find(next_section)
                    if index != -1 and index < next_section_index:
                        next_section_index = index
            
            if next_section_index != float('inf'):
                content = content[:next_section_index] + "</div>" + content[next_section_index:]
            else:
                content += "</div>"
    
    # Format numbered steps
    import re
    step_pattern = r'(\d+\. )'
    content = re.sub(step_pattern, r'<div class="step-number">\1</div>', content)
    
    return content

def get_agent_emoji(agent_name: str) -> str:
    """Get appropriate emoji for each agent type"""
    emoji_map = {
        'medical_response': '🏥',
        'personal_safety': '🛡️',
        'disaster_response': '🌪️',
        'women_safety': '👩',
        'final_analysis': '🚨'
    }
    return emoji_map.get(agent_name, '🤖')

if __name__ == "__main__":
    main()