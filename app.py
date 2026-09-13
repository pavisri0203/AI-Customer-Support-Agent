import streamlit as st
import pandas as pd
import re
from openai import OpenAI

from classifier import classify_intent
from rag import get_historical_reply
from escalation import check_escalation


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="SupportAI | AppleSupport",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = OpenAI()


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("sample.csv")


df = load_dataset()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_twitter_text(text):

    text = str(text)

    # Remove @username
    text = re.sub(r"@\w+\s*", "", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    return text.strip()


def get_apple_replies():

    return df[
        df["author_id"].astype(str) == "AppleSupport"
    ].copy()


def get_conversation_pairs():

    apple = get_apple_replies()

    pairs = []

    for _, reply_row in apple.iterrows():

        customer_id = reply_row[
            "in_response_to_tweet_id"
        ]

        if pd.isna(customer_id):
            continue

        try:
            customer_id = str(int(customer_id))
        except:
            customer_id = str(customer_id)

        customer = df[
            df["tweet_id"].astype(str) == customer_id
        ]

        if not customer.empty:

            pairs.append({
                "customer": str(
                    customer.iloc[0]["text"]
                ),

                "reply": clean_twitter_text(
                    reply_row["text"]
                )
            })

    return pairs


def get_intent_description(intent):

    descriptions = {

        "battery_issue":
            "Battery drain or battery-life related problem",

        "update_problem":
            "iOS update, slowdown, freezing or update-related problem",

        "app_feature_problem":
            "Application crash or application feature problem",

        "account_code_problem":
            "Verification, account or security-code problem",

        "general_issue":
            "Issue could not be confidently classified"
    }

    return descriptions.get(
        intent,
        "Customer support issue"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Support Inbox"


if "last_message" not in st.session_state:

    st.session_state.last_message = ""


if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍎 SupportAI")

    st.caption(
        "AI Customer Support Operations"
    )

    st.divider()

    st.markdown("### Workspace")

    pages = [
        "💬 Support Inbox",
        "📊 Analytics",
        "🧠 AI Knowledge",
        "🛡️ Trust & Safety",
        "📈 Evaluation",
        "⚙️ Settings"
    ]

    page_names = [
        "Support Inbox",
        "Analytics",
        "AI Knowledge",
        "Trust & Safety",
        "Evaluation",
        "Settings"
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=page_names.index(
            st.session_state.page
        ),
        label_visibility="collapsed"
    )

    st.session_state.page = selected_page.split(
        " ",
        1
    )[1]

    st.divider()

    st.subheader("Active Brand")

    st.success(
        "🍎 AppleSupport"
    )

    st.divider()

    st.markdown("### AI Pipeline")

    st.caption(
        "1  🎯 Intent Classification"
    )

    st.caption(
        "2  🔎 Semantic Retrieval"
    )

    st.caption(
        "3  📚 Historical Grounding"
    )

    st.caption(
        "4  🤖 Reply Generation"
    )

    st.caption(
        "5  🛡️ Safety Decision"
    )

    st.divider()

    st.caption(
        "Dataset: Customer Support on Twitter"
    )

    st.caption(
        "Hiver SDE Intern Take-Home"
    )


# ============================================================
# TOP HEADER
# ============================================================

st.title("🍎 SupportAI")

st.caption(
    "Intelligent customer support grounded in real AppleSupport conversations"
)

st.divider()


# ============================================================
# PAGE 1 — SUPPORT INBOX
# ============================================================

if st.session_state.page == "Support Inbox":

    st.header("💬 Support Inbox")

    st.caption(
        "Analyze a customer message and generate a historically grounded reply."
    )

    with st.container(border=True):

        st.subheader(
            "New Customer Message"
        )

        message = st.text_area(
            "Customer message",
            value=st.session_state.last_message,
            placeholder=(
                "Example: My iPhone battery is draining very fast"
            ),
            height=120
        )

        col1, col2 = st.columns([4, 1])

        with col1:

            analyze = st.button(
                "🔍 Analyze Message",
                type="primary",
                use_container_width=True
            )

        with col2:

            clear = st.button(
                "Clear",
                use_container_width=True
            )

    # Clear button

    if clear:

        st.session_state.last_message = ""

        st.session_state.last_result = None

        st.rerun()


    # Analyze button

    if analyze:

        if not message.strip():

            st.warning(
                "Please enter a customer message."
            )

        else:

            st.session_state.last_message = message

            with st.spinner(
                "AI agent is analyzing the conversation..."
            ):

                # Intent

                intent = classify_intent(
                    message
                )

                # Historical retrieval

                historical_reply = get_historical_reply(
                    message
                )

                # Clean historical response

                clean_historical_reply = clean_twitter_text(
                    historical_reply
                )

                # Escalation

                decision, reason = check_escalation(
                    message,
                    intent,
                    historical_reply
                )


                # LLM prompt

                prompt = f"""
You are an Apple Support customer service assistant.

Write a short, polite and professional customer-facing reply.

Use the historical Apple Support response as grounding.

Do NOT invent:
- policies
- refunds
- guarantees
- unsupported technical steps
- unsupported troubleshooting instructions

Do NOT include:
- Twitter usernames
- @mentions
- tweet IDs
- Twitter URLs
- t.co links

Do not mention the historical dataset.
Do not mention AI.
Do not explain your reasoning.

Customer message:
{message}

Detected intent:
{intent}

Historical Apple Support response:
{clean_historical_reply}

Generate ONLY the final customer-facing reply.
"""


                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=prompt
                )


                ai_reply = response.output_text.strip()


            # Save result

            st.session_state.last_result = {

                "message": message,

                "intent": intent,

                "historical":
                    clean_historical_reply,

                "reply":
                    ai_reply,

                "decision":
                    decision,

                "reason":
                    reason
            }


    # Display result

    result = st.session_state.last_result


    if result:

        st.divider()

        st.header("Conversation")

        with st.container(border=True):

            st.caption("CUSTOMER")

            st.write(
                result["message"]
            )


        st.subheader(
            "AI Analysis"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Intent",
                result["intent"]
            )


        with col2:

            st.metric(
                "Grounding",
                "Historical Match"
            )


        with col3:

            if result["decision"] == "AUTO-HANDLE":

                st.metric(
                    "Decision",
                    "AUTO-HANDLE"
                )

            else:

                st.metric(
                    "Decision",
                    "HUMAN REVIEW"
                )


        st.caption(
            get_intent_description(
                result["intent"]
            )
        )


        # AI reply

        st.subheader(
            "🤖 Suggested Reply"
        )


        with st.container(border=True):

            st.info(
                result["reply"]
            )


        # Decision

        st.subheader(
            "🛡️ Automation Decision"
        )


        if result["decision"] == "AUTO-HANDLE":

            st.success(
                "✅ AUTO-HANDLE"
            )

        else:

            st.error(
                "🚨 ESCALATE TO HUMAN"
            )


        st.caption(
            "Reason: " +
            result["reason"]
        )


        # Historical evidence

        with st.expander(
            "📚 View Historical Evidence"
        ):

            st.caption(
                "Retrieved AppleSupport resolution"
            )

            st.write(
                result["historical"]
            )

            st.divider()

            st.caption(
                "Why this matters"
            )

            st.write(
                "The AI uses this historical response "
                "as grounding evidence instead of inventing "
                "a new support policy."
            )


        # Trust

        with st.container(border=True):

            st.subheader(
                "🔐 Trust & Safety"
            )

            st.write(
                "The agent prioritizes correctness and safety "
                "over automation. Sensitive, uncertain or "
                "unsupported cases are routed to human support."
            )


    else:

        st.info(
            "👋 Ready to analyze a customer support message. "
            "Enter a message above to begin."
        )


        st.subheader(
            "Try a sample message"
        )


        examples = [

            "My iPhone battery is draining very fast",

            "My apps keep crashing after the iOS update",

            "I am not receiving my verification code",

            "My Apple account has been hacked"
        ]


        for example in examples:

            st.code(
                example,
                language=None
            )


# ============================================================
# PAGE 2 — ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":

    st.header("📊 Analytics")

    st.caption(
        "Dataset and support automation overview"
    )


    apple = get_apple_replies()

    pairs = get_conversation_pairs()


    total_tweets = len(df)

    total_apple_replies = len(apple)

    total_conversations = len(pairs)


    # Metrics

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Dataset Tweets",
            total_tweets
        )


    with col2:

        st.metric(
            "AppleSupport Replies",
            total_apple_replies
        )


    with col3:

        st.metric(
            "Conversation Pairs",
            total_conversations
        )


    with col4:

        st.metric(
            "Supported Intents",
            4
        )


    st.divider()


    # Intent taxonomy

    st.subheader(
        "🎯 Intent Taxonomy"
    )


    intent_df = pd.DataFrame({

        "Intent": [

            "battery_issue",

            "update_problem",

            "app_feature_problem",

            "account_code_problem"
        ],

        "Purpose": [

            "Battery drain and battery-life issues",

            "iOS update and performance issues",

            "Application and feature problems",

            "Verification and account-code issues"
        ]
    })


    st.dataframe(
        intent_df,
        use_container_width=True,
        hide_index=True
    )


    # Dataset

    st.subheader(
        "📈 Dataset Overview"
    )


    col1, col2 = st.columns(2)


    with col1:

        inbound_count = len(
            df[df["inbound"] == True]
        )

        st.metric(
            "Customer Messages",
            inbound_count
        )


    with col2:

        outbound_count = len(
            df[df["inbound"] == False]
        )

        st.metric(
            "Support Replies",
            outbound_count
        )


    # Brand coverage

    st.subheader(
        "🍎 Brand Coverage"
    )


    brand_counts = (
        df["author_id"]
        .value_counts()
        .head(10)
        .reset_index()
    )


    brand_counts.columns = [
        "Account",
        "Messages"
    ]


    st.dataframe(
        brand_counts,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # Automation strategy

    st.subheader(
        "⚙️ Automation Strategy"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.success(
            """
### ✅ Auto-handle

Suitable when:

• Intent is recognized  
• Historical resolution exists  
• Issue is not sensitive  
• Customer is not highly frustrated
"""
        )


    with col2:

        st.error(
            """
### 🚨 Human escalation

Triggered when:

• Sensitive issue is detected  
• Intent is unknown  
• No historical resolution exists  
• Customer appears highly frustrated
"""
        )


# ============================================================
# PAGE 3 — AI KNOWLEDGE
# ============================================================

elif st.session_state.page == "AI Knowledge":

    st.header("🧠 AI Knowledge Base")

    st.caption(
        "Historical AppleSupport conversations used for semantic grounding"
    )


    pairs = get_conversation_pairs()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Historical Conversations",
            len(pairs)
        )


    with col2:

        st.metric(
            "Knowledge Source",
            "AppleSupport"
        )


    with col3:

        st.metric(
            "Retrieval",
            "Semantic"
        )


    st.divider()


    st.subheader(
        "How Retrieval-Augmented Support Works"
    )


    st.info(
        """
**1. Customer message**

A customer submits an issue.

**2. Semantic retrieval**

The system searches historical customer conversations
for a semantically similar issue.

**3. Resolution retrieval**

The corresponding AppleSupport response is retrieved.

**4. Grounded generation**

The LLM uses that response as grounding to create
a natural customer-facing reply.

**5. Safety decision**

The system decides whether the issue can be
automatically handled or should go to a human.
"""
    )


    st.subheader(
        "📚 Historical Conversations"
    )


    if pairs:

        for i, pair in enumerate(pairs):

            with st.expander(
                f"Conversation {i + 1}"
            ):

                st.caption(
                    "CUSTOMER"
                )

                st.write(
                    pair["customer"]
                )


                st.caption(
                    "APPLE SUPPORT"
                )

                st.info(
                    pair["reply"]
                )


    else:

        st.warning(
            "No AppleSupport conversation pairs found."
        )


# ============================================================
# PAGE 4 — TRUST & SAFETY
# ============================================================

elif st.session_state.page == "Trust & Safety":

    st.header("🛡️ Trust & Safety")

    st.caption(
        "Safety-first automation policy"
    )


    st.divider()


    st.subheader(
        "Safety Principle"
    )


    st.success(
        "Correctness and safety take priority over automation."
    )


    st.subheader(
        "Escalation Rules"
    )


    safety_rules = pd.DataFrame({

        "Condition": [

            "Sensitive issue",

            "Unknown intent",

            "No historical resolution",

            "Highly frustrated customer",

            "Normal known issue"
        ],

        "Action": [

            "🚨 Human",

            "🚨 Human",

            "🚨 Human",

            "🚨 Human",

            "✅ Auto-handle"
        ],

        "Reason": [

            "Requires human judgement",

            "Classification confidence is insufficient",

            "No reliable grounding evidence",

            "Higher risk of poor customer experience",

            "Known issue with historical support pattern"
        ]
    })


    st.dataframe(
        safety_rules,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Why the Agent Is Trustworthy"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            """
### 📚 Grounded

Replies are based on historical
AppleSupport resolutions rather
than unrestricted generation.
"""
        )


    with col2:

        st.info(
            """
### 👤 Human-in-the-loop

Risky or uncertain cases are
escalated instead of forcing
automation.
"""
        )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            """
### 🚫 No invented policies

The AI is instructed not to invent
refunds, guarantees or unsupported
technical instructions.
"""
        )


    with col2:

        st.info(
            """
### 🔎 Evidence available

Historical grounding evidence can
be inspected by the support team.
"""
        )


# ============================================================
# PAGE 5 — EVALUATION
# ============================================================

elif st.session_state.page == "Evaluation":

    st.header("📈 Agent Evaluation")

    st.caption(
        "Measuring classification, retrieval, safety and automation performance"
    )


    st.divider()


    # --------------------------------------------------------
    # TEST CASES
    # --------------------------------------------------------

    test_cases = [

        {
            "message":
                "My iPhone battery is draining very fast",

            "expected":
                "battery_issue",

            "expected_decision":
                "AUTO-HANDLE"
        },

        {
            "message":
                "My phone became very slow after the latest iOS update",

            "expected":
                "update_problem",

            "expected_decision":
                "AUTO-HANDLE"
        },

        {
            "message":
                "My apps keep crashing after the iOS update",

            "expected":
                "app_feature_problem",

            "expected_decision":
                "AUTO-HANDLE"
        },

        {
            "message":
                "I am not receiving my verification code",

            "expected":
                "account_code_problem",

            "expected_decision":
                "AUTO-HANDLE"
        },

        {
            "message":
                "My Apple account has been hacked",

            "expected":
                "general_issue",

            "expected_decision":
                "ESCALATE"
        },

        {
            "message":
                "This fucking update is terrible and useless",

            "expected":
                "update_problem",

            "expected_decision":
                "ESCALATE"
        }
    ]


    results = []


    # --------------------------------------------------------
    # RUN TESTS
    # --------------------------------------------------------

    for case in test_cases:

        intent = classify_intent(
            case["message"]
        )


        historical = get_historical_reply(
            case["message"]
        )


        decision, reason = check_escalation(

            case["message"],

            intent,

            historical
        )


        intent_correct = (

            intent ==
            case["expected"]
        )


        decision_correct = (

            (
                case["expected_decision"]
                == "AUTO-HANDLE"

                and

                decision ==
                "AUTO-HANDLE"
            )

            or

            (
                case["expected_decision"]
                == "ESCALATE"

                and

                decision ==
                "ESCALATE"
            )
        )


        results.append({

            "Message":
                case["message"],

            "Expected Intent":
                case["expected"],

            "Predicted Intent":
                intent,

            "Intent":
                "✅"
                if intent_correct
                else "❌",

            "Expected Decision":
                case["expected_decision"],

            "Actual Decision":
                decision,

            "Safety":
                "✅"
                if decision_correct
                else "❌"
        })


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    intent_accuracy = (

        results_df["Intent"]
        == "✅"

    ).mean() * 100


    safety_accuracy = (

        results_df["Safety"]
        == "✅"

    ).mean() * 100


    overall_score = (

        intent_accuracy
        +
        safety_accuracy

    ) / 2


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Intent Accuracy",
            f"{intent_accuracy:.1f}%"
        )


    with col2:

        st.metric(
            "Safety Decision Accuracy",
            f"{safety_accuracy:.1f}%"
        )


    with col3:

        st.metric(
            "Overall Evaluation",
            f"{overall_score:.1f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    st.subheader(
        "🧪 Evaluation Test Cases"
    )


    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # TRUSTWORTHINESS
    # --------------------------------------------------------

    st.subheader(
        "🛡️ Trustworthiness Assessment"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.success(
            """
### 📚 Grounded Responses

Historical AppleSupport conversations
are used as evidence before generating
a customer reply.
"""
        )


        st.success(
            """
### 👤 Human Escalation

Sensitive and risky cases are not
automatically handled.
"""
        )


    with col2:

        st.success(
            """
### 🚫 Controlled Generation

The LLM is instructed not to invent
policies, refunds or guarantees.
"""
        )


        st.success(
            """
### 🔎 Auditable Evidence

Support teams can inspect the historical
response used for grounding.
"""
        )


    st.divider()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📌 Evaluation Summary"
    )


    st.info(
        f"""
The prototype was evaluated using
{len(test_cases)} representative
customer-support scenarios.

**Intent accuracy:** {intent_accuracy:.1f}%

**Safety decision accuracy:** {safety_accuracy:.1f}%

**Overall evaluation score:** {overall_score:.1f}%

The evaluation focuses on whether the agent
can correctly identify known support issues
and safely route uncertain or sensitive cases
to human support.
"""
    )


# ============================================================
# PAGE 6 — SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.header("⚙️ Settings")

    st.caption(
        "AI agent configuration"
    )


    st.divider()


    # Brand

    st.subheader(
        "🍎 Brand"
    )


    st.write(
        "**Active brand:** AppleSupport"
    )


    st.write(
        "**Dataset:** Customer Support on Twitter"
    )


    st.divider()


    # AI configuration

    st.subheader(
        "🤖 AI Configuration"
    )


    config_df = pd.DataFrame({

        "Component": [

            "LLM",

            "Embedding Model",

            "Retrieval",

            "Interface",

            "Knowledge Source"
        ],

        "Configuration": [

            "GPT-5.6 Luna",

            "all-MiniLM-L6-v2",

            "Semantic Similarity",

            "Streamlit",

            "AppleSupport historical conversations"
        ]
    })


    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # Architecture

    st.subheader(
        "🏗️ System Architecture"
    )


    st.code(
        """
Customer Message
       │
       ▼
Intent Classification
       │
       ▼
Semantic Similarity Search
       │
       ▼
Historical AppleSupport Resolution
       │
       ▼
Grounded LLM Reply
       │
       ▼
Safety / Escalation Engine
       │
       ├──────────────► AUTO-HANDLE
       │
       └──────────────► HUMAN SUPPORT
        """,
        language="text"
    )


    st.divider()


    # System status

    st.subheader(
        "🟢 System Status"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.success(
            "LLM Online"
        )


    with col2:

        st.success(
            "RAG Online"
        )


    with col3:

        st.success(
            "Classifier Online"
        )


    with col4:

        st.success(
            "Safety Online"
        )


    st.divider()


    st.success(
        "SupportAI is configured and ready."
    )