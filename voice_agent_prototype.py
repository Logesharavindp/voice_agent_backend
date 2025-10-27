import json
import difflib
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIGURE AZURE OPENAI ----------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ---------- LOAD STATIC DATA ----------
with open("static/user.json", "r") as f:
    data = json.load(f)

users = data["users"]
company_list = data["company_list"]

# ---------- SYSTEM PROMPT FOR VOICE AGENT ----------
system_prompt = """
    You are an empathetic, professional, and conversational Employment Verification Voice Agent.

    Your responsibilities:
    1. Greet the user warmly by their first name and make them feel comfortable.
    2. Confirm their current employment information from our records.
    3. If the user's employment matches our records:
    - Thank them and confirm successful verification clearly.
    4. If the employment does not match:
    - Politely ask for their current company name.
    - Suggest a few possible company matches from a predefined list.
    - Allow them to either pick one or provide their company name manually.
    5. Maintain a friendly, confident, and professional tone throughout.

    Voice & Style Guidelines:
    - Keep responses short and natural (2–3 sentences maximum).
    - Use clear, spoken-language phrasing — avoid long or complex sentences.
    - Always refer to the user by their first name (e.g., “Hi John!”).
    - Be empathetic if there’s a mismatch or confusion (“No worries, we’ll sort that out together.”).
    - Confirm information before proceeding (“Just to confirm, you’re currently with Global Solutions Ltd, right?”).
    - Express gratitude at the end (“Thanks, John — your verification is complete!”).

    Example Flow:
    1. Greeting: “Hi John, it’s great to talk with you today!”
    2. Confirmation: “Our records show you work at Tech Innovations Inc — is that still correct?”
    3. Mismatch handling: “Got it. Can you please tell me your current company name?”
    4. Suggestion: “Did you mean one of these companies: Cloud Services International, Global Solutions Ltd, or AI Research Labs?”
    5. Completion: “Perfect, thank you John! Your details are now verified.”

    Stay conversational, calm, and supportive throughout the interaction.
    """

# ---------- HELPER FUNCTIONS ----------
def get_employment(email: str):
    """Return employment record if found."""
    return users.get(email)

def suggest_company_matches(user_input: str):
    """Return close company name matches."""
    return difflib.get_close_matches(user_input, company_list, n=5, cutoff=0.4)

def chat_response(conversation_history: list):
    """Use Azure OpenAI to respond conversationally with full conversation history."""
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    completion = client.chat.completions.create(
        model="gpt-4-0613",  # replace with your deployed model name
        messages=messages
    )
    return completion.choices[0].message.content.strip()

# ---------- MAIN CHATBOT LOGIC ----------
def main():
    print("👋 Welcome to the Employment Verification Chatbot!\n")

    # Step 1: Collect user details
    print("Let's start by collecting some information about you.\n")

    user_name = input("Please enter your full name: ").strip()
    years_of_experience = input("How many years of experience do you have? ").strip()
    date_of_birth = input("What is your date of birth? (e.g., DD/MM/YYYY): ").strip()

    print("\nThank you! Let me confirm your details:")
    print(f"Name: {user_name}")
    print(f"Years of Experience: {years_of_experience}")
    print(f"Date of Birth: {date_of_birth}\n")

    # Extract first name for personalization
    first_name = user_name.split()[0]

    # Step 2: Get email to check employment records
    user_email = input("Please enter your email address: ").strip()
    user_data = get_employment(user_email)

    # Initialize conversation history
    conversation_history = []

    # Step 3: Greet the user by name
    conversation_history.append({
        "role": "user",
        "content": f"Greet {first_name} warmly and introduce yourself as an employment verification agent."
    })
    response = chat_response(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"\nAgent: {response}\n")

    # Step 4: Check employment records and confirm
    if not user_data:
        print(f"Agent: I couldn't find any employment records for {user_email} in our system.")
        print(f"Agent: Let me help you add your employment information.\n")

        # Ask for company name
        company_input = input("You: Please tell me your current company name: ").strip()
        matches = suggest_company_matches(company_input)

        if matches:
            print(f"\nAgent: I found some possible matches:")
            for i, m in enumerate(matches, start=1):
                print(f"{i}. {m}")

            choice = input("\nYou: Please select a number or type your company name if it's not listed: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                selected = matches[int(choice)-1]
                print(f"\nAgent: Great! I've recorded your company as {selected}. Thank you {first_name}, your verification is complete!")
            else:
                print(f"\nAgent: Thank you! I've recorded your company as {choice}. Your verification is complete, {first_name}!")
        else:
            print(f"\nAgent: Thank you! I've recorded your company as {company_input}. Your verification is complete, {first_name}!")
        return

    # Step 5: Confirm employment from dummy data
    conversation_history.append({
        "role": "user",
        "content": f"Tell {first_name} that our records show they work at {user_data['company_name']} and ask if that's still correct."
    })
    response = chat_response(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"Agent: {response}\n")

    # Step 6: Interactive conversation loop for employment verification
    verified = False
    while not verified:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        # Check if user is confirming the company
        if user_input.lower() in ["yes", "yeah", "yep", "correct", "right", "that's right", "that's correct", "yes it is"]:
            success_message = f"Perfect! Thank you {first_name}, your employment verification is complete. Have a great day!"
            conversation_history.append({"role": "assistant", "content": success_message})
            print(f"\nAgent: {success_message}\n")
            verified = True
            break

        # Check if user is denying or providing a different company
        elif user_input.lower() in ["no", "nope", "not correct", "wrong", "incorrect", "no it's not"]:
            ask_company_message = f"No worries, {first_name}! Could you please tell me your current company name?"
            conversation_history.append({"role": "assistant", "content": ask_company_message})
            print(f"\nAgent: {ask_company_message}\n")

            # Get the company name
            company_input = input("You: ").strip()
            conversation_history.append({"role": "user", "content": company_input})

            # Try to find matches
            matches = suggest_company_matches(company_input)

            if matches:
                print(f"\nAgent: I found some possible matches:")
                for i, m in enumerate(matches, start=1):
                    print(f"{i}. {m}")

                match_message = "Please select a number or type your company name if it's not listed."
                conversation_history.append({"role": "assistant", "content": match_message})
                print(f"\n{match_message}\n")

                choice = input("You: ").strip()
                conversation_history.append({"role": "user", "content": choice})

                if choice.isdigit() and 1 <= int(choice) <= len(matches):
                    selected = matches[int(choice)-1]
                    update_message = f"Great! I've updated your company to {selected}. Thank you {first_name}, your verification is complete!"
                    conversation_history.append({"role": "assistant", "content": update_message})
                    print(f"\nAgent: {update_message}\n")
                    verified = True
                else:
                    record_message = f"Thank you! I've recorded your company as {choice}. Your verification is complete, {first_name}!"
                    conversation_history.append({"role": "assistant", "content": record_message})
                    print(f"\nAgent: {record_message}\n")
                    verified = True
            else:
                no_match_message = f"Thank you! I've recorded your company as {company_input}. Your verification is complete, {first_name}!"
                conversation_history.append({"role": "assistant", "content": no_match_message})
                print(f"\nAgent: {no_match_message}\n")
                verified = True
        else:
            # For any other input, use AI to respond naturally
            response = chat_response(conversation_history)
            conversation_history.append({"role": "assistant", "content": response})
            print(f"\nAgent: {response}\n")

# ---------- RUN ----------
if __name__ == "__main__":
    main()
