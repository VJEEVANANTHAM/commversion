"""Run the chatbot in a terminal."""
from chatbot import LeadChatbot

def main():
    bot=LeadChatbot()
    sid="cli"
    print("Ashcombe & Rowe Estates: Hello! How may I help you? May I have your name?")
    while True:
        try: msg=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\\nGoodbye."); break
        if msg.lower() in {"/quit","/exit"}: print("Goodbye."); break
        print("Assistant:",bot.reply(sid,msg))

if __name__=="__main__": main()
