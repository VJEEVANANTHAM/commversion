"""Stateful lead qualification chatbot orchestrated with LangGraph."""
from typing import TypedDict, Any
import re
try:
    from langgraph.graph import StateGraph, END
except ImportError:  # lightweight fallback for local tests before dependencies are installed
    END = "__end__"
    class StateGraph:
        def __init__(self, state_type): self.nodes={}; self.entry=None
        def add_node(self,name,fn): self.nodes[name]=fn
        def set_entry_point(self,name): self.entry=name
        def add_edge(self,source,target): pass
        def compile(self):
            nodes=self.nodes; entry=self.entry
            class Graph:
                def invoke(self,state): return nodes[entry](state)
            return Graph()
from config import Settings
from data_access import PropertyData, normalize_postcode
from rag import FAQRetriever
from llm import make_llm

OFFICE="020 7946 0100"
M1=f"Sorry, we don't cater to new-build properties under £1 million. Our office number is {OFFICE}. Goodbye."
M2=f"Sorry, we don't cater to the postcode you provided. Our office number is {OFFICE}. Goodbye."
M3=f"Sorry, we have no available homes matching your requirements right now. Our office number is {OFFICE}. Goodbye."

class ChatState(TypedDict, total=False):
    message: str
    response: str
    session: dict

def parse_budget(text):
    """Parse £1.5m, 1,200,000, 1.2 million, 950k and common variants."""
    s=str(text).lower().replace(",","").replace("£","").strip()
    m=re.search(r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?",s)
    if not m: return None
    n=float(m.group(1)); unit=m.group(2) or ""
    if unit in ("m","million"): n*=1_000_000
    elif unit in ("k","thousand"): n*=1_000
    return int(n) if n>0 else None

def valid_email(s): return bool(re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",s.strip()))
def valid_phone(s):
    digits=re.sub(r"\D","",s)
    return 10 <= len(digits) <= 15

def valid_uk_postcode(s):
    s=re.sub(r"\s+","",s).upper()
    return bool(re.fullmatch(r"(GIR0AA|[A-Z]{1,2}\d[A-Z\d]?\d[A-Z]{2})",s))

class LeadChatbot:
    def __init__(self, data_dir=None):
        self.data_dir=data_dir or Settings().data_dir
        self.llm=make_llm()
        self.data=PropertyData(self.data_dir)
        self.faq=FAQRetriever(self.data_dir,self.llm)
        self.sessions={}
        graph=StateGraph(ChatState)
        graph.add_node("process_turn",self._graph_process)
        graph.set_entry_point("process_turn")
        graph.add_edge("process_turn",END)
        self.graph=graph.compile()

    def new_session(self):
        return {"step":"name","name":None,"journey":None,"category":None,"budget":None,
                "postcode":None,"property_query":None,"property_retry":False,"property_id":None,
                "phone":None,"email":None,"qualified":False,"finished":False,"history":[]}

    def reply(self, session_id, message):
        session=self.sessions.setdefault(session_id,self.new_session())
        result=self.graph.invoke({"message":message,"session":session})
        self.sessions[session_id]=result["session"]
        return result["response"]

    def _graph_process(self,state):
        session=state["session"]
        response=self._process(session,state["message"])
        session["history"].append({"user":state["message"],"assistant":response})
        return {"session":session,"response":response}

    def _faq_side_question(self,msg):
        m=msg.lower()
        return ("?" in msg or any(x in m for x in ["opening hours","head office","branch","fee","fees","who are","what is your","how do i","how does","phone number","contact number","process","address","where are you","tell me about"]))

    def _intent(self,msg):
        m=msg.lower()
        if re.search(r"\b(sell|selling|seller|put .* on the market)\b",m): return "sell"
        if re.search(r"\b(buy|buying|buyer|purchase|looking for a home|looking to buy)\b",m): return "buy"
        return None

    def _answer_faq(self,msg):
        answer=self.faq.answer(msg)
        if answer: return answer
        return f"I couldn't find that information in our records. Please call our office on {OFFICE}."

    def _property_card(self,row):
        return (f"{row['property_name']} — {row['bedrooms']} bedrooms, "
                f"{self.data.money(row['price_gbp'])}.")

    def _show_matches(self,rows):
        return "Here are up to 5 available matches:\n" + "\n".join(f"• {self._property_card(r)}" for _,r in rows.iterrows()) + "\nWould you like us to contact you?"

    def _qualify_buyer(self,s,category,postcode,budget=None):
        rows=self.data.at_postcode(postcode,category)
        if rows.empty: return M2
        available=rows[rows.status_norm=="available"]
        if available.empty: return M3
        if category=="new_build":
            matches=self.data.available_matches(postcode,category,budget)
            if matches.empty: return M3
        else: matches=available.sort_values("price_gbp").head(5)
        s.update({"category":category,"postcode":postcode,"qualified":True,"step":"phone"})
        return self._show_matches(matches)+"\nTo arrange a call-back, what is your phone number?"

    def _process(self,s,msg):
        raw=msg.strip(); m=raw.lower()
        if s.get("finished"):
            s.update(self.new_session())
            return "Hello! How may I help you? May I have your name?"
        if s["step"]=="name" and not s.get("name"):
            if not raw: return "Hello! How may I help you? May I have your name?"
            s["name"]=raw
            s["step"]="intent"
            return f"Thanks, {s['name']}. Are you looking to buy or sell a property?"
        # Property questions can be handled at any point without losing the current step.
        if s["step"] not in ("property_name","property_retry") and any(
            token in m for token in ("property", "bedroom", "bedrooms", "price of", "how much", "is it available", "is the apartment", "is the house")
        ):
            matched=None
            for _, candidate in self.data.properties.iterrows():
                name=str(candidate["property_name"]).lower()
                address=str(candidate["address"]).lower()
                if (name and name in m) or (address and address in m) or str(candidate["property_id"]).lower() in m:
                    matched=candidate
                    break
            if matched is not None:
                status=str(matched["status"]).replace("_"," ")
                answer=(f"{matched['property_name']} is {status}. It is a {matched['property_type']} "
                        f"with {matched['bedrooms']} bedrooms, listed at {self.data.money(matched['price_gbp'])}.")
                return answer + "\\n\\n" + self._prompt_for_step(s)
        # FAQ questions can be handled at any point without losing the current step.
        if self._faq_side_question(raw) and s["step"] not in ("property_name","property_retry"):
            answer=self._answer_faq(raw)
            return f"{answer}\n\nNow, {s.get('name') or 'there'}, "+self._prompt_for_step(s)
        intent=self._intent(raw)
        if intent and s.get("journey") and intent!=s["journey"] and s["step"] not in ("name","intent"):
            s.update({"journey":intent,"category":None,"budget":None,"postcode":None,"property_query":None,
                      "property_retry":False,"property_id":None,"qualified":False,"step":"postcode" if intent=="sell" else "specific"})
            return "Of course, let's switch to selling. What is the property's postcode?" if intent=="sell" else "Of course, let's switch to buying. Do you have a specific property in mind? (yes/no)"
        step=s["step"]
        if step=="intent":
            i=self._intent(raw)
            if i: s["journey"]=i; s["step"]="postcode" if i=="sell" else "specific"
            else:
                if "sell" in m: s["journey"]="sell"; s["step"]="postcode"
                elif "buy" in m: s["journey"]="buy"; s["step"]="specific"
                else: return "Are you looking to buy or sell a property?"
            return "What is the property's postcode?" if s["journey"]=="sell" else "Do you have a specific property in mind? (yes/no)"
        if step=="specific":
            if m in ("yes","y","yeah","yep") or "specific property" in m or "property name" in m:
                s["step"]="property_name"; return "Which property name or address are you interested in?"
            if m in ("no","n","nope","not really"): s["step"]="category"; return "Are you looking for a new-build or a resale home?"
            if any(x in m for x in ["new-build","new build","resale"]):
                s["category"]="new_build" if "new" in m else "resale"; s["step"]="budget"
                return "What is your budget?"
            return "Do you have a specific property in mind? Please answer yes or no."
        if step in ("property_name","property_retry"):
            found=self.data.find_property(raw)
            if len(found)==0:
                if not s["property_retry"]:
                    s["property_retry"]=True; s["step"]="property_retry"
                    return "I couldn't find that property. Could you provide its name or full address once more?"
                s["step"]="category"; s["property_retry"]=False
                return "I still couldn't find it. Are you looking for a new-build or a resale home?"
            if len(found)>1:
                s["step"]="property_name"
                return "I found multiple properties. Which one do you mean?\n" + "\n".join(f"• {r['property_name']} — {r['address']} ({r['postcode']})" for _,r in found.head(5).iterrows())
            r=found.iloc[0]; s["property_id"]=r["property_id"]; s["category"]=r["listing_category"]
            if r["status_norm"]=="available":
                s.update({"qualified":True,"step":"phone","postcode":r["postcode"]})
                return f"That property is available: {r['property_type']}, {r['bedrooms']} bedrooms, {self.data.money(r['price_gbp'])}. What is your phone number so we can arrange a call-back?"
            s["step"]="sold_property_type"
            return f"That property is {r['status'].replace('_',' ')}. I can help find a similar home. Would you prefer a new-build or resale?"
        if step=="sold_property_type":
            if "new" in m: s["category"]="new_build"; s["step"]="budget"; return "What is your budget?"
            if "resale" in m or "existing" in m: s["category"]="resale"; s["step"]="postcode"; return "What postcode are you interested in?"
            return "Would you prefer a new-build or resale?"
        if step=="category":
            if "new" in m: s["category"]="new_build"; s["step"]="budget"; return "What is your budget?"
            if "resale" in m or "existing" in m: s["category"]="resale"; s["step"]="budget"; return "What is your budget?"
            return "Are you looking for a new-build or a resale home?"
        if step=="budget":
            b=parse_budget(raw)
            if b is None: return "Please enter your budget, for example £1.2 million, £1.5m, £950k or 1,200,000."
            s["budget"]=b
            if s["category"]=="new_build" and b<1_000_000: s["step"]="done"; s["finished"]=True; return M1
            s["step"]="postcode"
            return "What postcode are you interested in?"
        if step=="postcode":
            if not valid_uk_postcode(raw): return "That doesn't look like a valid UK postcode. Please enter it again, for example GU5 7AQ."
            s["postcode"]=normalize_postcode(raw)
            if s["journey"]=="sell":
                if not self.data.is_seller_covered(raw): s["step"]="done"; s["finished"]=True; return M2
                s.update({"qualified":True,"step":"phone"})
                return "Great, we cover that postcode. What is your phone number?"
            return self._qualify_buyer(s,s["category"],raw,s.get("budget"))
        if step=="phone":
            if not valid_phone(raw): return "Please enter a valid phone number with 10 to 15 digits."
            s["phone"]=raw; s["step"]="email"; return "Thank you. What is your email address?"
        if step=="email":
            if not valid_email(raw): return "Please enter a valid email address, such as name@example.com."
            s["email"]=raw; s["step"]="close"
            return "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?"
        if step=="close":
            if m in ("yes","y","yeah","yes please","sure") or "help" in m:
                s["step"]="intent"; s["journey"]=None
                return "How can I help you?"
            if m in ("no","n","no thanks","no thank you","that's all","that is all"):
                s["step"]="done"; s["finished"]=True
                return "Thank you for chatting with us. Goodbye."
            return "Is there anything else I can help you with? Please answer yes or no."
        return self._prompt_for_step(s)

    def _prompt_for_step(self,s):
        return {"name":"May I have your name?","intent":"Are you looking to buy or sell a property?",
                "specific":"Do you have a specific property in mind? (yes/no)","property_name":"Which property name or address are you interested in?",
                "category":"Are you looking for a new-build or resale home?","budget":"What is your budget?",
                "postcode":"What postcode are you interested in?","phone":"What is your phone number?",
                "email":"What is your email address?","close":"Is there anything else I can help you with?"}.get(s["step"],"How can I help you?")
