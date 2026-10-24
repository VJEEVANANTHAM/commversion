"""FAQ RAG. Chroma + local sentence-transformers embeddings; lexical fallback if extras unavailable."""
from pathlib import Path
import pandas as pd
import re
from difflib import SequenceMatcher

class FAQRetriever:
    def __init__(self, data_dir="data", llm=None):
        self.df=pd.read_csv(Path(data_dir)/"faq.csv").fillna("")
        self.store=None
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_core.documents import Document
            from config import Settings
            s=Settings()
            embeddings=HuggingFaceEmbeddings(model_name=s.embedding_model)
            docs=[Document(page_content=f"Question: {r.question}\nAnswer: {r.answer}",
                           metadata={"faq_id":r.faq_id,"category":r.category})
                  for r in self.df.itertuples()]
            self.store=Chroma.from_documents(docs, embeddings, collection_name="ashcombe_faq")
        except Exception as exc:
            print(f"[FAQ vector store unavailable; lexical retrieval enabled: {exc}]")
        self.llm=llm

    def retrieve(self, question, k=3):
        if self.store:
            try: return [d.page_content for d in self.store.similarity_search(question,k=k)]
            except Exception: pass
        def score(row):
            q=(str(row["question"])+" "+str(row["category"])).lower()
            words=set(re.findall(r"[a-z0-9]+",question.lower()))
            other=set(re.findall(r"[a-z0-9]+",q))
            overlap=len(words&other)/max(1,len(words))
            return overlap + .25*SequenceMatcher(None,question.lower(),q).ratio()
        ranked=sorted(((score(r),r) for _,r in self.df.iterrows()), key=lambda x:x[0], reverse=True)
        return [f"Question: {r['question']}\nAnswer: {r['answer']}" for s,r in ranked[:k] if s>0]

    def answer(self, question):
        docs=self.retrieve(question)
        if not docs: return None
        if self.llm:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                response=self.llm.invoke([
                    SystemMessage(content="Answer only from the supplied Ashcombe & Rowe FAQ snippets. If they do not answer the question, say you do not have that information. Never invent details."),
                    HumanMessage(content=f"Question: {question}\nFAQ snippets:\n"+"\n---\n".join(docs))
                ])
                answer=str(response.content).strip()
                if answer and "do not have that information" not in answer.lower() and "don't have that information" not in answer.lower():
                    return answer
            except Exception: pass
        # Return the best matching stored answer, never synthesize unsupported facts.
        best=docs[0]
        m=re.search(r"Answer:\s*(.*)",best,re.S)
        return m.group(1).strip() if m else None
