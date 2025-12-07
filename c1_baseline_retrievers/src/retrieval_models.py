

from local_retrievers import BM25Retriever, DenseRetriever, RerankRetriever


class BasicRetrievalModel:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        
        # --- Retrievers 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)
        else:
            raise ValueError(f"Unknown retriever name: {args.retriever_name}")
    
    def retrieve(self, query, top_k=3):
        pass


class SingleStageRetrievalModel(BasicRetrievalModel):
    def __init__(self, device, args):
        super().__init__(device, args)
    
    def retrieve(self, query, top_k=3):
        return self.retriever.search(query, top_k)
    

# TODO: ReAct as a retriever model
# Src: https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb
class ReActRetrievalModel(BasicRetrievalModel):
    def __init__(self, device, args):
        super().__init__(device, args)
    
    def retrieve(self, query, top_k=3):
        pass


# TODO: LLM4CS & MQ4CS: 
#   - Generating Multi-Aspect Queries for Conversational Search: https://arxiv.org/pdf/2403.19302
#   - LLM4CS: Large language models know your contextual search intent: A prompting framework for conversational search.: https://aclanthology.org/2023.findings-emnlp.86/

# TODO: RMIT–ADM+S at the SIGIR 2025 LiveRAG Challenge
#  - Paper (report): https://arxiv.org/pdf/2506.14516
#  - Code: https://github.com/rmit-ir/G-RAG-LiveRAG

