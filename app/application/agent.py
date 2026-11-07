"""A bounded adaptive RAG graph: retrieve, assess evidence, optionally broaden, answer."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app.domain.retrieval import RetrievalResult
from app.domain.generation import AnswerResult


class AgentState(TypedDict, total=False):
    question: str
    options: dict
    retrieval: RetrievalResult
    attempts: int
    result: AnswerResult


class AgenticAnswerService:
    def __init__(self, retrieval_service, answer_service):
        self.retrieval = retrieval_service
        self.answer_service = answer_service
        graph = StateGraph(AgentState)
        graph.add_node('retrieve', self._retrieve)
        graph.add_node('broaden', self._broaden)
        graph.add_node('answer', self._answer)
        graph.add_edge(START, 'retrieve')
        graph.add_conditional_edges('retrieve', self._route, {'broaden': 'broaden', 'answer': 'answer'})
        graph.add_edge('broaden', 'answer')
        graph.add_edge('answer', END)
        self.graph = graph.compile()

    def _retrieve(self, state):
        return {'retrieval': self.retrieval.retrieve(state['question'], **state['options']), 'attempts': 1}

    def _route(self, state):
        return 'answer' if state['retrieval'].has_results else 'broaden'

    def _broaden(self, state):
        options = dict(state['options'])
        options['top_k'] = min(50, options.get('top_k', 5) * 2)
        # Relax relevance only; retain document scope to prevent cross-document leakage.
        options['min_similarity'] = None
        return {'retrieval': self.retrieval.retrieve(state['question'], **options), 'attempts': 2}

    def _answer(self, state):
        return {'result': self.answer_service.answer_retrieval(state['retrieval'])}

    def answer(self, question, **options):
        state = self.graph.invoke({'question': question, 'options': options}, {'recursion_limit': 8})
        return state['result']
