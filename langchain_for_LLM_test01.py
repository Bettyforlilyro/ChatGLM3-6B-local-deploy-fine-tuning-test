import time
import logging
import requests
from typing import Optional, List, Dict, Mapping, Any
import json
import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
import config.data_config as data_config

logging.basicConfig(level=logging.INFO)
# 启用LLM的缓存
langchain.llm_cache = InMemoryCache()


class GoChatGLM(LLM):
    # 模型服务URL，这里地址一定要与开启GLM服务的URL地址相同
    url: str = "http://" + data_config.host_ip + ":" + str(data_config.host_port) + "/chat"
    history: List[Dict] = []

    @property
    def _llm_type(self) -> str:
        return 'ChatGLM'

    def _construct_query(self, prompt: str) -> Dict:
        """
        Constructs a query for chat.
        :param prompt:
        :return:
        """
        query = {"query": prompt, "history": self.history}
        return query

    def _post(self, url: str, query: Dict) -> Any:
        """
        Sends a POST request to chat.
        :param url:
        :param query:
        :return:
        """
        # 发送和接受的数据格式均使用json
        response = requests.post(url, data=json.dumps(query)).json()
        return response

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Calls a chat.
        :param prompt:
        :param stop:
        :return:
        """
        # construct a query
        query = self._construct_query(prompt)

        # post
        response = self._post(self.url, query)
        response_chat = response['response']
        self.history = response['history']
        return response_chat

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters for the chat.
        :return:
        """
        _param_dict = {
            "url": self.url,
        }
        return _param_dict


if __name__ == '__main__':
    llm = GoChatGLM()
    while True:
        human_input = input('Human > ')
        begin_time = time.time()
        # 请求模型
        response = llm(human_input)
        end_time = time.time()
        user_response_time = end_time - begin_time
        logging.info(f'ChatGLM process time: {user_response_time}ms')
        print(f'ChatGLM: {response}')