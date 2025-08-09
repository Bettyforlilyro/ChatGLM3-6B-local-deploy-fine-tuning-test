import inspect
import traceback
from copy import deepcopy
from types import GenericAlias
from typing import get_origin, Annotated

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}


# 注意该函数并没有实际调用回调函数
def register_tool(func: callable):
    tool_name = func.__name__
    tools_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter {name} must be annotated with type {annotation}")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for parameter {name} must be typing.Annotated")
        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for {name} must be a str")
        if not isinstance(required, bool):
            raise TypeError(f"Required for {name} must be a bool")
        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = {
        "name": tool_name,
        "description": tools_description,
        "params": tool_params,
    }
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tools_description
    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool {tool_name} not registered, please register first."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)
    except:
        ret = traceback.format_exc()
    ret = str(ret)
    return ret     # 返回的是一个字符串，内容格式是一个字典


def get_tools() -> dict:
    return deepcopy(_TOOL_DESCRIPTIONS)


@register_tool
def get_city_weather(city_name: Annotated[str, 'The name of the city to be queried', True],
                     **kwargs: Annotated[str, 'Parameters', False]) -> str:
    """
    Get the current weather for 'city_name'.
    :param city_name:
    :param kwargs:
    :return:
    """
    if not isinstance(city_name, str):      # 城市名称必须是字符串类型
        raise TypeError('city_name must be str')
    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")   # 调用wttr.in的API获取指定城市的天气数据
        resp.raise_for_status()     # 如果返回的状态码不是200，抛出HTTPError异常
        resp = resp.json()
        # 将json格式数据解析成dict
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        ret = "Error encountered while fetching weather information!\n" + traceback.format_exc()
    return str(ret)


dispatch_tool("get_city_weather", {"city_name": "ShangHai"})
