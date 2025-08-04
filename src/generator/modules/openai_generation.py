from langchain_openai import ChatOpenAI

def get_llm_openai(
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = None,
):
    common_args = dict(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    if model_name.startswith("gpt-4o") and max_tokens is not None:
        return ChatOpenAI(**common_args, max_completion_tokens=max_tokens)
    elif max_tokens is not None:
        return ChatOpenAI(**common_args, max_tokens=max_tokens)
    else:
        return ChatOpenAI(**common_args)

# 테스트
if __name__ == "__main__":
    llm = get_llm_openai(model_name="gpt-4o")
    print(llm.invoke("안녕하세요! 당신은 누구인가요?"))
