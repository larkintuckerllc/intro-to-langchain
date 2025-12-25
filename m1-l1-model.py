from pprint import pprint

from langchain.chat_models import init_chat_model

def main():
    model = init_chat_model(model="gpt-5-nano")
    response = model.invoke("What's the capital of the Moon?")
    print(response.content)
    pprint(response.response_metadata)


if __name__ == "__main__":
    main()
