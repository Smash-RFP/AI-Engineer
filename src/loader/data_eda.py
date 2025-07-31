import os

def check_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print("설정")
    else:
        print("설정 안됨")
        
check_api_keys()

# if __name__ == "__main__":
#     check_api_keys()